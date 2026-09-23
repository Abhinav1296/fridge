"""Nutrition enrichment from Open Food Facts (Phase 2 — the "real data" layer).

The meal-plan :mod:`optimizer` can *report* a plan's calories and macros, but only for
ingredients it has nutrition data for. This module fills that cache with real figures
pulled from **Open Food Facts** (OFF) — a free, open, no-API-key food database — so the
optimizer's nutrition summary and the UI's nutrition dashboard have something to show.

Two lookup paths mirror the app's two input styles:

* :func:`fetch_by_name` — free-text search for a generic ingredient ("paneer", "tomato"),
  used to enrich the ingredient tokens already in the fridge; and
* :func:`fetch_by_barcode` — an exact product lookup for a scanned packaged good, which
  returns the product name/brand alongside its nutrition.

Both return **per-100 g** figures normalized to the same shape the nutrition cache and the
optimizer already speak (``kcal``, ``protein_g``, ``carbs_g``, ``fat_g``).

Like :mod:`optimizer`, :mod:`recommender`, :mod:`storage`, and :mod:`vision_service`, this
module is **framework-agnostic**: no Flask, no HTTP server. Network access goes out through
:mod:`requests` (so tests stub it), and the nutrition cache is reached through **injected
callables** (so tests need no database and the module never imports :mod:`storage`). The web
layer wires :func:`storage.get_nutrition` / :func:`storage.upsert_nutrition` in when it calls
:func:`enrich`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import requests

import recommender

logger = logging.getLogger(__name__)

# Provenance string stored on every cached row this module writes.
SOURCE = "openfoodfacts"

# Open Food Facts endpoints. Search is free-text; the product path is an exact barcode
# lookup. OFF asks every client to send a descriptive User-Agent so abusive traffic can be
# identified — we send one and keep request volume tiny by caching every result.
_OFF_BASE = "https://world.openfoodfacts.org"
_SEARCH_URL = f"{_OFF_BASE}/cgi/search.pl"
_PRODUCT_URL = _OFF_BASE + "/api/v2/product/{code}.json"
_USER_AGENT = "SmartFridge-ZeroWasteKitchen/1.0 (capstone project; contact via app)"

# (connect, read) timeouts, mirroring the agent's tuple. Nutrition is best-effort: a slow
# or unreachable OFF must fail fast and degrade, never hang a request.
_REQUEST_TIMEOUT: tuple[int, int] = (10, 30)

# Only the fields we actually use, to keep OFF payloads small.
_SEARCH_FIELDS = "product_name,nutriments,code,brands"
# How many search hits to consider before giving up on extracting usable nutriments.
_SEARCH_PAGE_SIZE = 5

# The per-100 g nutriment keys OFF uses, mapped to our canonical column names.
_NUTRIMENT_MAP = {
    "kcal": "energy-kcal_100g",
    "protein_g": "proteins_100g",
    "carbs_g": "carbohydrates_100g",
    "fat_g": "fat_100g",
}


class NutritionError(Exception):
    """A problem reaching or reading Open Food Facts (network, HTTP, or bad JSON)."""


# --- Public fetch API -------------------------------------------------------


def fetch_by_name(name: str) -> dict[str, Any] | None:
    """Look up per-100 g nutrition for a generic ingredient by name via OFF search.

    Searches Open Food Facts for ``name``, most-scanned products first, and returns the
    first hit that carries usable nutriments. Returns ``None`` when nothing matches or no
    hit has nutrition data (a normal, non-error outcome — the caller just skips it).

    Raises:
        NutritionError: If OFF can't be reached or returns something unparseable.
    """
    term = str(name or "").strip()
    if not term:
        return None
    params = {
        "search_terms": term,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": _SEARCH_PAGE_SIZE,
        "sort_by": "unique_scans_n",  # popularity — the "generic" product tends to win
        "fields": _SEARCH_FIELDS,
    }
    body = _get(_SEARCH_URL, params)
    products = body.get("products") if isinstance(body, dict) else None
    if not isinstance(products, list):
        return None
    for product in products:
        macros = _extract_nutriments(product)
        if macros:
            macros["product_name"] = _clean_str(product.get("product_name"))
            macros["source"] = SOURCE
            return macros
    return None


def fetch_by_barcode(code: str) -> dict[str, Any] | None:
    """Look up an exact packaged product by barcode; return its name/brand + nutrition.

    Returns a dict with the per-100 g macros plus ``product_name`` and ``brands`` when the
    barcode is known, or ``None`` when OFF has no such product. Macros may be ``None`` if
    the product exists but has no nutrition facts recorded.

    Raises:
        NutritionError: If OFF can't be reached or returns something unparseable.
    """
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    if not digits:
        return None
    body = _get(_PRODUCT_URL.format(code=digits), {"fields": _SEARCH_FIELDS})
    product = body.get("product") if isinstance(body, dict) else None
    if not isinstance(product, dict):
        return None  # OFF returns status 0 / no product for unknown barcodes
    macros = _extract_nutriments(product) or {}
    macros["product_name"] = _clean_str(product.get("product_name"))
    macros["brands"] = _clean_str(product.get("brands"))
    macros["code"] = digits
    macros["source"] = SOURCE
    return macros


# --- Cache enrichment (dependency-injected storage) -------------------------


def enrich(
    tokens: Iterable[str],
    *,
    get_cached: Callable[[str], Mapping[str, Any] | None],
    upsert: Callable[..., Any],
    force: bool = False,
    fetch: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Populate the nutrition cache for a set of ingredient tokens from OFF.

    For each distinct token, skip it when the cache already has a row (unless ``force``),
    otherwise fetch by its human-readable name and, on a hit, write it through ``upsert``.
    Network failures on a single token are swallowed and recorded as ``failed`` so one bad
    lookup never aborts the batch.

    Args:
        tokens: Canonical ingredient tokens (as produced by
            :func:`recommender.normalize_ingredient`).
        get_cached: ``token -> cached row | None`` (e.g. :func:`storage.get_nutrition`).
        upsert: Keyword-callable that writes a row, matching
            :func:`storage.upsert_nutrition`'s signature
            (``upsert(item, *, kcal, protein_g, carbs_g, fat_g, source)``).
        force: Re-fetch and overwrite even tokens that are already cached.
        fetch: The per-name fetcher (injectable for tests); resolved at call time to
            :func:`fetch_by_name` when not given, so it honours monkeypatching.

    Returns:
        A summary dict ``{"updated": [...], "cached": [...], "failed": [...],
        "not_found": [...], "source": SOURCE}`` listing tokens by outcome.
    """
    fetch = fetch or fetch_by_name
    updated: list[str] = []
    cached: list[str] = []
    failed: list[str] = []
    not_found: list[str] = []

    seen: set[str] = set()
    for raw in tokens:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)

        if not force and get_cached(token):
            cached.append(token)
            continue

        try:
            macros = fetch(recommender._prettify(token))
        except NutritionError as exc:
            logger.warning("Nutrition lookup failed for %s: %s", token, exc)
            failed.append(token)
            continue

        if not macros:
            not_found.append(token)
            continue

        upsert(
            token,
            kcal=macros.get("kcal"),
            protein_g=macros.get("protein_g"),
            carbs_g=macros.get("carbs_g"),
            fat_g=macros.get("fat_g"),
            source=macros.get("source", SOURCE),
        )
        updated.append(token)

    logger.info(
        "Nutrition enrich: %d updated, %d already cached, %d not found, %d failed",
        len(updated), len(cached), len(not_found), len(failed),
    )
    return {
        "updated": updated,
        "cached": cached,
        "not_found": not_found,
        "failed": failed,
        "source": SOURCE,
    }


def tokens_from_inventory(
    items: Iterable[Mapping[str, Any]],
    perishables: Iterable[Mapping[str, Any]] = (),
) -> list[str]:
    """Canonical ingredient tokens for the current fridge (scan items + tracked items).

    Deduplicated and free of empties, in first-seen order. This is what the web layer
    hands to :func:`enrich` so the cache tracks exactly what's on hand.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for source in (items, perishables):
        for entry in source:
            token = recommender.normalize_ingredient(str(entry.get("name", "")))
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


# --- HTTP + parsing helpers -------------------------------------------------


def _get(url: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """GET ``url`` with the OFF User-Agent and return parsed JSON, or raise NutritionError."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise NutritionError(f"Could not reach Open Food Facts: {exc}") from exc
    if not response.ok:
        raise NutritionError(f"Open Food Facts returned HTTP {response.status_code}.")
    try:
        body = response.json()
    except ValueError as exc:
        raise NutritionError("Open Food Facts returned invalid JSON.") from exc
    return body if isinstance(body, dict) else {}


def _extract_nutriments(product: Any) -> dict[str, Any] | None:
    """Pull the per-100 g macros we track from an OFF product, or None if none are present.

    A product row counts only when at least one macro is a real number — OFF has plenty of
    entries with an empty ``nutriments`` block, and those are worthless to the optimizer.
    """
    if not isinstance(product, Mapping):
        return None
    nutriments = product.get("nutriments")
    if not isinstance(nutriments, Mapping):
        return None
    macros: dict[str, Any] = {}
    have_any = False
    for our_key, off_key in _NUTRIMENT_MAP.items():
        value = _as_opt_float(nutriments.get(off_key))
        macros[our_key] = value
        if value is not None:
            have_any = True
    return macros if have_any else None


def _clean_str(value: Any) -> str | None:
    """Return a stripped string, or None for missing/empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_opt_float(value: Any) -> float | None:
    """Coerce to a non-negative float, returning None for missing/uncoercible values.

    OFF occasionally reports a stray negative or a string; both are treated as "unknown".
    """
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None
