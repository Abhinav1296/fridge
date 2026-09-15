"""Vision-model integration for SmartFridge Vision MVP.

This module is deliberately **framework-agnostic**: it has no Flask or HTTP-request
coupling. Its single public entry point, :func:`analyze_image`, takes raw image bytes
and returns a normalized inventory ``dict``. A future agent layer can therefore import
and call it directly without touching the web layer.

Configuration is read from the environment at call time (never hardcoded):

* ``VISION_API_KEY``   — API key for the provider (required).
* ``VISION_BASE_URL``  — OpenAI-compatible base URL, e.g. ``https://api.groq.com/openai/v1``.
* ``VISION_MODEL``     — model id, e.g. ``meta-llama/llama-4-scout-17b-16e-instruct``.

**Fallbacks.** Free vision tiers rotate model ids and rate-limit aggressively, so the
service supports a *primary* plus up to three *fallback* profiles, tried in order until
one succeeds. Each fallback is configured with a numeric suffix (``_2``, ``_3``, ``_4``)
on any of the three variables above::

    VISION_API_KEY_2 / VISION_BASE_URL_2 / VISION_MODEL_2   # fallback 1
    VISION_API_KEY_3 / VISION_BASE_URL_3 / VISION_MODEL_3   # fallback 2
    VISION_API_KEY_4 / VISION_BASE_URL_4 / VISION_MODEL_4   # fallback 3

Any field left blank in a fallback inherits the value from the previous slot, so the
common case — one key/provider, several models — needs only the ``VISION_MODEL_*`` vars.
A fallback slot is ignored unless at least one of its variables is set.

Any OpenAI-compatible *vision* chat-completions endpoint works, so the provider is
swappable purely through configuration.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

# --- Defaults (overridable via environment) ---------------------------------

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# How many configuration slots to read: the primary (index 1, no suffix) plus three
# fallbacks (indices 2-4, suffixed ``_2`` / ``_3`` / ``_4``). Raise this to allow more.
_MAX_ATTEMPTS = 4

# Network timeouts as (connect, read) seconds. Vision calls can be slow, so the
# read timeout is generous while the connect timeout stays short.
_REQUEST_TIMEOUT: tuple[int, int] = (10, 90)

# --- Allowed enum values (used for normalization) ---------------------------

_CATEGORIES = {"fruit", "vegetable", "dairy", "beverage", "packaged", "other"}
_FRESHNESS = {"fresh", "ripe", "use_soon", "spoiled", "unknown"}
_CONFIDENCE = {"high", "medium", "low"}


# --- Exception hierarchy ----------------------------------------------------
# These typed exceptions let the web layer map failures to clean, user-facing
# messages while raw model text / provider details are logged server-side only.


class VisionServiceError(Exception):
    """Base class for all vision-service failures."""


class VisionConfigError(VisionServiceError):
    """Configuration is missing or invalid (e.g. no API key)."""


class VisionAPIError(VisionServiceError):
    """The provider API was unreachable or returned an error response."""


class RateLimitError(VisionAPIError):
    """The provider returned HTTP 429 (rate limited)."""


class VisionParseError(VisionServiceError):
    """The model response could not be parsed into the expected JSON shape."""


# --- Prompt -----------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a food-inventory vision assistant for a smart fridge. You look at a "
    "single photo of fridge or box contents and return a precise, structured "
    "inventory. You are careful and honest: you never invent items you cannot see, "
    "and you say 'unknown' when you are not sure."
)

_USER_PROMPT = (
    "Analyze this photo of fridge/box contents and return ONLY a JSON object "
    "(no markdown, no code fences, no prose before or after) with EXACTLY this shape:\n"
    "{\n"
    '  "items": [\n'
    '    {"name": string, "count": integer,\n'
    '     "category": "fruit|vegetable|dairy|beverage|packaged|other",\n'
    '     "freshness": "fresh|ripe|use_soon|spoiled|unknown",\n'
    '     "freshness_confidence": "high|medium|low",\n'
    '     "notes": string}\n'
    "  ],\n"
    '  "unidentified": [ {"description": string, "reason": string} ]\n'
    "}\n\n"
    "Rules:\n"
    "- List EVERY visible food item, each with a count (how many of that item you see).\n"
    "- Judge produce freshness from visual cues only: colour, bruising, wilting, "
    "softness, mould, condensation. Map to the scale: fresh (peak), ripe (ready now), "
    "use_soon (starting to turn), spoiled (mould/rot/heavy bruising), unknown.\n"
    "- Set freshness_confidence honestly. Use 'low' and 'unknown' when the image is "
    "unclear or the item is packaged so freshness is not visible.\n"
    "- Put anything you CANNOT identify (opaque bags, heavy occlusion, unreadable "
    "labels) into 'unidentified' with a short description and reason — do NOT guess it "
    "into 'items'.\n"
    "- 'notes' is a brief free-text remark (may be an empty string).\n"
    "- If you see no food at all, return empty arrays. Never return anything except the "
    "JSON object."
)


# --- Public API -------------------------------------------------------------


def analyze_image(image_bytes: bytes) -> dict[str, Any]:
    """Analyze fridge/box contents in ``image_bytes`` and return a normalized inventory.

    Args:
        image_bytes: Raw image data (JPEG/PNG/etc.). Callers are expected to have
            already validated size and downscaled if appropriate.

    Returns:
        A dict of the form ``{"items": [...], "unidentified": [...]}`` with every
        item field present and every enum value normalized to a known value.

    Behaviour:
        The primary and any configured fallbacks (see the module docstring) are tried
        in order. The first attempt that returns a parseable inventory wins. Each
        failed attempt is logged server-side and the next fallback is tried; if every
        attempt fails, the last error is raised so the web layer can map it to a
        friendly message.

    Raises:
        VisionConfigError: No API key is configured at all.
        RateLimitError: Every attempt was rate limited (HTTP 429).
        VisionAPIError: The final attempt failed with a network / non-2xx / malformed
            API response.
        VisionParseError: The final attempt returned content that was not parseable
            JSON of the expected shape.
    """
    if not image_bytes:
        raise VisionParseError("No image data was provided to the vision service.")

    attempts = _load_attempts()
    if not attempts:
        raise VisionConfigError(
            "VISION_API_KEY is not set. Add it to your .env file."
        )

    data_url = _to_data_url(image_bytes)
    total = len(attempts)
    last_error: VisionServiceError | None = None

    for index, attempt in enumerate(attempts, start=1):
        payload = {
            "model": attempt.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        }

        try:
            content = _call_chat_completions(attempt.base_url, attempt.api_key, payload)
            data = _parse_model_json(content)
            result = _normalize(data)
        except VisionServiceError as exc:
            last_error = exc
            more = total - index
            logger.warning(
                "Vision attempt %d/%d failed (model=%s, base_url=%s): %s%s",
                index,
                total,
                attempt.model,
                attempt.base_url,
                exc,
                " -- trying next fallback." if more else " -- no fallbacks left.",
            )
            continue

        if index > 1:
            logger.info(
                "Vision attempt %d/%d succeeded (model=%s) after %d failed attempt(s).",
                index,
                total,
                attempt.model,
                index - 1,
            )
        return result

    # Every attempt failed. Re-raise the last error, preserving its type so the web
    # layer maps it to the correct status code / message. (last_error is always set
    # here because `attempts` is non-empty.)
    assert last_error is not None  # noqa: S101 — invariant guard for type-checkers
    raise last_error


# --- Internal helpers -------------------------------------------------------


@dataclass(frozen=True)
class _Attempt:
    """One fully-resolved provider attempt: which key, endpoint, and model to use."""

    api_key: str
    base_url: str
    model: str


def _load_attempts() -> list[_Attempt]:
    """Build the ordered list of attempts (primary first, then fallbacks) from env.

    Slot 1 uses the unsuffixed variables; slots 2..``_MAX_ATTEMPTS`` use the ``_N``
    suffix. A fallback slot is only considered if at least one of its three variables
    is set. Any field a slot leaves unset inherits the previous slot's value, so the
    common "same key, different model" case needs only the extra ``VISION_MODEL_*``
    variables. Base URL and model fall back to the module defaults if never set.

    Attempts with no usable API key or model are skipped, and exact duplicate
    ``(key, base_url, model)`` triples are collapsed so no call is wasted repeating an
    identical request. The API key is never logged.
    """
    attempts: list[_Attempt] = []
    seen: set[tuple[str, str, str]] = set()

    # Inheritance chain — seeded with the defaults so an unset base_url/model resolves
    # exactly as the pre-fallback code did.
    last_key = ""
    last_base = DEFAULT_BASE_URL
    last_model = DEFAULT_MODEL

    for index in range(1, _MAX_ATTEMPTS + 1):
        suffix = "" if index == 1 else f"_{index}"
        raw_key = os.getenv(f"VISION_API_KEY{suffix}")
        raw_base = os.getenv(f"VISION_BASE_URL{suffix}")
        raw_model = os.getenv(f"VISION_MODEL{suffix}")

        # A fallback slot that sets nothing at all is simply absent.
        if index > 1 and raw_key is None and raw_base is None and raw_model is None:
            continue

        key = (raw_key if raw_key is not None else last_key).strip()
        base = (raw_base if raw_base is not None else last_base).strip().rstrip("/")
        model = (raw_model if raw_model is not None else last_model).strip()

        # Advance the inheritance chain only from explicitly-set, non-empty values.
        if raw_key is not None and key:
            last_key = key
        if raw_base is not None and base:
            last_base = base
        if raw_model is not None and model:
            last_model = model

        # A slot with no key can't authenticate; with no model it can't be called.
        if not key or not model:
            continue

        triple = (key, base, model)
        if triple in seen:
            continue
        seen.add(triple)
        attempts.append(_Attempt(api_key=key, base_url=base, model=model))

    return attempts


def _detect_mime(image_bytes: bytes) -> str:
    """Best-effort image MIME-type sniff from magic bytes; defaults to JPEG."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _to_data_url(image_bytes: bytes) -> str:
    """Encode raw image bytes as a base64 ``data:`` URL for the message content."""
    mime = _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _call_chat_completions(base_url: str, api_key: str, payload: dict[str, Any]) -> str:
    """POST to the chat-completions endpoint and return the message content string.

    Raises the appropriate :class:`VisionServiceError` subclass on any failure.
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
        )
    except requests.exceptions.Timeout as exc:
        logger.warning("Vision API request timed out: %s", exc)
        raise VisionAPIError("The vision service timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        logger.warning("Vision API request failed: %s", exc)
        raise VisionAPIError(
            "Could not reach the vision service. Check your connection and VISION_BASE_URL."
        ) from exc

    if response.status_code == 429:
        # Log the body server-side for diagnostics; keep the client message friendly.
        logger.warning("Vision API rate limited (429): %s", _safe_body(response))
        raise RateLimitError(
            "The vision service is rate limited right now. Please wait a moment and try again."
        )

    if response.status_code == 401 or response.status_code == 403:
        logger.warning(
            "Vision API auth error (%s): %s", response.status_code, _safe_body(response)
        )
        raise VisionConfigError(
            "The vision service rejected the API key. Check VISION_API_KEY."
        )

    if not response.ok:
        logger.warning(
            "Vision API returned %s: %s", response.status_code, _safe_body(response)
        )
        raise VisionAPIError(
            f"The vision service returned an error (HTTP {response.status_code})."
        )

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected vision API response shape: %s", _safe_body(response))
        raise VisionAPIError(
            "The vision service returned an unexpected response."
        ) from exc

    if not isinstance(content, str):
        # Some providers return content as a list of parts; join any text parts.
        content = _coerce_content_to_text(content)

    return content


def _coerce_content_to_text(content: Any) -> str:
    """Flatten a structured message-content value into a plain text string."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _safe_body(response: requests.Response, limit: int = 2000) -> str:
    """Return a truncated response body for server-side logging only."""
    try:
        return response.text[:limit]
    except Exception:  # noqa: BLE001 — logging must never raise
        return "<unreadable response body>"


def _parse_model_json(content: str) -> dict[str, Any]:
    """Strip code fences/whitespace and parse the model output as JSON.

    On failure the raw text is logged server-side (never returned to the client) and
    a :class:`VisionParseError` is raised.
    """
    cleaned = _strip_code_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse model JSON. Raw content was: %r", content)
        raise VisionParseError(
            "The vision model returned a response we couldn't read. Please try again."
        ) from exc

    if not isinstance(data, dict):
        logger.warning("Model JSON was not an object. Raw content was: %r", content)
        raise VisionParseError(
            "The vision model returned an unexpected structure. Please try again."
        )
    return data


def _strip_code_fences(text: str) -> str:
    """Remove surrounding ```json ... ``` (or plain ``` ... ```) fences and whitespace."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json).
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        else:
            stripped = stripped[3:]
        # Drop a trailing closing fence if present.
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parsed-but-possibly-loose model dict into the canonical shape.

    This makes a slightly off (but valid-JSON) response still render cleanly rather
    than crashing the caller: enums are snapped to known values, ``count`` becomes an
    int, and the two top-level keys are always present lists.
    """
    raw_items = data.get("items")
    raw_unidentified = data.get("unidentified")

    items = [
        _normalize_item(item)
        for item in (raw_items if isinstance(raw_items, list) else [])
        if isinstance(item, dict)
    ]
    unidentified = [
        _normalize_unidentified(entry)
        for entry in (raw_unidentified if isinstance(raw_unidentified, list) else [])
        if isinstance(entry, dict)
    ]

    return {"items": items, "unidentified": unidentified}


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single inventory item to the canonical field set/values."""
    return {
        "name": _as_str(item.get("name"), default="unknown item"),
        "count": _as_positive_int(item.get("count")),
        "category": _snap(item.get("category"), _CATEGORIES, "other"),
        "freshness": _snap(item.get("freshness"), _FRESHNESS, "unknown"),
        "freshness_confidence": _snap(
            item.get("freshness_confidence"), _CONFIDENCE, "low"
        ),
        "notes": _as_str(item.get("notes"), default=""),
    }


def _normalize_unidentified(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single 'unidentified' entry."""
    return {
        "description": _as_str(entry.get("description"), default="unidentified object"),
        "reason": _as_str(entry.get("reason"), default=""),
    }


def _snap(value: Any, allowed: set[str], default: str) -> str:
    """Return ``value`` lowercased if it is an allowed enum member, else ``default``."""
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def _as_str(value: Any, default: str = "") -> str:
    """Coerce ``value`` to a trimmed string, falling back to ``default``."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    return str(value)


def _as_positive_int(value: Any) -> int:
    """Coerce ``value`` to an int >= 1 (defaults to 1 when missing/invalid)."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return count if count >= 1 else 1
