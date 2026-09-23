"""Recipe & shopping recommendation engine for SmartFridge Vision (Phase 3).

This is the part of the project the team builds itself: given what the fridge currently
holds (the latest scan) plus any user-entered "use by" dates, it produces three things a
household actually wants:

1. **Use soon** — items that are expiring, starting to turn, or already spoiled, so
   nothing is quietly wasted.
2. **Recipes you can cook** — dishes ranked by how much of the recipe you already have
   and by whether they use up something expiring, so the suggestion is both convenient
   and waste-reducing.
3. **Shopping suggestions** — the few missing ingredients that would unlock the most
   near-complete recipes.

**Why it counts as machine learning.** Matching "what's in the fridge" to "what I can
cook" is not just string equality: ``curd`` and ``yogurt`` are the same thing, and a
fridge full of ``onion``/``tomato``/``ginger`` is *semantically* close to a curry even
before an exact ingredient match. To capture that, each ingredient is represented by a
learned **embedding** (a dense vector trained by :mod:`build_embeddings` using the
skip-gram / word2vec technique on the recipe corpus). Recipe ranking blends exact
ingredient coverage, expiry urgency, and the **cosine similarity** of these embeddings.
If the trained vectors are absent the engine degrades gracefully to a purely lexical
(coverage + urgency) ranking, so the app never breaks.

Like :mod:`vision_service` and :mod:`storage`, this module is **framework-agnostic** —
no Flask, no HTTP — so the web layer or a future autonomous agent can call
:func:`recommend` directly. It reads its data files with the standard library only (no
numpy at runtime), which keeps the deployed app lightweight.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import UTC, date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# --- Data-file locations (overridable via environment, non-secret) -----------

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "data")


def _data_path(env_var: str, filename: str) -> str:
    """Resolve a data file path from the environment, defaulting to the bundled copy."""
    return os.getenv(env_var, "").strip() or os.path.join(_DATA_DIR, filename)


# Pantry staples assumed to be on hand: they are ignored for recipe matching and never
# appear as "missing" or in the shopping list, because nobody wants "buy salt" as advice.
PANTRY = frozenset({
    "salt", "oil", "water", "sugar", "ghee", "vinegar", "soy sauce",
    "turmeric", "chili powder", "red chili powder", "cumin", "mustard seed",
    "garam masala", "black pepper", "pepper", "coriander powder", "curry leaves",
    "bay leaf", "asafoetida", "baking soda", "spices",
})

# Default window (days) within which a dated perishable is treated as "use soon".
DEFAULT_USE_SOON_DAYS = 3

# Freshness values (from vision_service) that mean "cook this soon" vs "throw this out".
_FRESHNESS_SOON = "use_soon"
_FRESHNESS_SPOILED = "spoiled"

# Ranking weights. Coverage (how much of the recipe you already have) dominates; expiry
# urgency and semantic embedding similarity refine the order.
_W_COVERAGE = 0.55
_W_URGENCY = 0.30
_W_EMBED = 0.15


# --- Lazily-loaded, cached data ---------------------------------------------
# Loaded once on first use and reused. Kept module-level (not per-call) so repeated
# requests on a warm server don't re-read disk. Call reset_cache() in tests.

_recipes_cache: list[dict[str, Any]] | None = None
_aliases_cache: dict[str, str] | None = None
_embeddings_cache: dict[str, Any] | None = None


def reset_cache() -> None:
    """Drop cached data files so the next call reloads them (used by tests)."""
    global _recipes_cache, _aliases_cache, _embeddings_cache
    _recipes_cache = _aliases_cache = _embeddings_cache = None


def _load_recipes() -> list[dict[str, Any]]:
    global _recipes_cache
    if _recipes_cache is None:
        path = _data_path("RECIPES_PATH", "recipes.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        _recipes_cache = [r for r in data.get("recipes", []) if r.get("ingredients")]
        logger.info("Loaded %d recipes from %s", len(_recipes_cache), path)
    return _recipes_cache


def _load_aliases() -> dict[str, str]:
    global _aliases_cache
    if _aliases_cache is None:
        path = _data_path("INGREDIENT_ALIASES_PATH", "ingredient_aliases.json")
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            _aliases_cache = {k.lower(): v for k, v in data.get("aliases", {}).items()}
        except (OSError, ValueError):
            logger.warning("Ingredient aliases unavailable at %s; using names as-is", path)
            _aliases_cache = {}
    return _aliases_cache


def _load_embeddings() -> dict[str, Any]:
    """Return ``{"dim": int, "vectors": {token: [floats]}}`` or an empty dict if absent."""
    global _embeddings_cache
    if _embeddings_cache is None:
        path = _data_path("INGREDIENT_EMBEDDINGS_PATH", "ingredient_embeddings.json")
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            _embeddings_cache = {
                "dim": int(data.get("dim", 0)),
                "vectors": {k: [float(x) for x in v] for k, v in data.get("vectors", {}).items()},
            }
            logger.info(
                "Loaded %d ingredient embeddings from %s",
                len(_embeddings_cache["vectors"]),
                path,
            )
        except (OSError, ValueError):
            logger.info(
                "No trained embeddings at %s; recommender using lexical ranking only", path
            )
            _embeddings_cache = {"dim": 0, "vectors": {}}
    return _embeddings_cache


# --- Ingredient name normalization ------------------------------------------

_KNOWN_TOKENS_CACHE: set[str] | None = None


def _known_tokens() -> set[str]:
    """Every canonical token that appears in the recipe corpus (for plural fallback)."""
    global _KNOWN_TOKENS_CACHE
    if _KNOWN_TOKENS_CACHE is None:
        tokens: set[str] = set()
        for recipe in _load_recipes():
            tokens.update(t.lower() for t in recipe.get("ingredients", []))
        _KNOWN_TOKENS_CACHE = tokens
    return _KNOWN_TOKENS_CACHE


def normalize_ingredient(name: str) -> str:
    """Map a free-text ingredient name to a canonical recipe token.

    Order: lower/trim → alias lookup → collapse spaces to underscores → simple
    de-pluralization when it lands on a known token. Anything unrecognized is returned
    in a cleaned canonical form so it can still be displayed and compared consistently.
    """
    cleaned = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not cleaned:
        return ""

    aliases = _load_aliases()
    if cleaned in aliases:
        return aliases[cleaned]

    token = cleaned.replace(" ", "_")
    known = _known_tokens()
    if token in known:
        return token

    # Simple plural → singular ("tomatoes"/"onions" → "tomato"/"onion") when it helps.
    if cleaned.endswith("es") and cleaned[:-2] in aliases:
        return aliases[cleaned[:-2]]
    for singular in (cleaned[:-1], cleaned[:-2], cleaned[:-3] + "y" if cleaned.endswith("ies") else ""):
        if singular and singular.replace(" ", "_") in known:
            return singular.replace(" ", "_")
    return token


def _prettify(token: str) -> str:
    """Turn a canonical token into a human label ('green_chili' → 'Green chili')."""
    overrides = {"rajma": "Rajma", "besan": "Besan", "poha": "Poha", "paneer": "Paneer"}
    if token in overrides:
        return overrides[token]
    return token.replace("_", " ").capitalize()


# --- Embedding math (pure Python) -------------------------------------------


def _mean_vector(tokens: list[str], vectors: dict[str, list[float]], dim: int) -> list[float] | None:
    """Return the L2-normalized mean of the given tokens' vectors, or None if none apply."""
    present = [vectors[t] for t in tokens if t in vectors]
    if not present:
        return None
    mean = [sum(vals) / len(present) for vals in zip(*present)]
    norm = math.sqrt(sum(v * v for v in mean))
    if norm == 0:
        return None
    return [v / norm for v in mean]


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two already-L2-normalized vectors (0.0 if either is missing)."""
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# --- Public accessors (shared with the semantic search index) ----------------


def all_recipes() -> list[dict[str, Any]]:
    """Return the recipe corpus as shallow copies (id/title/ingredients/tags/time_min).

    Exposed for :mod:`vectorstore`, which builds a semantic search index over the
    same corpus. Copies keep callers from mutating the module cache.
    """
    return [dict(recipe) for recipe in _load_recipes()]


def embed_ingredients(tokens: list[str]) -> list[float] | None:
    """Return the L2-normalized mean embedding of the given canonical tokens.

    None when no token has a trained vector (or embeddings are unavailable). Shared
    by recipe scoring and the semantic search index so both rank in the same vector
    space. Tokens are lower-cased to match the embedding vocabulary.
    """
    embeddings = _load_embeddings()
    return _mean_vector(
        [str(token).lower() for token in tokens], embeddings["vectors"], embeddings["dim"]
    )


# --- Date helpers ------------------------------------------------------------


def _parse_use_by(value: Any) -> date | None:
    """Parse a 'YYYY-MM-DD' use-by string into a date, tolerating full ISO timestamps."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# --- Public API --------------------------------------------------------------


def recommend(
    inventory_items: list[dict[str, Any]] | None,
    perishables: list[dict[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    max_recipes: int = 6,
    use_soon_days: int = DEFAULT_USE_SOON_DAYS,
) -> dict[str, Any]:
    """Produce use-soon, recipe, and shopping suggestions for the current fridge state.

    Args:
        inventory_items: Items from the latest scan (each a dict with at least ``name``,
            and optionally ``freshness``/``count``), or None/empty for an empty fridge.
        perishables: User-tracked items with a ``use_by`` date (dicts with ``name`` and
            ``use_by``), or None.
        now: Reference time (UTC) for expiry math; defaults to the current time. Present
            for deterministic testing.
        max_recipes: Maximum number of recipe suggestions to return.
        use_soon_days: A dated perishable within this many days is treated as "use soon".

    Returns:
        A JSON-serializable dict: ``{generated_at, use_soon, recipes, shopping, ml}``.
    """
    inventory_items = inventory_items or []
    perishables = perishables or []
    today = (now or datetime.now(UTC)).date()

    embeddings = _load_embeddings()
    vectors = embeddings["vectors"]
    dim = embeddings["dim"]
    ml_backend = "embedding" if vectors else "lexical"

    have, urgent, use_soon = _build_inventory_state(
        inventory_items, perishables, today, use_soon_days
    )

    fridge_vec = _mean_vector(sorted(have), vectors, dim) if vectors else None
    scored = _score_recipes(have, urgent, fridge_vec, vectors, dim)

    recipes = _top_recipes(scored, max_recipes)
    shopping = _shopping_from(scored)

    return {
        "generated_at": today.strftime("%Y-%m-%dT00:00:00Z"),
        "use_soon": use_soon,
        "recipes": recipes,
        "shopping": shopping,
        "ml": {
            "backend": ml_backend,
            "dim": dim or None,
            "recipes_considered": len(_load_recipes()),
            "ingredients_on_hand": len(have),
        },
    }


# --- Inventory → normalized state -------------------------------------------


def _build_inventory_state(
    inventory_items: list[dict[str, Any]],
    perishables: list[dict[str, Any]],
    today: date,
    use_soon_days: int,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Fold the scan + perishables into (available tokens, urgent tokens, use-soon list).

    * ``have`` — canonical tokens usable as ingredients (excludes pantry staples and
      anything visually spoiled or already past its use-by date).
    * ``urgent`` — tokens to *cook soon* (freshness ``use_soon`` or a perishable due
      within the window), used to prioritize recipes that consume them.
    * ``use_soon`` — the display list for the "Use soon" panel, covering soon / overdue /
      spoiled, most-pressing first.
    """
    have: set[str] = set()
    urgent: set[str] = set()
    use_soon: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()  # de-dup the display list by token
    blocked: set[str] = set()      # tokens to discard (overdue/spoiled) — never cook with these

    # Perishables carry the most reliable signal — an explicit date.
    for entry in perishables:
        name = str(entry.get("name", "")).strip()
        token = normalize_ingredient(name)
        if not token or token in PANTRY:
            continue
        use_by = _parse_use_by(entry.get("use_by"))
        days_left = (use_by - today).days if use_by else None

        if days_left is not None and days_left < 0:
            blocked.add(token)  # explicit date says expired — this wins over any visual guess
            _add_use_soon(use_soon, seen_tokens, token, name, "overdue",
                          f"{-days_left} day{'s' if -days_left != 1 else ''} past use-by", days_left)
            continue  # past its date → not a usable ingredient
        have.add(token)
        if days_left is not None and days_left <= use_soon_days:
            urgent.add(token)
            label = "today" if days_left == 0 else f"{days_left} day{'s' if days_left != 1 else ''} left"
            _add_use_soon(use_soon, seen_tokens, token, name, "soon", label, days_left)

    # Scan items — freshness comes from the vision model.
    for item in inventory_items:
        name = str(item.get("name", "")).strip()
        token = normalize_ingredient(name)
        if not token or token in PANTRY:
            continue
        freshness = str(item.get("freshness", "")).strip().lower()

        if freshness == _FRESHNESS_SPOILED:
            blocked.add(token)
            _add_use_soon(use_soon, seen_tokens, token, name, "spoiled",
                          "looks spoiled - check before using", None)
            continue  # spoiled → flag it, don't cook with it
        have.add(token)
        if freshness == _FRESHNESS_SOON:
            urgent.add(token)
            _add_use_soon(use_soon, seen_tokens, token, name, "soon",
                          "starting to turn", None)

    # "Discard" is authoritative over "usable": if either signal flagged a token for
    # disposal, drop it from the cookable/urgent sets no matter which loop added it.
    have -= blocked
    urgent -= blocked

    # Order the panel: overdue/spoiled first, then soonest by days-left.
    severity_rank = {"overdue": 0, "spoiled": 1, "soon": 2}

    def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
        days = row.get("days_left")
        return (severity_rank.get(row["severity"], 3), days if days is not None else 99)

    use_soon.sort(key=_sort_key)
    return have, urgent, use_soon


def _add_use_soon(
    use_soon: list[dict[str, Any]],
    seen_tokens: set[str],
    token: str,
    display_name: str,
    severity: str,
    reason: str,
    days_left: int | None,
) -> None:
    """Append a de-duplicated 'use soon' row keyed by token (first mention wins)."""
    if token in seen_tokens:
        return
    seen_tokens.add(token)
    use_soon.append({
        "name": display_name or _prettify(token),
        "token": token,
        "severity": severity,   # 'overdue' | 'spoiled' | 'soon'
        "reason": reason,
        "days_left": days_left,
    })


# --- Recipe scoring ----------------------------------------------------------


def _score_recipes(
    have: set[str],
    urgent: set[str],
    fridge_vec: list[float] | None,
    vectors: dict[str, list[float]],
    dim: int,
) -> list[dict[str, Any]]:
    """Score every recipe against the current fridge, returning enriched, ranked rows.

    Only recipes with at least one ingredient on hand and either reasonable coverage or a
    use-soon ingredient are kept — no point suggesting a dish you have almost nothing for.
    """
    scored: list[dict[str, Any]] = []
    for recipe in _load_recipes():
        req = [t.lower() for t in recipe.get("ingredients", [])]
        req_set = set(req)
        if not req_set:
            continue

        matched = sorted(req_set & have)
        missing = sorted(req_set - have)
        coverage = len(matched) / len(req_set)
        uses_expiring = sorted(t for t in matched if t in urgent)

        if not matched:
            continue
        if coverage < 0.4 and not uses_expiring:
            continue

        recipe_vec = _mean_vector(req, vectors, dim) if vectors else None
        # Map cosine (-1..1) to (0..1) so it contributes as a gentle positive signal.
        embed_score = (_cosine(fridge_vec, recipe_vec) + 1.0) / 2.0 if fridge_vec else 0.0
        urgency_score = min(1.0, 0.6 * len(uses_expiring))

        base = _W_COVERAGE * coverage + _W_URGENCY * urgency_score + _W_EMBED * embed_score
        can_make = not missing
        # Cookable-now and expiry-using recipes float to the top.
        rank = base + (0.5 if can_make else 0.0) + min(0.5, 0.25 * len(uses_expiring))

        scored.append({
            "id": recipe.get("id"),
            "title": recipe.get("title", recipe.get("id", "Recipe")),
            "time_min": recipe.get("time_min"),
            "tags": recipe.get("tags", []),
            "matched": [_prettify(t) for t in matched],
            "missing": [_prettify(t) for t in missing],
            "missing_tokens": missing,
            "uses_expiring": [_prettify(t) for t in uses_expiring],
            "coverage": round(coverage, 3),
            "can_make": can_make,
            "score": round(rank, 4),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


def _top_recipes(scored: list[dict[str, Any]], max_recipes: int) -> list[dict[str, Any]]:
    """Trim to the top recipes and cap each one's displayed 'missing' list."""
    top = scored[: max(0, int(max_recipes))]
    out = []
    for row in top:
        row = dict(row)
        row["missing"] = row["missing"][:4]
        row.pop("missing_tokens", None)
        out.append(row)
    return out


def _shopping_from(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Suggest the few ingredients that unlock the most near-complete recipes.

    Considers recipes missing just 1–2 ingredients and counts, per missing ingredient,
    how many such recipes buying it would complete. The biggest unlockers come first.
    """
    unlocks: dict[str, list[str]] = {}
    for row in scored:
        missing = row.get("missing_tokens", [])
        if 1 <= len(missing) <= 2:
            for token in missing:
                unlocks.setdefault(token, []).append(row["title"])

    ranked = sorted(unlocks.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    return [
        {"item": _prettify(token), "unlocks": titles[:4], "count": len(titles)}
        for token, titles in ranked[:6]
    ]
