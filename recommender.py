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

import difflib
import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    global _KNOWN_TOKENS_CACHE, _RESOLVABLE_CACHE, _FUZZY_INDEX_CACHE
    _recipes_cache = _aliases_cache = _embeddings_cache = None
    # Derived normalization caches depend on the above, so drop them together.
    _KNOWN_TOKENS_CACHE = _RESOLVABLE_CACHE = _FUZZY_INDEX_CACHE = None


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
_RESOLVABLE_CACHE: set[str] | None = None
_FUZZY_INDEX_CACHE: dict[str, str] | None = None

# Words that qualify an ingredient without changing its identity (state, size, cut,
# form) plus a few grocery brands a photo or user might include. Stripped from either
# end of a multi-word name when doing so lands on a recognised ingredient, e.g.
# "fresh ginger"→ginger, "boiled egg"→egg, "amul butter"→butter, "full cream milk"→milk.
_DESCRIPTORS = frozenset({
    "fresh", "frozen", "dried", "raw", "ripe", "boiled", "cooked", "roasted",
    "ground", "whole", "chopped", "minced", "grated", "shredded", "sliced",
    "diced", "cubed", "cubes", "cube", "powdered", "crushed", "peeled", "washed",
    "large", "small", "medium", "big", "extra", "full", "cream", "skimmed",
    "unsalted", "salted", "plain", "greek", "baby", "organic", "pure", "canned",
    "tinned", "packet", "pack", "of",
    # common grocery brands (kept short and unambiguous)
    "amul", "nestle", "britannia", "verka", "nandini", "gowardhan",
})

# Similarity threshold for the last-resort typo snap. High enough to avoid mapping a
# genuinely new ingredient onto an unrelated token, low enough to catch real typos
# ("panner"→paneer ≈ 0.83, "corriander"→coriander ≈ 0.95).
_FUZZY_CUTOFF = 0.82


def _known_tokens() -> set[str]:
    """Every canonical token that appears in the recipe corpus (for plural fallback)."""
    global _KNOWN_TOKENS_CACHE
    if _KNOWN_TOKENS_CACHE is None:
        tokens: set[str] = set()
        for recipe in _load_recipes():
            tokens.update(t.lower() for t in recipe.get("ingredients", []))
        _KNOWN_TOKENS_CACHE = tokens
    return _KNOWN_TOKENS_CACHE


def _resolvable_tokens() -> set[str]:
    """Tokens we are willing to resolve onto: app recipe tokens, alias targets, and the
    full trained embedding vocabulary. Extending past the ~49 recipe tokens to the whole
    embedding vocab means normalization lands on a token that actually has a vector."""
    global _RESOLVABLE_CACHE
    if _RESOLVABLE_CACHE is None:
        tokens = set(_known_tokens())
        tokens.update(_load_aliases().values())
        tokens.update(_load_embeddings()["vectors"])
        _RESOLVABLE_CACHE = tokens
    return _RESOLVABLE_CACHE


def _fuzzy_index() -> dict[str, str]:
    """Map a space-form search string → canonical token, used only for typo snapping.
    Every resolvable token contributes its space form; alias keys contribute too so a
    misspelled synonym ("grean chilli") can still snap to its canonical target."""
    global _FUZZY_INDEX_CACHE
    if _FUZZY_INDEX_CACHE is None:
        index: dict[str, str] = {}
        for token in _resolvable_tokens():
            index.setdefault(token.replace("_", " "), token)
        for key, value in _load_aliases().items():
            index.setdefault(key, value)
        _FUZZY_INDEX_CACHE = index
    return _FUZZY_INDEX_CACHE


def _resolve(text: str) -> str | None:
    """Alias / direct-token / de-pluralization lookup. Returns a canonical token, or None
    if the text is not recognised. Plurals and direct hits are checked against the full
    resolvable vocab, not just the recipe tokens."""
    aliases = _load_aliases()
    if text in aliases:
        return aliases[text]
    resolvable = _resolvable_tokens()
    token = text.replace(" ", "_")
    if token in resolvable:
        return token
    if text.endswith("es") and text[:-2] in aliases:
        return aliases[text[:-2]]
    singulars = [text[:-1], text[:-2]]
    if text.endswith("ies"):
        singulars.append(text[:-3] + "y")
    for singular in singulars:
        if not singular:
            continue
        if singular in aliases:
            return aliases[singular]
        if singular.replace(" ", "_") in resolvable:
            return singular.replace(" ", "_")
    return None


def _strip_descriptors(cleaned: str) -> str:
    """Drop leading/trailing qualifier and brand words, always keeping ≥1 core word."""
    words = cleaned.split()
    while len(words) > 1 and words[0] in _DESCRIPTORS:
        words = words[1:]
    while len(words) > 1 and words[-1] in _DESCRIPTORS:
        words = words[:-1]
    return " ".join(words)


def _fuzzy_snap(text: str) -> str | None:
    """Last-resort: snap a likely typo onto the closest known ingredient string. Returns
    the canonical token or None if nothing is within :data:`_FUZZY_CUTOFF`."""
    match = difflib.get_close_matches(text, _fuzzy_index(), n=1, cutoff=_FUZZY_CUTOFF)
    return _fuzzy_index()[match[0]] if match else None


def normalize_ingredient(name: str) -> str:
    """Map a free-text ingredient name to a canonical recipe token.

    Order: lower/trim → alias / direct-token / de-pluralization → strip qualifier & brand
    words ("fresh ginger"→ginger) → fuzzy-snap obvious typos ("panner"→paneer). Anything
    still unrecognised is returned in a cleaned canonical form so it can still be displayed
    and compared consistently. Every recognition step is additive: a name that already
    resolves is returned unchanged from the earlier, cheaper step.
    """
    cleaned = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not cleaned:
        return ""

    resolved = _resolve(cleaned)
    if resolved is not None:
        return resolved

    core = _strip_descriptors(cleaned)
    if core != cleaned:
        resolved = _resolve(core)
        if resolved is not None:
            return resolved

    snapped = _fuzzy_snap(cleaned)
    if snapped is None and core != cleaned:
        snapped = _fuzzy_snap(core)
    if snapped is not None:
        return snapped

    return (core or cleaned).replace(" ", "_")


def _prettify(token: str) -> str:
    """Turn a canonical token into a human label ('green_chili' → 'Green chili')."""
    overrides = {"rajma": "Rajma", "besan": "Besan", "poha": "Poha", "paneer": "Paneer"}
    if token in overrides:
        return overrides[token]
    return token.replace("_", " ").capitalize()


def is_known_ingredient(name: str) -> bool:
    """True if ``name`` resolves onto a real ingredient we know (recipe token, alias,
    or trained-embedding vocab), rather than an unrecognised string.

    :func:`normalize_ingredient` always returns *something* (a cleaned token) so callers
    can display it, but that token only lands in the resolvable vocab when the name was
    actually recognised. Receipt import uses this to tell a grocery line ("Amul Milk")
    from receipt noise that slipped past the meta-line filters.
    """
    token = normalize_ingredient(name)
    return bool(token) and token in _resolvable_tokens()


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


def similar_ingredients(
    token: str,
    *,
    k: int = 5,
    candidates: Any = None,
    min_similarity: float = 0.0,
) -> list[tuple[str, float]]:
    """Rank ingredient tokens most similar to ``token`` by trained-embedding cosine.

    Returns up to ``k`` ``(token, similarity)`` pairs, most-similar first. Only tokens
    that have a trained vector are considered, and the query token itself is always
    excluded. ``candidates`` restricts the pool (any iterable of tokens); by default the
    pool is every ingredient that appears in the recipe corpus, which keeps swap ideas
    grounded in dishes the app can actually cook. Pairs scoring below ``min_similarity``
    are dropped.

    This powers Cook Mode's "swap ideas": the vectors are trained purely on ingredient
    co-occurrence, so neighbours are ingredients that behave alike across recipes rather
    than a curated substitution table — a best-effort enhancement, never a guarantee.
    Returns an empty list when embeddings are unavailable or the token has no vector.
    """
    query = str(token).lower()
    vectors = _load_embeddings()["vectors"]
    base = vectors.get(query)
    if not base:
        return []
    if candidates is None:
        pool: set[str] = set(_known_tokens())
    else:
        pool = {str(c).lower() for c in candidates}
    scored: list[tuple[str, float]] = []
    for cand in pool:
        if cand == query:
            continue
        sim = _cosine(base, vectors.get(cand))
        if sim >= min_similarity:
            scored.append((cand, sim))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


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


def _local_zone() -> tzinfo:
    """The household's time zone: ``APP_TIMEZONE`` (an IANA name), else this machine's."""
    name = os.getenv("APP_TIMEZONE", "").strip()
    if name.upper() == "UTC":
        return UTC
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            logger.warning("Unknown APP_TIMEZONE %r; using the server's local time zone.", name)
    return datetime.now().astimezone().tzinfo or UTC


def local_today(now: datetime | None = None) -> date:
    """Today's date on the household's calendar.

    Use-by dates are entered as local calendar dates, so "days left" has to be counted from
    the local date too. Counting from the UTC date is a day off for part of every day — in
    India, from midnight to 05:30. ``now`` is a UTC instant (naive values are read as UTC).
    """
    when = now or datetime.now(UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(_local_zone()).date()


# --- Public API --------------------------------------------------------------


def recommend(
    inventory_items: list[dict[str, Any]] | None,
    perishables: list[dict[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    max_recipes: int = 6,
    use_soon_days: int = DEFAULT_USE_SOON_DAYS,
    recipe_filter: Callable[[Mapping[str, Any]], bool] | None = None,
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
        recipe_filter: Optional predicate over a raw recipe dict; recipes it rejects are
            never suggested (nor counted toward shopping ideas). The agent uses this to
            honour remembered household rules — diet, allergies, dislikes (see
            :mod:`preferences`).

    Returns:
        A JSON-serializable dict: ``{generated_at, use_soon, recipes, shopping, ml}``.
    """
    inventory_items = inventory_items or []
    perishables = perishables or []
    today = local_today(now)

    embeddings = _load_embeddings()
    vectors = embeddings["vectors"]
    dim = embeddings["dim"]
    ml_backend = "embedding" if vectors else "lexical"

    have, urgent, use_soon = _build_inventory_state(
        inventory_items, perishables, today, use_soon_days
    )

    fridge_vec = _mean_vector(sorted(have), vectors, dim) if vectors else None
    scored = _score_recipes(have, urgent, fridge_vec, vectors, dim, recipe_filter)

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
    recipe_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Score every recipe against the current fridge, returning enriched, ranked rows.

    Only recipes with at least one ingredient on hand and either reasonable coverage or a
    use-soon ingredient are kept — no point suggesting a dish you have almost nothing for.
    Recipes rejected by ``recipe_filter`` (household diet/allergy rules) are skipped.
    """
    scored: list[dict[str, Any]] = []
    for recipe in _load_recipes():
        if recipe_filter is not None and not recipe_filter(recipe):
            continue
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
