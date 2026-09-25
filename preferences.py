"""Household preferences the Chef agent remembers — and the rules they imply.

The agent's long-term memory (:func:`storage.get_memory`) is a small key/value store. This
module is the single place that knows what those keys *mean*:

* :func:`normalize` validates and canonicalizes a ``(key, value)`` pair before it is stored
  (so a model can't write arbitrary junk into memory, and "Vegetarian " / "veg" collapse to
  one value);
* :func:`build_filter` turns remembered diet / allergy / dislike rules into a recipe
  predicate that :func:`recommender.recommend` and :func:`optimizer.plan_meals` apply, so
  every suggestion — from the agent *and* from the regular tabs — respects them;
* :func:`violations` explains why a specific recipe breaks the rules (the agent uses it to
  refuse to "cook" a dish containing an allergen); and
* :func:`summarize` renders memory as one short line for the agent's system prompt.

Pure and framework-agnostic: callers pass the memory dict in, nothing here touches storage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import recommender

# Hard limits on what can be remembered (defence against a model stuffing memory).
MAX_TEXT = 60        # characters per remembered value / list entry
MAX_LIST = 20        # entries per list-valued key

# --- Vocabulary --------------------------------------------------------------

DIETS = ("any", "vegetarian", "eggetarian", "vegan", "jain")

_DIET_ALIASES = {
    "veg": "vegetarian", "vegetarian": "vegetarian", "pure veg": "vegetarian",
    "lacto vegetarian": "vegetarian", "lacto-vegetarian": "vegetarian",
    "eggetarian": "eggetarian", "ovo vegetarian": "eggetarian", "egg ok": "eggetarian",
    "vegan": "vegan", "plant based": "vegan", "plant-based": "vegan",
    "jain": "jain",
    "any": "any", "none": "any", "no restrictions": "any", "non-veg": "any",
    "non veg": "any", "nonveg": "any", "non-vegetarian": "any", "omnivore": "any",
}

# Ingredient tokens by the dietary class that excludes them.
MEAT = frozenset({
    "chicken", "mutton", "lamb", "goat", "beef", "pork", "bacon", "ham", "sausage",
    "fish", "prawn", "shrimp", "crab", "squid", "tuna", "salmon", "anchovy", "turkey",
})
EGG = frozenset({"egg"})
DAIRY = frozenset({"milk", "butter", "cream", "paneer", "cheese", "yogurt", "ghee", "curd"})
# Jain cooking avoids root vegetables (and honey, mushrooms) on top of being vegetarian.
JAIN_EXCLUDED = frozenset({
    "onion", "garlic", "potato", "carrot", "ginger", "beetroot", "radish", "sweet_potato",
    "mushroom", "honey",
})

# Allergy / intolerance groups a user will name ("I'm lactose intolerant", "nut allergy")
# mapped to the ingredient tokens they rule out. Anything else is treated as a single
# ingredient and normalized through the recommender (so "peanuts" -> "peanut").
ALLERGEN_GROUPS: dict[str, frozenset[str]] = {
    "dairy": DAIRY,
    "lactose": DAIRY,
    "milk": DAIRY,
    "nuts": frozenset({"peanut", "cashew", "almond", "walnut", "pistachio", "hazelnut"}),
    "tree nuts": frozenset({"cashew", "almond", "walnut", "pistachio", "hazelnut"}),
    "peanuts": frozenset({"peanut"}),
    "gluten": frozenset({"flour", "bread", "pasta", "noodles", "semolina", "wheat", "maida"}),
    "wheat": frozenset({"flour", "bread", "pasta", "noodles", "semolina", "wheat", "maida"}),
    "egg": EGG,
    "eggs": EGG,
    "soy": frozenset({"tofu", "soy", "soybean"}),
    "shellfish": frozenset({"prawn", "shrimp", "crab"}),
    "fish": frozenset({"fish", "tuna", "salmon", "anchovy"}),
    "sesame": frozenset({"sesame"}),
}

# Keys the agent may write, with how each value is validated.
_LIST_KEYS = ("allergies", "dislikes", "favorite_cuisines", "notes")
_TEXT_KEYS = ("spice_level", "goal")
KEYS = ("diet", "household_size", *_LIST_KEYS, *_TEXT_KEYS)

_SPICE = {"mild": "mild", "low": "mild", "medium": "medium", "normal": "medium",
          "hot": "hot", "spicy": "hot", "very spicy": "hot", "high": "hot"}


# --- Validation ---------------------------------------------------------------


_KEY_ALIASES = {
    "allergy": "allergies", "dislike": "dislikes", "cuisine": "favorite_cuisines",
    "cuisines": "favorite_cuisines", "people": "household_size", "servings": "household_size",
    "spice": "spice_level", "note": "notes", "diet_type": "diet", "goals": "goal",
}


def canonical_key(key: str) -> str:
    """Map a (possibly loosely-worded) key to its canonical name.

    Raises:
        ValueError: the key isn't one the household memory supports.
    """
    name = str(key or "").strip().lower().replace(" ", "_").replace("-", "_")
    name = _KEY_ALIASES.get(name, name)
    if name not in KEYS:
        raise ValueError(f"Unknown preference '{key}'. Allowed keys: {', '.join(KEYS)}.")
    return name


def is_list_key(key: str) -> bool:
    """True for keys that hold a list (allergies, dislikes, ...)."""
    return key in _LIST_KEYS


def normalize(key: str, value: Any) -> tuple[str, Any]:
    """Validate a memory write and return its canonical ``(key, value)``.

    Raises:
        ValueError: with a short, model-readable reason when the key is unknown or the
            value doesn't fit (so the agent can correct itself instead of storing junk).
    """
    name = canonical_key(key)

    if name == "diet":
        text = _clean_text(value).lower()
        diet = _DIET_ALIASES.get(text)
        if diet is None:
            raise ValueError(f"Unknown diet '{value}'. Use one of: {', '.join(DIETS)}.")
        return name, diet

    if name == "household_size":
        try:
            size = int(value)
        except (TypeError, ValueError):
            raise ValueError("household_size must be a whole number of people.") from None
        if not 1 <= size <= 20:
            raise ValueError("household_size must be between 1 and 20.")
        return name, size

    if name == "spice_level":
        text = _clean_text(value).lower()
        if text not in _SPICE:
            raise ValueError("spice_level must be mild, medium or hot.")
        return name, _SPICE[text]

    if name in _TEXT_KEYS:
        text = _clean_text(value)
        if not text:
            raise ValueError(f"{name} can't be empty.")
        return name, text

    # List-valued keys: accept a list or a comma-separated string.
    entries = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    cleaned: list[str] = []
    for entry in entries:
        text = _clean_text(entry)
        if name in ("allergies", "dislikes"):
            text = text.lower()
        if text and text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        raise ValueError(f"{name} needs at least one entry.")
    if len(cleaned) > MAX_LIST:
        raise ValueError(f"{name} can hold at most {MAX_LIST} entries.")
    return name, cleaned


def merge(memory: Mapping[str, Any], key: str, value: Any) -> tuple[str, Any]:
    """Normalize ``(key, value)`` and, for list keys, union it with what's remembered.

    "I'm also allergic to sesame" should *add* sesame, not replace the peanut allergy.
    """
    name, clean = normalize(key, value)
    if name in _LIST_KEYS:
        existing = memory.get(name) or []
        combined = [e for e in existing if isinstance(e, str)]
        for entry in clean:
            if entry not in combined:
                combined.append(entry)
        if len(combined) > MAX_LIST:
            raise ValueError(f"{name} can hold at most {MAX_LIST} entries.")
        return name, combined
    return name, clean


# --- Rules --------------------------------------------------------------------


def excluded_tokens(memory: Mapping[str, Any] | None) -> dict[str, str]:
    """Return ``{ingredient_token: reason}`` for every ingredient the household avoids."""
    memory = memory or {}
    blocked: dict[str, str] = {}

    diet = memory.get("diet")
    if diet in ("vegetarian", "eggetarian", "vegan", "jain"):
        for token in MEAT:
            blocked.setdefault(token, f"not {diet}")
    if diet in ("vegetarian", "vegan", "jain"):
        for token in EGG:
            blocked.setdefault(token, f"not {diet}")
    if diet == "vegan":
        for token in DAIRY | {"honey"}:
            blocked.setdefault(token, "not vegan")
    if diet == "jain":
        for token in JAIN_EXCLUDED:
            blocked.setdefault(token, "not jain")

    # Allergies outrank diet/dislikes in the reason (they are a safety issue).
    for entry in _as_list(memory.get("allergies")):
        for token in _expand(entry):
            blocked[token] = f"allergy: {entry}"
    for entry in _as_list(memory.get("dislikes")):
        for token in _expand(entry):
            blocked.setdefault(token, f"dislikes {entry}")
    return blocked


def violations(recipe: Mapping[str, Any], memory: Mapping[str, Any] | None) -> list[str]:
    """Explain why ``recipe`` breaks the household rules (empty list = it's fine)."""
    blocked = excluded_tokens(memory)
    reasons: list[str] = []
    diet = (memory or {}).get("diet")
    tags = {str(t).lower() for t in recipe.get("tags", [])}
    if diet in ("vegetarian", "eggetarian", "vegan", "jain") and "non-veg" in tags:
        reasons.append(f"non-veg dish (household is {diet})")
    for token in recipe.get("ingredients", []):
        reason = blocked.get(str(token).lower())
        if reason:
            reasons.append(f"{recommender._prettify(str(token).lower())} ({reason})")
    return reasons


def has_allergen(recipe: Mapping[str, Any], memory: Mapping[str, Any] | None) -> bool:
    """True if the recipe contains something the household is *allergic* to."""
    return any("allergy" in reason for reason in violations(recipe, memory))


def build_filter(
    memory: Mapping[str, Any] | None,
) -> Callable[[Mapping[str, Any]], bool] | None:
    """Return a recipe predicate for the remembered rules, or ``None`` if there are none.

    ``None`` (rather than an always-true function) lets callers skip filtering entirely,
    keeping the no-preferences path byte-for-byte identical to before.
    """
    blocked = excluded_tokens(memory)
    diet = (memory or {}).get("diet")
    strict_veg = diet in ("vegetarian", "eggetarian", "vegan", "jain")
    if not blocked and not strict_veg:
        return None

    def _allowed(recipe: Mapping[str, Any]) -> bool:
        if strict_veg and "non-veg" in {str(t).lower() for t in recipe.get("tags", [])}:
            return False
        return not any(str(t).lower() in blocked for t in recipe.get("ingredients", []))

    return _allowed


def summarize(memory: Mapping[str, Any] | None) -> str:
    """One plain line describing the household, for the agent's system prompt."""
    memory = memory or {}
    parts: list[str] = []
    if memory.get("diet") and memory["diet"] != "any":
        parts.append(f"diet: {memory['diet']}")
    if memory.get("allergies"):
        parts.append(f"ALLERGIES (never suggest): {', '.join(_as_list(memory['allergies']))}")
    if memory.get("dislikes"):
        parts.append(f"dislikes: {', '.join(_as_list(memory['dislikes']))}")
    if memory.get("household_size"):
        parts.append(f"cooks for {memory['household_size']}")
    if memory.get("spice_level"):
        parts.append(f"spice: {memory['spice_level']}")
    if memory.get("favorite_cuisines"):
        parts.append(f"likes: {', '.join(_as_list(memory['favorite_cuisines']))}")
    if memory.get("goal"):
        parts.append(f"goal: {memory['goal']}")
    if memory.get("notes"):
        parts.append(f"notes: {'; '.join(_as_list(memory['notes']))}")
    return "; ".join(parts)


# --- Helpers -------------------------------------------------------------------


def _expand(entry: str) -> set[str]:
    """Map one allergy/dislike entry to ingredient tokens (group or single ingredient)."""
    text = str(entry).strip().lower()
    if not text:
        return set()
    group = ALLERGEN_GROUPS.get(text)
    if group is not None:
        return set(group)
    token = recommender.normalize_ingredient(text)
    return {token} if token else {text}


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _clean_text(value: Any) -> str:
    """Collapse whitespace and strip control characters; reject over-long values."""
    text = " ".join(str(value if value is not None else "").split())
    text = "".join(ch for ch in text if ch.isprintable())
    if len(text) > MAX_TEXT:
        raise ValueError(f"Keep each remembered value under {MAX_TEXT} characters.")
    return text
