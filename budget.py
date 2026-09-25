"""Budget layer for the Autopilot meal planner — rough grocery costs, kept honest.

The zero-waste :mod:`optimizer` decides *which* dishes to cook; this module answers the
follow-up a real household always asks: **what will the shopping cost, and can I stay under
a budget?** It does two small, framework-agnostic jobs:

* :func:`price_of` maps a canonical ingredient token (or a display name — it re-normalises
  through :mod:`recommender`) to an **estimated per-unit retail price**. The estimates live
  in a curated table covering the whole bundled recipe corpus; anything unknown falls back to
  a single sane default. These are deliberately labelled *estimates*, not live prices — there
  is no store integration, and the UI says so.
* :func:`group_by_day` folds the optimizer's flat meal list into day buckets so the
  Autopilot view can show "a week, one card per day" instead of one long list.

Prices are in the household currency (``APP_CURRENCY``, default ``₹`` to match the corpus of
Indian home cooking). Pantry staples are treated as already owned, so they cost nothing to a
plan — you don't re-buy salt to make dinner. The optimizer consumes :func:`price_of` through
its injected ``cost_of`` hook, so the planner can both *report* a plan's cost and *constrain*
the shop to a ceiling; this module never imports the optimizer, keeping the dependency
one-way (web/agent → optimizer → budget hook).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import recommender

# The currency symbol is cosmetic and configurable; the numbers are the same either way.
CURRENCY = os.environ.get("APP_CURRENCY", "₹").strip() or "₹"

# Fallback for any cookable ingredient not in the table below (the corpus is fully covered,
# so this only bites if the recipe data grows). A middling vegetable price.
_DEFAULT_PRICE = 20.0

# Estimated retail price per unit *as a dish uses it* — one block of paneer, one onion, a cup
# of rice, and so on. Rounded, plausible grocery prices; this is guidance, not a live quote.
# Covers every non-pantry token in ``data/recipes.json`` (see the corpus vocabulary).
_PRICE_TABLE: dict[str, float] = {
    # Dairy & proteins (the expensive end — where a budget actually bites).
    "paneer": 80.0, "cheese": 60.0, "chicken": 90.0, "tofu": 50.0, "egg": 6.0,
    "milk": 25.0, "cream": 40.0, "yogurt": 30.0, "butter": 30.0,
    # Grains, pulses & other staples.
    "rice": 20.0, "flour": 15.0, "besan": 20.0, "semolina": 15.0, "poha": 12.0,
    "oats": 20.0, "bread": 30.0, "pasta": 40.0, "noodles": 25.0,
    "lentil": 25.0, "chickpea": 25.0, "rajma": 30.0, "peanut": 20.0,
    # Everyday vegetables & aromatics.
    "onion": 15.0, "tomato": 15.0, "potato": 12.0, "garlic": 10.0, "ginger": 10.0,
    "green_chili": 5.0, "capsicum": 25.0, "carrot": 12.0, "peas": 20.0, "beans": 20.0,
    "cabbage": 18.0, "cauliflower": 25.0, "broccoli": 40.0, "spinach": 15.0,
    "okra": 25.0, "eggplant": 20.0, "mushroom": 45.0, "corn": 20.0, "cucumber": 12.0,
    # Herbs, garnish, fruit & sweeteners.
    "coriander": 8.0, "lemon": 5.0, "mint": 8.0,
    "banana": 8.0, "apple": 20.0, "mango": 40.0, "orange": 15.0, "grapes": 40.0,
    "honey": 40.0,
}


def price_of(name: str) -> float:
    """Estimated price for one unit of an ingredient, in the household currency.

    Accepts either a canonical token (``green_chili``) or a display name (``Green chili``);
    both resolve to the same entry. Pantry staples are assumed already owned and cost 0. An
    unknown cookable ingredient falls back to :data:`_DEFAULT_PRICE`.
    """
    token = recommender.normalize_ingredient(str(name or ""))
    if not token:
        return 0.0
    if token in recommender.PANTRY:
        return 0.0
    return _PRICE_TABLE.get(token, _DEFAULT_PRICE)


def price_tokens(tokens: Sequence[str]) -> float:
    """Total estimated price for a collection of ingredient tokens/names."""
    return round(sum(price_of(t) for t in tokens), 2)


def group_by_day(
    plan: Sequence[Mapping[str, Any]], meals_per_day: int
) -> list[dict[str, Any]]:
    """Fold the optimizer's flat, slot-ordered meal list into per-day buckets.

    ``meals_per_day`` meals go in day 1, the next ``meals_per_day`` in day 2, and so on. The
    final day may hold fewer meals if the plan couldn't fill every slot. Each bucket also
    carries the day's estimated shopping cost (its meals' ``est_cost``) so the UI can show a
    per-day spend without re-pricing.
    """
    per_day = max(1, int(meals_per_day or 1))
    days: list[dict[str, Any]] = []
    for meal in plan:
        if not days or len(days[-1]["meals"]) >= per_day:
            days.append({"day": len(days) + 1, "meals": [], "est_cost": 0.0})
        bucket = days[-1]
        bucket["meals"].append(meal)
        cost = meal.get("est_cost")
        if isinstance(cost, (int, float)):
            bucket["est_cost"] = round(bucket["est_cost"] + float(cost), 2)
    return days
