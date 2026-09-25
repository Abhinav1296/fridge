"""Unit tests for :mod:`budget` — the Autopilot pricing + day-grouping layer.

These are plain, framework-agnostic unit tests (no Flask/HTTP, no database). They pin down
the small contract the optimizer and the UI depend on:

* :func:`budget.price_of` maps a canonical token *or* a display name to a non-negative
  estimate, treats pantry staples as free, and covers every non-pantry ingredient in the
  bundled corpus (so a real plan is always fully priced);
* :func:`budget.price_tokens` sums a collection; and
* :func:`budget.group_by_day` folds the optimizer's flat meal list into per-day buckets and
  totals each day's estimated cost.
"""

from __future__ import annotations

import pytest

import budget
import recommender

# --- price_of ----------------------------------------------------------------


def test_pantry_staples_are_free():
    # Salt/cumin are always-on-hand pantry staples: they cost a plan nothing.
    for staple in ("salt", "cumin", "turmeric", "oil"):
        if staple in recommender.PANTRY:
            assert budget.price_of(staple) == 0.0


def test_empty_or_blank_name_is_free():
    assert budget.price_of("") == 0.0
    assert budget.price_of("   ") == 0.0


def test_display_names_and_tokens_price_identically():
    # "Green chili" normalises to the same token as "green_chili".
    assert budget.price_of("Green chili") == budget.price_of("green_chili")
    assert budget.price_of("Paneer") == budget.price_of("paneer")
    assert budget.price_of("PANEER") == budget.price_of("paneer")


def test_known_ingredient_uses_the_table_price():
    assert budget.price_of("paneer") == budget._PRICE_TABLE["paneer"]
    assert budget.price_of("egg") == budget._PRICE_TABLE["egg"]


def test_unknown_cookable_ingredient_falls_back_to_default():
    # A cookable token that isn't pantry and isn't in the table gets the sane default.
    token = "dragonfruit"
    assert token not in recommender.PANTRY
    assert token not in budget._PRICE_TABLE
    assert budget.price_of(token) == budget._DEFAULT_PRICE


def test_every_corpus_ingredient_is_priced_non_negative():
    # No cookable ingredient in the shipped corpus should be unpriced or negative.
    seen = set()
    for recipe in recommender.all_recipes():
        for name in recipe.get("ingredients", []):
            token = recommender.normalize_ingredient(str(name))
            if not token or token in recommender.PANTRY:
                continue
            seen.add(token)
            assert budget.price_of(token) >= 0.0
    # Sanity: the corpus actually exercised a decent slice of the price table.
    assert len(seen) >= 20


def test_proteins_cost_more_than_common_vegetables():
    # A crude ordering check so the estimates stay economically plausible.
    assert budget.price_of("chicken") > budget.price_of("onion")
    assert budget.price_of("paneer") > budget.price_of("potato")


# --- price_tokens ------------------------------------------------------------


def test_price_tokens_sums_and_rounds():
    total = budget.price_tokens(["paneer", "tomato", "onion"])
    expected = round(
        budget.price_of("paneer") + budget.price_of("tomato") + budget.price_of("onion"), 2
    )
    assert total == expected


def test_price_tokens_empty_is_zero():
    assert budget.price_tokens([]) == 0.0


def test_price_tokens_ignores_pantry_cost():
    # Adding a pantry staple to the list doesn't change the total.
    base = budget.price_tokens(["paneer", "tomato"])
    with_staple = budget.price_tokens(["paneer", "tomato", "salt"])
    assert with_staple == base


# --- group_by_day ------------------------------------------------------------


def _meal(slot, est_cost=0.0):
    return {"slot": slot, "title": f"Dish {slot}", "est_cost": est_cost}


def test_group_by_day_buckets_in_order():
    plan = [_meal(i) for i in range(1, 7)]
    days = budget.group_by_day(plan, meals_per_day=2)
    assert [d["day"] for d in days] == [1, 2, 3]
    assert all(len(d["meals"]) == 2 for d in days)
    # Slots stay in their original order within and across days.
    flat = [m["slot"] for d in days for m in d["meals"]]
    assert flat == [1, 2, 3, 4, 5, 6]


def test_group_by_day_last_day_may_be_short():
    plan = [_meal(i) for i in range(1, 6)]  # 5 meals, 2 per day → 3/2? no: 2,2,1
    days = budget.group_by_day(plan, meals_per_day=2)
    assert [len(d["meals"]) for d in days] == [2, 2, 1]


def test_group_by_day_totals_est_cost_per_day():
    plan = [_meal(1, 10.0), _meal(2, 5.0), _meal(3, 20.0)]
    days = budget.group_by_day(plan, meals_per_day=2)
    assert days[0]["est_cost"] == pytest.approx(15.0)
    assert days[1]["est_cost"] == pytest.approx(20.0)


def test_group_by_day_handles_missing_cost_gracefully():
    # A meal with no est_cost (unpriced plan) contributes 0, never crashes.
    plan = [{"slot": 1, "title": "x"}, {"slot": 2, "title": "y", "est_cost": 8.0}]
    days = budget.group_by_day(plan, meals_per_day=3)
    assert len(days) == 1
    assert days[0]["est_cost"] == pytest.approx(8.0)


def test_group_by_day_empty_plan_is_empty():
    assert budget.group_by_day([], meals_per_day=2) == []


def test_group_by_day_clamps_bad_meals_per_day():
    # A zero/None meals_per_day is treated as 1 rather than dividing by zero.
    plan = [_meal(1), _meal(2)]
    days = budget.group_by_day(plan, meals_per_day=0)
    assert [len(d["meals"]) for d in days] == [1, 1]
