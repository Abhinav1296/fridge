"""Tests for the zero-waste meal-plan optimizer.

These are hermetic: they call :func:`optimizer.plan_meals` directly with in-memory
inventory (no database, no network). Each behavioural test runs against *both* solver
engines — the greedy heuristic (always available) and the exact ILP (skipped when PuLP is
not installed) — so the two paths are held to the same contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import budget
import optimizer
import recommender

# Run every parametrized test once per available engine.
_ENGINES = ["greedy"]
if optimizer._HAS_PULP:
    _ENGINES.append("ilp")

NOW = datetime(2026, 9, 23, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _fresh_corpus():
    """Ensure the recommender's cached recipe corpus is loaded fresh for each test."""
    recommender.reset_cache()
    yield
    recommender.reset_cache()


def _items(*names):
    return [{"name": n} for n in names]


def test_empty_fridge_returns_empty_plan():
    result = optimizer.plan_meals([], [], now=NOW, days=2, meals_per_day=2)
    assert result["plan"] == []
    assert result["shopping_list"] == []
    assert result["metrics"]["slots_filled"] == 0
    assert result["horizon"]["slots"] == 4
    assert any("No cookable plan" in n for n in result["notes"])


@pytest.mark.parametrize("engine", _ENGINES)
def test_plan_shape_and_slot_cap(engine):
    result = optimizer.plan_meals(
        _items("paneer", "tomato", "onion", "rice", "dal"),
        now=NOW,
        days=1,
        meals_per_day=3,
        solver=engine,
    )
    assert result["solver"] == engine
    # Never schedules more dishes than slots.
    assert len(result["plan"]) <= result["horizon"]["slots"] == 3
    # Every plan row carries the public contract keys.
    for row in result["plan"]:
        assert {"slot", "id", "title", "uses", "uses_expiring", "to_buy"} <= set(row)
    # Metrics block is complete.
    assert {
        "slots_filled",
        "urgent_total",
        "urgent_used",
        "waste_avoided_pct",
        "new_ingredients_to_buy",
        "objective",
    } <= set(result["metrics"])


@pytest.mark.parametrize("engine", _ENGINES)
def test_scarce_resource_not_overcommitted(engine):
    # One block of paneer, four slots: no more than one scheduled dish may consume paneer.
    result = optimizer.plan_meals(
        [{"name": "paneer", "count": 1}, {"name": "tomato", "count": 3},
         {"name": "onion", "count": 3}, {"name": "spinach", "count": 2}],
        now=NOW,
        days=2,
        meals_per_day=2,
        solver=engine,
    )
    paneer_users = sum(
        1 for row in result["plan"] if "Paneer" in row["uses"]
    )
    assert paneer_users <= 1


@pytest.mark.parametrize("engine", _ENGINES)
def test_prioritizes_expiring_ingredient(engine):
    # Paneer is due tomorrow; the plan should schedule a dish that uses it up.
    result = optimizer.plan_meals(
        _items("paneer", "tomato", "onion"),
        perishables=[{"name": "paneer", "use_by": "2026-09-24"}],
        now=NOW,
        days=1,
        meals_per_day=2,
        solver=engine,
    )
    assert result["metrics"]["urgent_total"] >= 1
    assert result["metrics"]["urgent_used"] >= 1
    assert result["metrics"]["waste_avoided_pct"] is not None
    # Paneer shows up as an expiring ingredient consumed by some scheduled dish.
    assert any("Paneer" in row["uses_expiring"] for row in result["plan"])


@pytest.mark.parametrize("engine", _ENGINES)
def test_spoiled_item_is_not_cooked(engine):
    result = optimizer.plan_meals(
        [{"name": "paneer", "freshness": "spoiled"},
         {"name": "tomato"}, {"name": "onion"}, {"name": "rice"}],
        now=NOW,
        solver=engine,
    )
    # Spoiled paneer must never appear as an ingredient the plan uses.
    assert all("Paneer" not in row["uses"] for row in result["plan"])


@pytest.mark.parametrize("engine", _ENGINES)
def test_shopping_list_aggregates_missing(engine):
    result = optimizer.plan_meals(
        _items("tomato", "onion", "rice"),
        now=NOW,
        days=2,
        meals_per_day=2,
        solver=engine,
    )
    # Anything a scheduled dish is missing appears in the shopping list, and every
    # shopping entry is referenced by at least one planned recipe.
    planned_titles = {row["title"] for row in result["plan"]}
    for entry in result["shopping_list"]:
        assert entry["count"] == len(entry["for_recipes"]) >= 1
        assert set(entry["for_recipes"]) <= planned_titles


@pytest.mark.parametrize("engine", _ENGINES)
def test_nutrition_summary_when_cache_supplied(engine):
    nutrition = {
        "paneer": {"kcal": 265.0, "protein_g": 18.0, "carbs_g": 1.2, "fat_g": 21.0},
        "tomato": {"kcal": 18.0, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2},
    }
    result = optimizer.plan_meals(
        _items("paneer", "tomato", "onion"),
        now=NOW,
        nutrition=nutrition,
        solver=engine,
    )
    summary = result["metrics"].get("nutrition")
    if result["plan"]:
        assert summary is not None
        assert summary["kcal"] > 0
        assert summary["ingredients_with_data"] >= 1


@pytest.mark.parametrize("engine", _ENGINES)
def test_no_nutrition_when_no_cache(engine):
    result = optimizer.plan_meals(_items("paneer", "tomato", "onion"), now=NOW, solver=engine)
    assert "nutrition" not in result["metrics"]


@pytest.mark.skipif(not optimizer._HAS_PULP, reason="PuLP not installed")
def test_ilp_matches_or_beats_greedy_objective():
    # On the same input the exact solver's objective should be >= the heuristic's.
    inv = _items("paneer", "tomato", "onion", "rice", "dal", "spinach", "potato")
    perishables = [{"name": "paneer", "use_by": "2026-09-24"},
                   {"name": "spinach", "use_by": "2026-09-24"}]
    greedy = optimizer.plan_meals(inv, perishables, now=NOW, days=2, meals_per_day=2, solver="greedy")
    ilp = optimizer.plan_meals(inv, perishables, now=NOW, days=2, meals_per_day=2, solver="ilp")
    assert ilp["metrics"]["objective"] >= greedy["metrics"]["objective"] - 1e-6


# --- Budget layer (Autopilot pricing) ----------------------------------------
#
# When a ``cost_of`` hook is supplied the plan is priced; with ``max_budget`` the shop is
# constrained. Both engines must respect the ceiling and expose the same cost contract.


@pytest.mark.parametrize("engine", _ENGINES)
def test_pricing_is_off_by_default(engine):
    # No cost_of hook → no cost fields anywhere (backwards-compatible payload).
    result = optimizer.plan_meals(
        _items("paneer", "tomato", "onion", "rice"), now=NOW, solver=engine
    )
    assert "currency" not in result["metrics"]
    assert "shopping_cost" not in result["metrics"]
    for meal in result["plan"]:
        assert "est_cost" not in meal
    for row in result["shopping_list"]:
        assert "est_cost" not in row


@pytest.mark.parametrize("engine", _ENGINES)
def test_cost_of_hook_prices_the_whole_plan(engine):
    result = optimizer.plan_meals(
        _items("paneer", "tomato", "onion", "rice"),
        now=NOW,
        days=2,
        meals_per_day=2,
        solver=engine,
        cost_of=budget.price_of,
    )
    m = result["metrics"]
    assert m["currency"] == budget.CURRENCY
    assert m["shopping_cost"] >= 0.0
    assert m["pantry_value_used"] >= 0.0
    # Each meal is priced; est_cost sums its to-buy items, pantry_value its on-hand items.
    for meal in result["plan"]:
        assert isinstance(meal["est_cost"], (int, float)) and meal["est_cost"] >= 0.0
        assert isinstance(meal["pantry_value"], (int, float)) and meal["pantry_value"] >= 0.0
    # Each shopping row is priced.
    for row in result["shopping_list"]:
        assert isinstance(row["est_cost"], (int, float)) and row["est_cost"] >= 0.0
    # The shop's cost counts each distinct bought ingredient once — so it never exceeds the
    # naive sum of per-meal est_cost (which would double-count a shared new ingredient).
    assert m["shopping_cost"] <= sum(meal["est_cost"] for meal in result["plan"]) + 1e-6


@pytest.mark.parametrize("engine", _ENGINES)
def test_shopping_cost_equals_distinct_bought_prices(engine):
    result = optimizer.plan_meals(
        _items("tomato", "onion", "rice"),
        now=NOW,
        days=2,
        meals_per_day=2,
        solver=engine,
        cost_of=budget.price_of,
    )
    # The reported shopping_cost is exactly the sum of the priced shopping-list rows.
    expected = round(sum(row["est_cost"] for row in result["shopping_list"]), 2)
    assert result["metrics"]["shopping_cost"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("engine", _ENGINES)
def test_max_budget_is_a_hard_ceiling(engine):
    # A stingy budget must not be exceeded, whichever engine plans, and the plan is flagged
    # as within budget.
    result = optimizer.plan_meals(
        _items("tomato", "onion", "rice", "potato"),
        now=NOW,
        days=3,
        meals_per_day=2,
        solver=engine,
        cost_of=budget.price_of,
        max_budget=15.0,
    )
    m = result["metrics"]
    assert m["budget"] == 15.0
    assert m["shopping_cost"] <= 15.0 + 1e-6
    assert m["within_budget"] is True


@pytest.mark.parametrize("engine", _ENGINES)
def test_zero_budget_forbids_any_shopping(engine):
    # With a zero budget, no dish that requires buying anything (priced > 0) may be scheduled;
    # every planned meal must be fully cookable from what's on hand.
    result = optimizer.plan_meals(
        _items("tomato", "onion", "rice", "potato", "spinach"),
        now=NOW,
        days=3,
        meals_per_day=2,
        solver=engine,
        cost_of=budget.price_of,
        max_budget=0.0,
    )
    assert result["metrics"]["shopping_cost"] == pytest.approx(0.0, abs=0.01)
    for meal in result["plan"]:
        assert meal["est_cost"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("engine", _ENGINES)
def test_budget_ignored_without_cost_hook(engine):
    # max_budget alone (no cost_of) is a no-op: the plan is unpriced and unconstrained.
    result = optimizer.plan_meals(
        _items("tomato", "onion", "rice"),
        now=NOW,
        solver=engine,
        max_budget=1.0,
    )
    assert "budget" not in result["metrics"]
    assert "shopping_cost" not in result["metrics"]


@pytest.mark.skipif(not optimizer._HAS_PULP, reason="PuLP not installed")
def test_tighter_budget_never_raises_shopping_cost():
    # Monotonicity: shrinking the budget can only lower (or hold) the exact solver's spend.
    inv = _items("tomato", "onion", "rice", "potato", "spinach", "paneer")
    loose = optimizer.plan_meals(
        inv, now=NOW, days=3, meals_per_day=2, solver="ilp",
        cost_of=budget.price_of, max_budget=200.0,
    )
    tight = optimizer.plan_meals(
        inv, now=NOW, days=3, meals_per_day=2, solver="ilp",
        cost_of=budget.price_of, max_budget=30.0,
    )
    assert tight["metrics"]["shopping_cost"] <= loose["metrics"]["shopping_cost"] + 1e-6
    assert tight["metrics"]["shopping_cost"] <= 30.0 + 1e-6
