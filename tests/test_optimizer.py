"""Tests for the zero-waste meal-plan optimizer.

These are hermetic: they call :func:`optimizer.plan_meals` directly with in-memory
inventory (no database, no network). Each behavioural test runs against *both* solver
engines — the greedy heuristic (always available) and the exact ILP (skipped when PuLP is
not installed) — so the two paths are held to the same contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

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
