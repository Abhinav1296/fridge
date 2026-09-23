"""Tests for :mod:`recommender` — normalization, structure, and use-soon behavior.

These are pure-function tests: no database, no network. A fixed ``now`` is passed so the
expiry math is deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

import recommender


def test_normalize_ingredient_maps_aliases():
    # Aliases from data/ingredient_aliases.json should collapse to a canonical token.
    assert recommender.normalize_ingredient("Curd") == "yogurt"
    # Plain names normalize to a lowercased token regardless of casing/whitespace.
    assert recommender.normalize_ingredient("  Tomato ") == "tomato"


# Realistic messy inputs a user or the vision model might emit: regional names, typos,
# and brand/qualifier noise. These lock in the normalization robustness layer (aliases +
# descriptor strip + fuzzy snap) so the app keeps landing on a token that has a vector.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # regional / synonym aliases
        ("pyaaz", "onion"),
        ("dhaniya", "coriander"),
        ("aloo", "potato"),
        ("doodh", "milk"),
        ("jeera", "cumin"),
        ("haldi", "turmeric"),
        ("shimla mirch", "capsicum"),
        ("beaten rice", "poha"),
        # descriptor / brand stripping
        ("fresh ginger", "ginger"),
        ("boiled egg", "egg"),
        ("amul butter", "butter"),
        ("full cream milk", "milk"),
        ("paneer cubes", "paneer"),
        # typo fuzzy-snap
        ("corriander", "coriander"),
        ("panner", "paneer"),
        ("capsicm", "capsicum"),
        ("buttr", "butter"),
        ("grean chilli", "green_chili"),
    ],
)
def test_normalize_ingredient_is_robust_to_messy_input(raw, expected):
    assert recommender.normalize_ingredient(raw) == expected


def test_normalize_ingredient_does_not_mis_snap_distinct_items():
    # The fuzzy layer must not map a genuinely distinct ingredient onto a lookalike, and
    # gibberish must pass through untouched rather than snapping to something unrelated.
    assert recommender.normalize_ingredient("sour cream") == "sour_cream"
    assert recommender.normalize_ingredient("xqzptvw") == "xqzptvw"
    # 'broccoli' must never collapse to 'cauliflower'.
    assert recommender.normalize_ingredient("broccoli") == "broccoli"


def test_recommend_returns_expected_shape_for_empty_fridge():
    result = recommender.recommend([], [])
    assert set(result) >= {"generated_at", "use_soon", "recipes", "shopping", "ml"}
    assert result["use_soon"] == []
    assert result["recipes"] == []
    assert isinstance(result["ml"], dict)
    assert result["ml"]["backend"] in {"embedding", "lexical"}


def test_recommend_surfaces_recipes_from_inventory():
    inventory = [
        {"name": "Paneer", "count": 1, "freshness": "fresh"},
        {"name": "Tomato", "count": 3, "freshness": "fresh"},
        {"name": "Onion", "count": 2, "freshness": "fresh"},
    ]
    result = recommender.recommend(inventory, [])
    # With paneer + tomato + onion on hand, at least one recipe should be suggested,
    # and every recipe row should carry the standard fields.
    assert len(result["recipes"]) >= 1
    row = result["recipes"][0]
    assert {"title", "missing"} <= set(row)


def test_use_by_date_moves_item_into_use_soon():
    now = datetime(2026, 9, 23, tzinfo=UTC)
    perishables = [{"name": "Paneer", "use_by": "2026-09-25"}]  # 2 days out
    result = recommender.recommend([], perishables, now=now, use_soon_days=5)

    tokens = {row["token"] for row in result["use_soon"]}
    assert "paneer" in tokens
    paneer = next(r for r in result["use_soon"] if r["token"] == "paneer")
    assert paneer["severity"] in {"soon", "overdue", "spoiled"}


def test_far_off_use_by_is_not_use_soon():
    now = datetime(2026, 9, 23, tzinfo=UTC)
    perishables = [{"name": "Paneer", "use_by": "2026-12-31"}]  # far away
    result = recommender.recommend([], perishables, now=now, use_soon_days=3)
    assert all(row["token"] != "paneer" for row in result["use_soon"])
