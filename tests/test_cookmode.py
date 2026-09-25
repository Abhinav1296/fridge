"""Unit tests for :mod:`cookmode` — the guided, step-by-step cooking walkthrough.

Cook Mode *synthesises* a method-aware walkthrough from the structured recipe data (tags,
ingredients, time) rather than shipping hand-written instructions, and layers embedding-based
swap ideas on top. These tests pin down the contract the web layer and the UI rely on:

* the right **method** is chosen from a recipe's tags, then its hero ingredient;
* the **checklist** marks each ingredient have/missing against what's in the fridge, and
  pantry staples never get swap ideas;
* a **vegetarian** dish never proposes a meat/egg swap, while a **non-veg** dish may; and
* the **steps** are well-formed (numbered, timed) and the timer total is self-consistent.

The module is framework-agnostic, so these are plain unit tests with no Flask/HTTP. Swap-idea
tests that must be deterministic monkeypatch :func:`recommender.similar_ingredients` so they
don't depend on the trained embeddings' exact neighbours.
"""

from __future__ import annotations

import pytest

import cookmode
import recommender

# --- Fixtures: real corpus recipes, looked up once by id ---------------------


@pytest.fixture(scope="module")
def recipes() -> dict[str, dict]:
    """The bundled recipe corpus keyed by id, for picking known dishes by method."""
    return {r["id"]: r for r in recommender.all_recipes()}


# --- Method classification ---------------------------------------------------


def _synthetic(tags, ingredients, *, time_min=20, rid="synthetic", title="Synthetic"):
    """A minimal recipe dict — build_cook_session reads only these keys."""
    return {
        "id": rid,
        "title": title,
        "tags": list(tags),
        "ingredients": list(ingredients),
        "time_min": time_min,
    }


@pytest.mark.parametrize(
    ("tags", "ingredients", "expected_method"),
    [
        (["curry"], ["paneer", "tomato", "onion"], "curry"),
        (["dal"], ["lentil", "onion", "tomato"], "dal"),
        (["dry-sabzi"], ["potato", "cauliflower", "onion"], "dry-sabzi"),
        (["rice"], ["rice", "carrot", "peas"], "rice"),
        (["soup"], ["tomato", "onion", "garlic"], "soup"),
        (["grill"], ["paneer", "capsicum", "yogurt"], "grill"),
        (["drink"], ["banana", "milk", "honey"], "drink"),
    ],
)
def test_method_is_selected_from_the_type_tag(tags, ingredients, expected_method):
    session = cookmode.build_cook_session(_synthetic(tags, ingredients))
    assert session["method"] == expected_method


def test_curry_tag_wins_over_a_generic_meal_tag():
    # A recipe carries both a meal tag (dinner) and a method tag (curry); the method wins.
    session = cookmode.build_cook_session(
        _synthetic(["dinner", "curry", "vegetarian"], ["paneer", "tomato", "onion"])
    )
    assert session["method"] == "curry"


@pytest.mark.parametrize(
    ("hero", "expected_method"),
    [
        ("egg", "egg"),
        ("besan", "besan"),
        ("semolina", "semolina"),
        ("poha", "poha"),
        ("oats", "oats"),
        ("flour", "flour"),
        ("noodles", "noodles"),
        ("pasta", "pasta"),
        ("tofu", "tofu"),
    ],
)
def test_tagless_dishes_fall_back_to_the_hero_ingredient(hero, expected_method):
    # No method tag: the first (hero) ingredient picks the builder.
    session = cookmode.build_cook_session(_synthetic([], [hero, "onion", "tomato"]))
    assert session["method"] == expected_method


def test_unknown_tagless_dish_uses_the_generic_walkthrough():
    session = cookmode.build_cook_session(_synthetic([], ["jackfruit", "onion", "tomato"]))
    assert session["method"] == "generic"
    assert session["steps"]  # generic still yields a usable walkthrough


# --- Steps are well-formed ---------------------------------------------------


def test_steps_are_numbered_and_timer_total_is_consistent(recipes):
    session = cookmode.build_cook_session(recipes["paneer_butter_masala"])
    steps = session["steps"]
    assert steps, "a curry should produce steps"
    # Steps are numbered 1..N in order.
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1))
    # Every step carries the fields the UI renders.
    for s in steps:
        assert set(s) >= {"n", "title", "text", "seconds", "uses"}
        assert isinstance(s["title"], str) and s["title"]
        assert isinstance(s["text"], str) and s["text"]
        assert isinstance(s["seconds"], int) and s["seconds"] >= 0
        assert isinstance(s["uses"], list)
    # The advertised total equals the sum of the per-step timers.
    assert session["total_timer_seconds"] == sum(s["seconds"] for s in steps)
    assert session["total_timer_seconds"] > 0  # a cooked dish has at least one timed step


def test_every_corpus_recipe_builds_a_non_empty_session(recipes):
    # Nothing in the shipped corpus should blow up or produce an empty walkthrough.
    for rid, recipe in recipes.items():
        session = cookmode.build_cook_session(recipe)
        assert session["steps"], f"{rid} produced no steps"
        assert session["total_count"] == len(recipe["ingredients"])
        assert session["recipe_id"] == rid


# --- have / missing accounting -----------------------------------------------


def test_have_tokens_mark_the_checklist_and_counts_add_up(recipes):
    pbm = recipes["paneer_butter_masala"]  # paneer, tomato, onion, butter, cream, ginger, garlic
    session = cookmode.build_cook_session(pbm, have_tokens=["paneer", "tomato"])

    have = {i["token"] for i in session["ingredients"] if i["have"]}
    assert have == {"paneer", "tomato"}
    assert session["have_count"] == 2
    assert session["missing_count"] == session["total_count"] - 2
    assert session["have_count"] + session["missing_count"] == session["total_count"]


def test_have_tokens_are_matched_case_insensitively(recipes):
    session = cookmode.build_cook_session(
        recipes["paneer_butter_masala"], have_tokens=["Paneer", "TOMATO"]
    )
    assert {i["token"] for i in session["ingredients"] if i["have"]} == {"paneer", "tomato"}


def test_no_have_tokens_means_everything_is_missing(recipes):
    session = cookmode.build_cook_session(recipes["dal_tadka"])
    assert session["have_count"] == 0
    assert session["missing_count"] == session["total_count"]


# --- Substitutions -----------------------------------------------------------


def test_pantry_ingredients_never_get_swap_ideas():
    # 'cumin' and 'salt' are pantry staples: they're always assumed on hand and never swapped.
    assert "cumin" in recommender.PANTRY and "salt" in recommender.PANTRY
    session = cookmode.build_cook_session(
        _synthetic(["curry"], ["paneer", "cumin", "salt", "tomato"])
    )
    by_token = {i["token"]: i for i in session["ingredients"]}
    assert by_token["cumin"]["pantry"] is True
    assert by_token["cumin"]["substitutes"] == []
    assert by_token["salt"]["substitutes"] == []
    # A non-pantry ingredient is still eligible for swaps.
    assert by_token["paneer"]["pantry"] is False


def test_swap_ideas_are_well_formed_and_exclude_the_recipes_own_ingredients(recipes):
    session = cookmode.build_cook_session(recipes["paneer_butter_masala"])
    own = {i["token"] for i in session["ingredients"]}
    for ing in session["ingredients"]:
        for sub in ing["substitutes"]:
            assert set(sub) == {"token", "name", "similarity"}
            assert isinstance(sub["name"], str) and sub["name"]
            assert 0.0 <= sub["similarity"] <= 1.0
            assert sub["token"] not in own  # never suggest something already in the dish
            assert sub["token"] not in recommender.PANTRY  # nor a pantry staple


def test_explicit_exclude_tokens_never_surface_as_a_swap(monkeypatch, recipes):
    # Whatever the embeddings would suggest, an allergen/dislike must never appear.
    monkeypatch.setattr(
        recommender,
        "similar_ingredients",
        lambda token, k=3, candidates=None, min_similarity=0.0: [
            (c, 0.9) for c in sorted(candidates or ())
        ],
    )
    session = cookmode.build_cook_session(
        recipes["paneer_butter_masala"], exclude_tokens=["cheese", "carrot"]
    )
    suggested = {s["token"] for i in session["ingredients"] for s in i["substitutes"]}
    assert "cheese" not in suggested
    assert "carrot" not in suggested


def test_vegetarian_recipe_never_suggests_a_meat_or_egg_swap(recipes):
    # Across every vegetarian dish in the corpus, no swap idea is a meat/egg token.
    for rid, recipe in recipes.items():
        tags = {str(t).lower() for t in recipe.get("tags", [])}
        if "non-veg" in tags:
            continue
        session = cookmode.build_cook_session(recipe)
        suggested = {s["token"] for i in session["ingredients"] for s in i["substitutes"]}
        leaked = suggested & cookmode._NON_VEG
        assert not leaked, f"{rid} leaked meat/egg swaps: {sorted(leaked)}"


def test_vegetarian_dish_filters_meat_even_when_embeddings_offer_it(monkeypatch, recipes):
    # Force the embeddings to *want* to suggest egg; the veg guard must still drop it.
    monkeypatch.setattr(
        recommender,
        "similar_ingredients",
        lambda token, k=3, candidates=None, min_similarity=0.0: [
            (c, 0.9) for c in sorted(candidates or ())
        ],
    )
    veg = recipes["paneer_butter_masala"]
    assert "non-veg" not in {t.lower() for t in veg["tags"]}
    session = cookmode.build_cook_session(veg)
    suggested = {s["token"] for i in session["ingredients"] for s in i["substitutes"]}
    assert not (suggested & cookmode._NON_VEG)


def test_non_veg_dish_may_suggest_a_meat_swap(monkeypatch, recipes):
    # A non-veg recipe does NOT force-exclude meat, so a meat/egg neighbour can be suggested.
    monkeypatch.setattr(
        recommender,
        "similar_ingredients",
        lambda token, k=3, candidates=None, min_similarity=0.0: [
            (c, 0.9) for c in sorted(candidates or ())
        ],
    )
    chicken_curry = recipes["chicken_curry"]  # tagged non-veg; 'egg' is not one of its ingredients
    assert "non-veg" in {t.lower() for t in chicken_curry["tags"]}
    session = cookmode.build_cook_session(chicken_curry)
    suggested = {s["token"] for i in session["ingredients"] for s in i["substitutes"]}
    # 'egg' exists in the corpus, isn't in this recipe, and isn't a pantry token, so with the
    # stubbed embeddings it survives into the suggestions — proving meat isn't force-excluded.
    assert "egg" in suggested


# --- Metadata ----------------------------------------------------------------


def test_session_carries_the_metadata_the_ui_needs(recipes):
    session = cookmode.build_cook_session(recipes["dal_tadka"], servings=4)
    assert session["recipe_id"] == "dal_tadka"
    assert session["title"] == recipes["dal_tadka"]["title"]
    assert session["time_min"] == recipes["dal_tadka"]["time_min"]
    assert session["servings"] == 4
    assert "dal" in session["tags"]
    assert isinstance(session["note"], str) and session["note"]
