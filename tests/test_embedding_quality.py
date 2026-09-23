"""Golden-quality guardrails for the shipped ingredient embeddings.

These run against the committed ``data/ingredient_embeddings.json`` using the standard
library only (no numpy/sklearn, no 2 GB corpus), so they run in CI. They lock in the two
things that must never silently regress when the model is re-shipped:

1. **Coverage** — every ingredient used by an app recipe must have a trained vector, or
   recommendation quietly falls back to lexical-only for that item.
2. **Semantics** — relative similarities that encode real culinary structure (paneer is
   nearer a curry than a dessert, etc.). These are *relative* comparisons, so they stay
   valid across re-trains as long as the space is sane.

If the embeddings file is absent the recommender degrades to lexical ranking by design;
these tests then skip rather than fail.
"""

from __future__ import annotations

import math

import pytest

import recommender


@pytest.fixture(autouse=True)
def _fresh_caches():
    recommender.reset_cache()
    yield
    recommender.reset_cache()


def _embeddings_or_skip() -> dict:
    emb = recommender._load_embeddings()
    if not emb["vectors"]:
        pytest.skip("no trained embeddings shipped; recommender is in lexical mode")
    return emb


def test_embeddings_have_consistent_dimension_and_unit_norm():
    emb = _embeddings_or_skip()
    dim = emb["dim"]
    assert dim > 0
    # Spot-check a slice: every stored vector is dim-long and L2-normalized (norm≈1).
    for _token, vec in list(emb["vectors"].items())[:200]:
        assert len(vec) == dim
        assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, abs_tol=1e-3)


def test_every_app_recipe_ingredient_has_a_vector():
    emb = _embeddings_or_skip()
    vectors = emb["vectors"]
    app_tokens = {
        recommender.normalize_ingredient(t)
        for recipe in recommender.all_recipes()
        for t in recipe["ingredients"]
    }
    missing = sorted(t for t in app_tokens if t not in vectors)
    assert not missing, f"app ingredients without a vector: {missing}"


def test_paneer_is_closer_to_curry_than_dessert():
    _embeddings_or_skip()
    paneer = recommender.embed_ingredients(["paneer"])
    curry = recommender.embed_ingredients(["garam_masala", "turmeric", "cumin"])
    dessert = recommender.embed_ingredients(["sugar", "flour", "butter"])
    assert recommender._cosine(paneer, curry) > recommender._cosine(paneer, dessert)


def test_aromatics_cluster_together():
    _embeddings_or_skip()
    onion = recommender.embed_ingredients(["onion"])
    garlic = recommender.embed_ingredients(["garlic"])
    sugar = recommender.embed_ingredients(["sugar"])
    # An aromatic base ingredient should sit nearer another aromatic than a sweetener.
    assert recommender._cosine(onion, garlic) > recommender._cosine(onion, sugar)


def test_embedding_lookup_is_deterministic():
    _embeddings_or_skip()
    first = recommender.embed_ingredients(["paneer", "onion", "tomato"])
    second = recommender.embed_ingredients(["paneer", "onion", "tomato"])
    assert first == second  # exact, order-stable pure-Python mean


def test_recommender_uses_the_embedding_backend_when_vectors_present():
    _embeddings_or_skip()
    result = recommender.recommend([{"name": "paneer"}, {"name": "onion"}], [])
    assert result["ml"]["backend"] == "embedding"
    assert result["ml"]["dim"] == recommender._load_embeddings()["dim"]
