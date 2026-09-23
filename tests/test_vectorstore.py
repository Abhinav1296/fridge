"""Unit tests for the semantic recipe search index (vectorstore.py).

Hermetic: these use a hand-built embedding space and a tiny recipe corpus so the
cosine ordering is fully controlled and independent of the real trained vectors.
"""

from __future__ import annotations

import math

import vectorstore

# --- A small, controlled embedding space ------------------------------------
# Chosen so paneer/cream point one way, tomato/onion another, rice a third axis.
_VECS = {
    "paneer": [1.0, 0.0, 0.0],
    "cream": [0.9, 0.1, 0.0],
    "tomato": [0.0, 1.0, 0.0],
    "onion": [0.0, 0.9, 0.1],
    "rice": [0.0, 0.0, 1.0],
}


def _embed(tokens):
    present = [_VECS[t] for t in tokens if t in _VECS]
    if not present:
        return None
    mean = [sum(col) / len(present) for col in zip(*present)]
    norm = math.sqrt(sum(v * v for v in mean))
    return [v / norm for v in mean] if norm else None


def _normalize(text):
    return str(text or "").strip().lower().replace(" ", "_")


_RECIPES = [
    {
        "id": "paneer_curry",
        "title": "Paneer Curry",
        "ingredients": ["paneer", "cream", "tomato"],
        "tags": ["dinner", "curry"],
        "time_min": 30,
    },
    {
        "id": "tomato_rice",
        "title": "Tomato Rice",
        "ingredients": ["tomato", "onion", "rice"],
        "tags": ["lunch", "quick"],
        "time_min": 20,
    },
    {
        "id": "plain_rice",
        "title": "Plain Rice",
        "ingredients": ["rice"],
        "tags": ["side", "quick"],
        "time_min": 15,
    },
]


def _searcher(embed=_embed):
    return vectorstore.RecipeSearcher(_RECIPES, embed=embed, normalize=_normalize)


# --- VectorIndex ------------------------------------------------------------


def test_index_query_ranks_by_cosine_and_normalizes():
    idx = vectorstore.VectorIndex()
    idx.add("x", [2.0, 0.0, 0.0])  # not unit — must be normalized on insert
    idx.add("y", [0.0, 1.0, 0.0])
    idx.add("z", [0.0, 0.0, 1.0])

    ranked = idx.query([1.0, 0.0, 0.0], k=3)
    assert [rid for rid, _ in ranked] == ["x", "y", "z"]
    assert ranked[0][1] == 1.0  # stored vector was normalized, so cosine is exactly 1
    assert len(idx) == 3
    assert "x" in idx and "missing" not in idx


def test_index_query_respects_k_and_exclude():
    idx = vectorstore.VectorIndex()
    idx.add("x", [1.0, 0.0])
    idx.add("y", [0.9, 0.1])
    assert [rid for rid, _ in idx.query([1.0, 0.0], k=1)] == ["x"]
    assert [rid for rid, _ in idx.query([1.0, 0.0], k=5, exclude={"x"})] == ["y"]


def test_index_zero_query_returns_nothing():
    idx = vectorstore.VectorIndex()
    idx.add("x", [1.0, 0.0])
    assert idx.query([0.0, 0.0]) == []


def test_index_vector_for_and_metadata():
    idx = vectorstore.VectorIndex()
    idx.add("x", [3.0, 4.0], {"title": "X"})
    vec = idx.vector_for("x")
    assert vec is not None and abs(math.sqrt(sum(v * v for v in vec)) - 1.0) < 1e-9
    assert idx.metadata("x") == {"title": "X"}
    assert idx.vector_for("missing") is None
    assert idx.metadata("missing") == {}


# --- RecipeSearcher.search --------------------------------------------------


def test_search_ranks_semantically_closest_first():
    results = _searcher().search("paneer", k=3)
    assert results[0]["id"] == "paneer_curry"
    assert "paneer" in results[0]["matched"]
    assert results[0]["similarity"] > 0.0


def test_search_lexical_fallback_when_no_ingredient_tokens():
    # "quick" is a tag, not an ingredient in the embedding vocab, so ranking is
    # purely lexical — both quick recipes match, neither gets a semantic score.
    results = _searcher().search("quick")
    ids = {r["id"] for r in results}
    assert ids == {"tomato_rice", "plain_rice"}
    for r in results:
        assert r["similarity"] == 0.0
        assert "quick" in r["matched_terms"]


def test_search_blank_query_returns_empty():
    assert _searcher().search("   ") == []


def test_search_respects_k():
    assert len(_searcher().search("tomato rice", k=1)) == 1


def test_search_uses_injected_prettify():
    searcher = vectorstore.RecipeSearcher(
        _RECIPES, embed=_embed, normalize=_normalize, prettify=str.title
    )
    results = searcher.search("paneer", k=1)
    assert results[0]["matched"] == ["Paneer"]


# --- RecipeSearcher.similar -------------------------------------------------


def test_similar_excludes_self_and_reports_shared_ingredients():
    results = _searcher().similar("paneer_curry", k=5)
    ids = [r["id"] for r in results]
    assert "paneer_curry" not in ids
    tomato_rice = next(r for r in results if r["id"] == "tomato_rice")
    assert "tomato" in tomato_rice["matched"]  # the shared ingredient


def test_similar_unknown_recipe_is_empty():
    assert _searcher().similar("does_not_exist") == []


# --- Graceful degradation without embeddings --------------------------------


def test_searcher_without_embeddings_still_searches_lexically():
    searcher = _searcher(embed=lambda tokens: None)
    assert searcher.embedded is False
    assert len(searcher.index) == 0
    # Lexical search still works...
    results = searcher.search("quick")
    assert {r["id"] for r in results} == {"tomato_rice", "plain_rice"}
    # ...but similarity needs vectors, so there are none.
    assert searcher.similar("paneer_curry") == []
