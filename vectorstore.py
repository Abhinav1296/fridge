"""Semantic recipe retrieval — a small in-process vector database.

Stores one unit vector per recipe (the mean of its ingredients' learned
embeddings; see :mod:`build_embeddings`) and answers cosine k-nearest-neighbour
queries in pure Python. This powers two things the fridge-aware recommender does
not: *semantic search* — a query for "creamy paneer dinner" surfaces creamy
paneer dishes even when the exact words don't overlap — and *"find similar
recipes"* from any starting dish.

Design goals, matching the rest of the codebase:

* **Framework-agnostic.** No Flask, no file IO, no model training here. The
  caller passes in the recipe corpus plus two callables — ``embed(tokens) -> unit
  vector | None`` and ``normalize(text) -> token`` — which :mod:`app` wires from
  :mod:`recommender` (the single source of truth for recipes and embeddings).
* **Dependency-light.** Pure Python, so the deployed app never needs numpy at
  request time (the vectors were computed offline).
* **Swappable.** :class:`VectorIndex` is a plain cosine store; if the corpus ever
  outgrows memory it can be replaced by a hosted vector DB (Chroma / pgvector)
  behind the same :meth:`VectorIndex.query` API without touching the searcher.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

# Tokenizer for free-text queries and recipe metadata: runs of letters/digits.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Blend weights when a query yields both a semantic vector and lexical overlap.
# Semantic similarity leads; lexical overlap on title/tags is a supporting signal.
_W_VECTOR = 0.65
_W_LEXICAL = 0.35


def _unit(vector: Sequence[float]) -> list[float]:
    """Return the L2-normalized vector, or all-zeros if it has no magnitude."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


class VectorIndex:
    """A minimal cosine-similarity vector store: add records, query k nearest.

    Vectors are unit-normalized on insertion and on query, so a plain dot product
    is the cosine similarity. Tiny by design — linear scan is ample for a few
    hundred recipes and keeps the code obvious.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._meta: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id: str) -> bool:
        return id in self._meta

    def add(self, id: str, vector: Sequence[float], metadata: dict[str, Any] | None = None) -> None:
        """Insert (or append) a record. Vectors are stored unit-normalized."""
        self._ids.append(id)
        self._vectors.append(_unit(vector))
        self._meta[id] = dict(metadata or {})

    def vector_for(self, id: str) -> list[float] | None:
        """Return the stored unit vector for ``id``, or None if it isn't indexed."""
        try:
            return self._vectors[self._ids.index(id)]
        except ValueError:
            return None

    def metadata(self, id: str) -> dict[str, Any]:
        """Return a copy of the metadata stored with ``id`` (empty if unknown)."""
        return dict(self._meta.get(id, {}))

    def query(
        self, vector: Sequence[float], k: int = 5, *, exclude: Iterable[str] = ()
    ) -> list[tuple[str, float]]:
        """Return the ``k`` nearest ``(id, cosine)`` pairs, highest similarity first.

        A zero query vector (nothing embeddable) yields no results. Ties break by
        id for deterministic ordering.
        """
        q = _unit(vector)
        if not any(q):
            return []
        skip = set(exclude)
        scored = [
            (id_, sum(a * b for a, b in zip(q, vec)))
            for id_, vec in zip(self._ids, self._vectors)
            if id_ not in skip
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(0, int(k))]


class RecipeSearcher:
    """Semantic + lexical recipe search over a fixed recipe corpus.

    Build once from the recipe list and the recommender's embedding math, then
    call :meth:`search` (free-text query) or :meth:`similar` (nearest recipes to a
    given one). Both return JSON-serializable dicts shaped for the recipe-card UI.
    """

    def __init__(
        self,
        recipes: Iterable[dict[str, Any]],
        *,
        embed: Callable[[list[str]], list[float] | None],
        normalize: Callable[[str], str],
        prettify: Callable[[str], str] | None = None,
    ) -> None:
        self._embed = embed
        self._normalize = normalize
        self._prettify = prettify or (lambda token: token)
        self.index = VectorIndex()
        self._by_id: dict[str, dict[str, Any]] = {}
        self._ingredients: dict[str, list[str]] = {}
        self._doc_words: dict[str, set[str]] = {}
        self._vocab: set[str] = set()

        for recipe in recipes:
            rid = str(recipe.get("id") or recipe.get("title") or "").strip()
            if not rid or rid in self._by_id:
                continue
            ingredients = [
                str(tok).strip().lower() for tok in recipe.get("ingredients", []) if str(tok).strip()
            ]
            if not ingredients:
                continue
            self._by_id[rid] = recipe
            self._ingredients[rid] = ingredients
            self._vocab.update(ingredients)

            vector = embed(ingredients)
            if vector:
                self.index.add(rid, vector, {"title": recipe.get("title") or rid})

            # Lexical bag-of-words: recipe title + tags + ingredient word parts
            # ("green_chili" -> {"green", "chili"}), for descriptor queries the
            # vectors can't express (e.g. "quick vegetarian").
            words: set[str] = set(_WORD_RE.findall(str(recipe.get("title", "")).lower()))
            for tag in recipe.get("tags", []):
                words.update(_WORD_RE.findall(str(tag).lower()))
            for token in ingredients:
                words.update(_WORD_RE.findall(token))
            self._doc_words[rid] = words

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def embedded(self) -> bool:
        """True if any recipe carries a semantic vector (embeddings were loaded)."""
        return len(self.index) > 0

    # -- Query helpers -------------------------------------------------------

    def _query_tokens(self, text: str) -> list[str]:
        """Extract known ingredient tokens from free text (unigrams + bigrams).

        Bigrams let two-word ingredients normalize correctly ("green chili",
        "olive oil"); only tokens present in the recipe vocabulary are kept, so a
        query vector is built from meaningful ingredients rather than stop-words.
        """
        words = _WORD_RE.findall(str(text or "").lower())
        candidates = list(words) + [f"{a} {b}" for a, b in zip(words, words[1:])]
        tokens: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            token = self._normalize(candidate)
            if token and token in self._vocab and token not in seen:
                seen.add(token)
                tokens.append(token)
        return tokens

    def _card(
        self, rid: str, *, score: float, similarity: float, matched: list[str], terms: list[str]
    ) -> dict[str, Any]:
        """Assemble one JSON-serializable result card for a recipe id."""
        recipe = self._by_id[rid]
        return {
            "id": rid,
            "title": recipe.get("title") or rid,
            "tags": list(recipe.get("tags", [])),
            "time_min": recipe.get("time_min"),
            "ingredients": [self._prettify(tok) for tok in self._ingredients[rid]],
            "score": round(score, 4),
            "similarity": round(similarity, 3),
            "matched": [self._prettify(tok) for tok in matched],
            "matched_terms": terms,
        }

    # -- Public API ----------------------------------------------------------

    def search(self, text: str, k: int = 6) -> list[dict[str, Any]]:
        """Rank the corpus against a free-text query, best match first.

        Combines semantic similarity (query-ingredient vector vs each recipe
        vector) with lexical overlap on title/tags/ingredient words, so both
        ingredient queries ("paneer tomato") and descriptor queries ("quick
        vegetarian dinner") return sensible results. Returns at most ``k`` cards.
        """
        query = str(text or "").strip()
        if not query:
            return []

        tokens = self._query_tokens(query)
        query_words = set(_WORD_RE.findall(query.lower()))
        query_vec = self._embed(tokens) if tokens else None

        vec_score: dict[str, float] = {}
        if query_vec:
            for rid, cosine in self.index.query(query_vec, k=len(self.index)):
                vec_score[rid] = (cosine + 1.0) / 2.0  # map cosine (-1..1) -> (0..1)

        results: list[dict[str, Any]] = []
        for rid in self._by_id:
            vec = vec_score.get(rid, 0.0)
            overlap = query_words & self._doc_words[rid]
            lex = len(overlap) / len(query_words) if query_words else 0.0
            combined = (_W_VECTOR * vec + _W_LEXICAL * lex) if query_vec else lex
            if combined <= 0.0:
                continue
            matched = [tok for tok in tokens if tok in self._ingredients[rid]]
            terms = sorted(overlap - set(_WORD_RE.findall(" ".join(tokens))))
            results.append(
                self._card(rid, score=combined, similarity=vec, matched=matched, terms=terms)
            )

        results.sort(key=lambda card: (-card["score"], card["title"]))
        return results[: max(0, int(k))]

    def similar(self, recipe_id: str, k: int = 4) -> list[dict[str, Any]]:
        """Return the ``k`` recipes most semantically similar to ``recipe_id``.

        Empty if the recipe is unknown or has no vector (no embeddings loaded).
        Each card's ``matched`` lists the shared ingredients that explain the match.
        """
        rid = str(recipe_id or "").strip()
        base_vec = self.index.vector_for(rid)
        if base_vec is None:
            return []
        base_ingredients = set(self._ingredients.get(rid, []))
        results: list[dict[str, Any]] = []
        for other, cosine in self.index.query(base_vec, k=k, exclude={rid}):
            shared = sorted(base_ingredients & set(self._ingredients.get(other, [])))
            results.append(
                self._card(
                    other,
                    score=(cosine + 1.0) / 2.0,
                    similarity=(cosine + 1.0) / 2.0,
                    matched=shared,
                    terms=[],
                )
            )
        return results
