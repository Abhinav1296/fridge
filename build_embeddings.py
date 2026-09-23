"""Train ingredient embeddings for the SmartFridge recommender (offline, one-time).

This is the small machine-learning model the project trains itself. It learns a dense
vector ("embedding") for every ingredient from the seed recipe corpus using the
**skip-gram** technique (the method behind word2vec): a recipe is treated as a bag of
co-occurring ingredients, and the model learns to predict which ingredients appear
together. Ingredients that keep showing up in the same recipes (e.g. ``onion`` and
``tomato``) end up close together in the vector space; unrelated ones end up far apart.

The recommender then measures how well a recipe matches what is in the fridge partly by
the cosine similarity of these vectors, so it can rank recipes semantically rather than
by exact string matching alone.

**This script is the only place that needs a heavy dependency (numpy).** It runs on a
laptop or a free Colab notebook, once, and writes a small JSON file of vectors
(``data/ingredient_embeddings.json``). The web app and the recommender read that JSON
with pure Python — no numpy, no model download — so the deployed app (Vercel) stays
lightweight. Re-run this whenever ``data/recipes.json`` changes:

    python build_embeddings.py

Training is fully deterministic (fixed random seed), so the same corpus always produces
the same vectors.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# --- Paths ------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
RECIPES_PATH = os.path.join(DATA_DIR, "recipes.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "ingredient_embeddings.json")

# --- Hyperparameters (small corpus → small, fast, deterministic model) -------

EMBED_DIM = 32          # length of each ingredient vector
EPOCHS = 600            # full-batch gradient-descent passes over the co-occurrence pairs
LEARNING_RATE = 0.25    # initial step size; linearly decayed toward LEARNING_RATE_MIN
LEARNING_RATE_MIN = 0.02
SEED = 1234             # fixes initialization so results are reproducible


def load_recipe_ingredients(path: str = RECIPES_PATH) -> list[list[str]]:
    """Return each recipe's ingredient list (as token lists) from the recipe corpus."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    recipes = data.get("recipes", [])
    return [
        [str(tok).strip().lower() for tok in recipe.get("ingredients", []) if str(tok).strip()]
        for recipe in recipes
    ]


def build_vocab(recipes: list[list[str]]) -> list[str]:
    """Build a sorted, de-duplicated vocabulary of every ingredient token seen."""
    vocab: set[str] = set()
    for ingredients in recipes:
        vocab.update(ingredients)
    return sorted(vocab)


def build_pairs(
    recipes: list[list[str]], token_to_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build skip-gram (center, context) index pairs.

    A recipe is a bag of co-occurring ingredients, so within each recipe every ordered
    pair of distinct ingredients becomes one training example: the model learns to
    predict ``context`` given ``center``.
    """
    centers: list[int] = []
    contexts: list[int] = []
    for ingredients in recipes:
        indices = [token_to_index[tok] for tok in dict.fromkeys(ingredients)]  # de-dup, keep order
        for i, center in enumerate(indices):
            for j, context in enumerate(indices):
                if i != j:
                    centers.append(center)
                    contexts.append(context)
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train(
    vocab: list[str],
    centers: np.ndarray,
    contexts: np.ndarray,
) -> np.ndarray:
    """Train a skip-gram model with full softmax and return the L2-normalized embeddings.

    Two weight matrices are learned: ``E`` (the input/ingredient embeddings we keep) and
    ``W`` (output weights, discarded after training). For each (center, context) pair the
    model scores every vocabulary word and is nudged, by gradient descent on the
    cross-entropy loss, to raise the score of the true context word.
    """
    rng = np.random.default_rng(SEED)
    vocab_size = len(vocab)

    # Small random init keeps early gradients well-behaved.
    embeddings = rng.normal(0.0, 0.1, size=(vocab_size, EMBED_DIM))
    output = rng.normal(0.0, 0.1, size=(vocab_size, EMBED_DIM))

    num_pairs = len(centers)
    if num_pairs == 0:
        raise ValueError("No co-occurrence pairs to train on — is recipes.json empty?")

    one_hot_rows = np.arange(num_pairs)
    for epoch in range(EPOCHS):
        lr = LEARNING_RATE - (LEARNING_RATE - LEARNING_RATE_MIN) * (epoch / max(1, EPOCHS - 1))

        # Forward: center embedding -> scores over the whole vocabulary -> softmax.
        hidden = embeddings[centers]                 # (P, d)
        logits = hidden @ output.T                   # (P, V)
        probs = _softmax(logits)                     # (P, V)

        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            loss = -np.log(probs[one_hot_rows, contexts] + 1e-9).mean()
            print(f"  epoch {epoch:4d}/{EPOCHS}  lr={lr:.3f}  loss={loss:.4f}")

        # Cross-entropy gradient on the logits (copy so the loss above stays readable).
        d_logits = probs.copy()
        d_logits[one_hot_rows, contexts] -= 1.0
        d_logits /= num_pairs

        # Backprop into the two weight matrices.
        d_output = d_logits.T @ hidden               # (V, d)
        d_hidden = d_logits @ output                 # (P, d)

        output -= lr * d_output
        # Scatter-add the per-pair hidden gradients back onto the center rows of E.
        grad_embeddings = np.zeros_like(embeddings)
        np.add.at(grad_embeddings, centers, d_hidden)
        embeddings -= lr * grad_embeddings

    # L2-normalize so the recommender can use a plain dot product as cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def main() -> None:
    print(f"Reading recipes from {RECIPES_PATH}")
    recipes = load_recipe_ingredients()
    vocab = build_vocab(recipes)
    token_to_index = {tok: i for i, tok in enumerate(vocab)}
    centers, contexts = build_pairs(recipes, token_to_index)
    print(
        f"Corpus: {len(recipes)} recipes, {len(vocab)} ingredients, "
        f"{len(centers)} co-occurrence pairs. Training {EMBED_DIM}-dim embeddings..."
    )

    vectors = train(vocab, centers, contexts)

    payload: dict[str, Any] = {
        "method": "skip-gram (word2vec) over recipe ingredient co-occurrence",
        "dim": EMBED_DIM,
        "epochs": EPOCHS,
        "seed": SEED,
        "vocab_size": len(vocab),
        "source": "data/recipes.json",
        # Rounded to keep the JSON small; 6 decimals is ample for cosine ranking.
        "vectors": {tok: [round(float(v), 6) for v in vectors[i]] for i, tok in enumerate(vocab)},
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=0)
    print(f"Wrote {len(vocab)} ingredient vectors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
