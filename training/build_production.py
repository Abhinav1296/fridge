"""Build the *shippable* ingredient-embedding file the app loads at runtime.

The bake-off (:mod:`train_word2vec` + :mod:`evaluate`) answered "which free dataset trains
the better ingredient space?" — RecipeNLG won on every axis (recipe-completion recall,
downstream cuisine accuracy, and coverage of the app's own ingredients). This script turns
that finding into the production artifact ``data/ingredient_embeddings.json``.

Two things the raw bake-off winner can't do, which this script fixes:

1. **App-key alignment.** The app looks up *underscore-joined* tokens (``normalize_ingredient``
   in :mod:`recommender` turns "green chili" into ``green_chili``), but RecipeNLG stores the
   phrase with a space. Each ingredient phrase is already a single word2vec token, so
   rewriting ``"green chili" -> "green_chili"`` is a pure key rename with zero effect on the
   learned geometry — it just makes the vectors findable by the app.
2. **App-ingredient coverage.** A few app ingredients (``poha``, ``rajma``) are genuinely
   absent from RecipeNLG's Western-leaning corpus. Training RecipeNLG **unioned with the
   app's own recipes** (repeated so those tokens clear ``min_count``) gives every app
   ingredient a real vector in the *same* 100-dim space as the millions of RecipeNLG
   ingredients — breadth plus guaranteed coverage. Independently trained spaces can't be
   compared with cosine, so the union has to be a single joint training run.

The file the app reads is kept small (it is committed to the repo) by exporting only a
bounded vocabulary: every app ingredient plus the top-N most frequent RecipeNLG ingredients,
so common fridge items a user types ("spinach", "olive oil") still resolve, without
committing a ~30 MB full-vocabulary dump.

gensim/pandas are build-time deps only (``requirements-train.txt``); the deployed app reads
the resulting JSON with pure Python. Example::

    python -m training.build_production --recipenlg data/raw/recipenlg_full.csv
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from typing import Any

from training import corpora, train_word2vec

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_APP_RECIPES = os.path.join(_HERE, os.pardir, "data", "recipes.json")
_DEFAULT_OUTPUT = os.path.join(_HERE, os.pardir, "data", "ingredient_embeddings.json")

DEFAULT_TOP = 5000       # most-frequent RecipeNLG ingredients to keep (plus every app token)
DEFAULT_APP_REPEAT = 10  # times to replay the app corpus so app-only tokens clear min_count


def _to_app_key(token: str) -> str:
    """Rewrite a normalized ingredient phrase into the app's canonical key form.

    ``normalize_ingredient`` in the app collapses whitespace to underscores, so a vector
    keyed "green chili" would never be found; keyed "green_chili" it drops straight in.
    """
    return corpora.normalize_token(token).replace(" ", "_")


def _app_tokens(app_recipes_path: str) -> list[str]:
    """The app's own ingredient vocabulary (canonical keys), e.g. ``["onion", "green_chili"]``."""
    with open(app_recipes_path, encoding="utf-8") as handle:
        data = json.load(handle)
    tokens: list[str] = []
    seen: set[str] = set()
    for recipe in data.get("recipes", []):
        for ing in recipe.get("ingredients", []):
            key = _to_app_key(str(ing))
            if key and key not in seen:
                seen.add(key)
                tokens.append(key)
    return tokens


def _union_stream(
    recipenlg_path: str, app_recipes_path: str, *, app_repeat: int, limit: int | None
) -> Iterator[list[str]]:
    """Yield every training recipe once: all of RecipeNLG, then the app corpus ``app_repeat``×.

    RecipeNLG tokens are rewritten to the app's underscore key form so identical ingredients
    from the two sources share a single vocabulary entry (and thus one learned vector).
    """
    for recipe in corpora.iter_recipenlg(recipenlg_path, limit=limit):
        yield [_to_app_key(tok) for tok in recipe]
    with open(app_recipes_path, encoding="utf-8") as handle:
        app_recipes = [
            [_to_app_key(str(ing)) for ing in r.get("ingredients", [])]
            for r in json.load(handle).get("recipes", [])
        ]
    app_recipes = [[t for t in r if t] for r in app_recipes if len([t for t in r if t]) >= 2]
    for _ in range(app_repeat):
        yield from app_recipes


def build(
    recipenlg_path: str,
    app_recipes_path: str,
    *,
    top: int = DEFAULT_TOP,
    app_repeat: int = DEFAULT_APP_REPEAT,
    limit: int | None = None,
    dim: int = train_word2vec.DEFAULT_DIM,
    epochs: int = train_word2vec.DEFAULT_EPOCHS,
    window: int = train_word2vec.DEFAULT_WINDOW,
    min_count: int = train_word2vec.DEFAULT_MIN_COUNT,
    negative: int = train_word2vec.DEFAULT_NEGATIVE,
) -> dict[str, Any]:
    """Train the union model and return the bounded, app-keyed embeddings payload."""
    app_tokens = _app_tokens(app_recipes_path)
    sentences = train_word2vec._ReIterable(
        lambda: _union_stream(recipenlg_path, app_recipes_path, app_repeat=app_repeat, limit=limit)
    )
    model = train_word2vec.train(
        sentences, dim=dim, epochs=epochs, window=window, min_count=min_count, negative=negative
    )
    wv = model.wv

    # Bounded vocab: every app ingredient (guaranteed) + the most frequent RecipeNLG tokens.
    keep: list[str] = [t for t in app_tokens if t in wv.key_to_index]
    keep_set = set(keep)
    for token in wv.index_to_key:  # gensim orders index_to_key by descending frequency
        if len(keep) >= top + len(app_tokens):
            break
        if token not in keep_set:
            keep.append(token)
            keep_set.add(token)

    vectors = {token: [round(float(x), 6) for x in wv.get_vector(token, norm=True)] for token in keep}
    missing_app = [t for t in app_tokens if t not in vectors]
    return {
        "method": "skip-gram (gensim word2vec, negative sampling) - RecipeNLG + app recipes (union)",
        "dim": int(wv.vector_size),
        "vocab_size": len(vectors),
        "source": f"recipenlg:{recipenlg_path} + app:{app_recipes_path} (x{app_repeat})",
        "app_ingredients_covered": f"{len(app_tokens) - len(missing_app)}/{len(app_tokens)}",
        "app_ingredients_missing": missing_app,
        "vectors": vectors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shippable app embeddings file.")
    parser.add_argument("--recipenlg", required=True, help="Path to RecipeNLG full_dataset.csv")
    parser.add_argument("--app-recipes", default=_DEFAULT_APP_RECIPES)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="Frequent RecipeNLG tokens to keep")
    parser.add_argument("--app-repeat", type=int, default=DEFAULT_APP_REPEAT)
    parser.add_argument("--limit", type=int, default=None, help="Cap RecipeNLG recipes (quick trials)")
    parser.add_argument("--dim", type=int, default=train_word2vec.DEFAULT_DIM)
    parser.add_argument("--epochs", type=int, default=train_word2vec.DEFAULT_EPOCHS)
    parser.add_argument("--window", type=int, default=train_word2vec.DEFAULT_WINDOW)
    parser.add_argument("--min-count", type=int, default=train_word2vec.DEFAULT_MIN_COUNT)
    parser.add_argument("--negative", type=int, default=train_word2vec.DEFAULT_NEGATIVE)
    args = parser.parse_args()

    print(
        f"Building production embeddings from {args.recipenlg} + {args.app_recipes} "
        f"union (app x{args.app_repeat})..."
    )
    payload = build(
        args.recipenlg,
        args.app_recipes,
        top=args.top,
        app_repeat=args.app_repeat,
        limit=args.limit,
        dim=args.dim,
        epochs=args.epochs,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
    )
    train_word2vec.write_json(payload, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Wrote {payload['vocab_size']} vectors ({payload['dim']}-dim) to {args.output} [{size_mb:.1f} MB].")
    print(
        f"App ingredients covered: {payload['app_ingredients_covered']}  "
        f"missing: {payload['app_ingredients_missing']}"
    )
    print("Spot checks:")
    train_word2vec._sanity_report(payload)


if __name__ == "__main__":
    main()
