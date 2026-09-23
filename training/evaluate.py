"""Score and compare trained ingredient-embedding files on ONE shared benchmark.

This is what turns "which dataset is better?" from an opinion into a number. Every
candidate embeddings file (whatever corpus produced it) is judged on the same three axes:

1. **Recipe completion (intrinsic, unsupervised).** Hold out one ingredient from each
   test recipe, average the vectors of the rest, and ask the model for its nearest
   ingredients. ``Recall@k`` = how often the held-out ingredient is in the top-k;
   ``MRR`` = mean reciprocal rank. This measures the exact skill the app relies on:
   "given what's in the fridge, which ingredient belongs with it."
2. **Cuisine accuracy (downstream, supervised).** Represent each What's Cooking recipe as
   the mean of its ingredient vectors, fit a logistic-regression cuisine classifier on a
   fixed train split, and report **accuracy** on the held-out test split — directly
   comparable to the published What's Cooking baselines (~78-81%). Same split for every
   model, so the comparison is fair.
3. **App coverage.** Of the ingredients the deployed app actually uses
   (``data/recipes.json``), how many does this vocabulary contain? A model can win on the
   benchmarks yet be useless to the app if it never learned "paneer"; this keeps us honest.

What's Cooking's ``train.json`` supplies BOTH the completion recipes and the cuisine
labels, so the whole comparison needs just that one labeled file plus the candidate
embeddings. numpy/scikit-learn are build-time deps only (``requirements-train.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from training import corpora

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_RECIPES = os.path.join(_HERE, os.pardir, "data", "recipes.json")


def load_embeddings(path: str) -> tuple[list[str], Any]:
    """Load an app-format embeddings JSON as ``(tokens, matrix)`` with unit-norm rows."""
    import numpy as np

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    vectors = data.get("vectors", {})
    tokens = list(vectors)
    matrix = np.asarray([vectors[t] for t in tokens], dtype=np.float32)
    # Re-normalize defensively; the files are already unit-norm but rounding drifts.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tokens, matrix / norms


def recipe_completion(
    recipes: list[list[str]],
    tokens: list[str],
    matrix: Any,
    *,
    k: int = 10,
    candidates: set[str] | None = None,
) -> dict[str, float]:
    """Recall@k and MRR for predicting one held-out ingredient per recipe.

    The held-out ingredient is ranked against a **candidate pool of real ingredients**
    (``candidates`` — e.g. the benchmark's own vocabulary), NOT the model's entire
    vocabulary. Ranking against the full training vocabulary is misleading: noisy NER
    fragments ("level tsp baking powder", "packagepre") crowd the top-k, so no held-out
    surface form ever wins and recall collapses to ~0 for every model regardless of
    quality. Restricting to a clean pool measures the app's real task — "given what's in
    the fridge, predict the missing ingredient" — and, when the SAME pool is used for
    every model, keeps the comparison fair (identical distractors).
    """
    import numpy as np

    index = {tok: i for i, tok in enumerate(tokens)}
    cand_tokens = list(tokens) if candidates is None else [t for t in candidates if t in index]
    if len(cand_tokens) < k + 1:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n_scored": 0, "k": float(k), "n_candidates": len(cand_tokens)}
    cand_global = np.asarray([index[t] for t in cand_tokens])
    cand_matrix = matrix[cand_global]  # (n_candidates, dim), unit-norm rows
    pos_of_global = {int(g): j for j, g in enumerate(cand_global)}  # model idx -> candidate row

    hits = 0
    scored = 0
    reciprocal_rank_sum = 0.0
    for recipe in recipes:
        known = [t for t in recipe if t in index]
        if len(known) < 3:  # need context + a target worth predicting
            continue
        target = known[-1]
        target_pos = pos_of_global.get(index[target])
        if target_pos is None:  # target must be a real candidate to be findable at all
            continue
        context = known[:-1]
        query = matrix[[index[t] for t in context]].mean(axis=0)
        norm = np.linalg.norm(query)
        if norm == 0:
            continue
        query = query / norm
        sims = cand_matrix @ query
        for t in context:  # never let the context predict itself
            cp = pos_of_global.get(index[t])
            if cp is not None:
                sims[cp] = -1.0
        # Rank = how many candidates score strictly higher than the target (+1); ties don't inflate it.
        rank = int((sims > sims[target_pos]).sum()) + 1
        scored += 1
        if rank <= k:
            hits += 1
        reciprocal_rank_sum += 1.0 / rank
    if scored == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n_scored": 0, "k": float(k), "n_candidates": len(cand_tokens)}
    return {
        "recall_at_k": hits / scored,
        "mrr": reciprocal_rank_sum / scored,
        "n_scored": scored,
        "k": float(k),
        "n_candidates": len(cand_tokens),
    }


def cuisine_accuracy(
    recipes: list[list[str]], labels: list[str], tokens: list[str], matrix: Any
) -> dict[str, float]:
    """Downstream cuisine-classification accuracy on a fixed What's Cooking split."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    index = {tok: i for i, tok in enumerate(tokens)}
    features = []
    kept_labels = []
    for recipe, label in zip(recipes, labels):
        rows = [index[t] for t in recipe if t in index]
        if not rows:
            continue
        features.append(matrix[rows].mean(axis=0))
        kept_labels.append(label)
    if len(set(kept_labels)) < 2:
        return {"accuracy": 0.0, "n_train": 0, "n_test": 0, "n_classes": len(set(kept_labels))}
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(kept_labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    return {
        "accuracy": float(clf.score(x_test, y_test)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_classes": int(len(set(kept_labels))),
    }


def app_coverage(tokens: list[str], app_recipes_path: str = _APP_RECIPES) -> dict[str, Any]:
    """How many of the deployed app's ingredients this vocabulary actually contains."""
    try:
        with open(app_recipes_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {"covered": 0, "total": 0, "missing": []}
    wanted: set[str] = set()
    for recipe in data.get("recipes", []):
        for ing in recipe.get("ingredients", []):
            norm = corpora.normalize_token(str(ing))
            if norm:
                wanted.add(norm)
    have = set(tokens)
    covered = sorted(wanted & have)
    missing = sorted(wanted - have)
    return {
        "covered": len(covered),
        "total": len(wanted),
        "coverage": (len(covered) / len(wanted)) if wanted else 0.0,
        "missing": missing[:20],
    }


SEED = 1234


def _vocab_of(path: str) -> set[str]:
    """Cheaply read just the token set of an embeddings file (no matrix build)."""
    with open(path, encoding="utf-8") as handle:
        return set(json.load(handle).get("vectors", {}))


def evaluate_file(
    path: str, recipes: list[list[str]], labels: list[str], candidates: set[str] | None
) -> dict[str, Any]:
    tokens, matrix = load_embeddings(path)
    return {
        "file": os.path.basename(path),
        "vocab": len(tokens),
        "completion": recipe_completion(recipes, tokens, matrix, candidates=candidates),
        "cuisine": cuisine_accuracy(recipes, labels, tokens, matrix),
        "app": app_coverage(tokens),
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    header = f"{'model':<28}{'vocab':>8}{'recall@10':>11}{'mrr':>8}{'cuisine acc':>13}{'app cov':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['file']:<28}{r['vocab']:>8}"
            f"{r['completion']['recall_at_k']:>11.3f}{r['completion']['mrr']:>8.3f}"
            f"{r['cuisine']['accuracy']:>12.1%}"
            f"{r['app']['coverage']:>10.1%}"
        )
    print("\nBaseline for cuisine accuracy (published What's Cooking): ~0.78-0.81")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ingredient-embedding files.")
    parser.add_argument("--embeddings", nargs="+", required=True, help="One or more app-format JSON files")
    parser.add_argument("--whats-cooking", required=True, help="Path to What's Cooking train.json")
    parser.add_argument("--json-out", help="Optional path to also dump full metrics as JSON")
    args = parser.parse_args()

    recipes, labels = corpora.load_whats_cooking(args.whats_cooking)
    print(f"Loaded {len(recipes)} labeled recipes across {len(set(labels))} cuisines.")

    # Shared candidate pool for recipe completion: benchmark ingredients that EVERY model
    # knows. Same clean distractors for all models → recall@10/MRR are directly comparable
    # and free of training-vocab noise. (cuisine acc + coverage still use each full vocab.)
    eval_vocab: set[str] = {tok for recipe in recipes for tok in recipe}
    candidates = eval_vocab.intersection(*(_vocab_of(path) for path in args.embeddings))
    print(
        f"Recipe-completion candidate pool: {len(candidates)} ingredients "
        f"(benchmark vocab shared by all {len(args.embeddings)} models).\n"
    )

    results = [evaluate_file(path, recipes, labels, candidates) for path in args.embeddings]
    _print_table(results)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nFull metrics written to {args.json_out}")


if __name__ == "__main__":
    main()
