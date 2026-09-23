"""Corpus adapters for the ingredient-embedding bake-off.

Each real dataset stores ingredients differently, so this module hides those differences
behind ONE shape: a stream of recipes, where each recipe is a ``list[str]`` of normalized
ingredient tokens. The trainer (:mod:`train_word2vec`) and the evaluator
(:mod:`evaluate`) only ever see that shape, so adding a new dataset later means writing
one adapter here and nothing else.

Datasets (you download these yourself — see ``training/README.md``):

* **RecipeNLG** — ``full_dataset.csv``. We read the ``NER`` column, which is the paper's
  already-extracted clean ingredient names (e.g. ``["butter", "eggs", "flour"]``) rather
  than the noisy free-text ``ingredients`` lines ("1 c. shortening, melted").
* **Food.com** — ``RAW_recipes.csv``. We read the ``ingredients`` column, a Python-list
  literal string of cleaned ingredient names.
* **What's Cooking** — ``train.json``. A list of ``{id, cuisine, ingredients: [...]}``;
  used as the LABELED yardstick (cuisine classification) both models are scored on.

All adapters are lazy/streaming where the file is large, so RecipeNLG's ~2 GB never has
to sit in memory at once.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator

# --- Normalization ----------------------------------------------------------

_WS = re.compile(r"\s+")
# Leading measurement/qualifier noise that sometimes clings to an ingredient name.
_LEADING_JUNK = re.compile(r"^(fresh|dried|ground|chopped|minced|large|small|whole)\s+")


def normalize_token(raw: str) -> str:
    """Lower-case, trim, and collapse internal whitespace of one ingredient name.

    Kept deliberately light: the NER/ingredients columns we read are already cleaned
    noun phrases, so aggressive stemming would only fragment the vocabulary. We do strip
    a few leading qualifiers so ``fresh basil`` and ``basil`` collapse together.
    """
    token = _WS.sub(" ", str(raw).strip().lower())
    token = _LEADING_JUNK.sub("", token)
    return token.strip()


def _normalize_row(tokens: list[str]) -> list[str]:
    """Normalize a recipe's tokens and drop blanks/dupes while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        norm = normalize_token(tok)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _parse_list_cell(cell: str) -> list[str]:
    """Parse a spreadsheet cell that holds a list of strings.

    RecipeNLG's ``NER`` is JSON (double quotes); Food.com's ``ingredients`` is a Python
    repr (single quotes). Try JSON first, then ``ast.literal_eval``; give up to an empty
    list rather than raising, so one malformed row can't abort a multi-hour training run.
    """
    if not isinstance(cell, str) or not cell.strip():
        return []
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(cell)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
    return []


# --- Streaming CSV adapters -------------------------------------------------


def _iter_csv_column(path: str, column: str, *, limit: int | None) -> Iterator[list[str]]:
    """Yield the parsed+normalized list from ``column`` for each row of a large CSV.

    Uses pandas' chunked reader so a multi-GB file is never fully materialized. pandas is
    a build-time dependency only (see ``requirements-train.txt``); it is imported lazily
    here so importing this module for the synthetic smoke test needs nothing installed.
    """
    import pandas as pd  # lazy: only needed when actually reading a real dataset

    yielded = 0
    for chunk in pd.read_csv(path, usecols=[column], chunksize=10_000, dtype=str):
        for cell in chunk[column].tolist():
            row = _normalize_row(_parse_list_cell(cell))
            if len(row) < 2:  # a co-occurrence model needs at least a pair
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def iter_recipenlg(path: str, *, limit: int | None = None) -> Iterator[list[str]]:
    """Stream RecipeNLG recipes as ingredient-token lists (from the ``NER`` column)."""
    return _iter_csv_column(path, "NER", limit=limit)


def iter_foodcom(path: str, *, limit: int | None = None) -> Iterator[list[str]]:
    """Stream Food.com recipes as ingredient-token lists (from the ``ingredients`` column)."""
    return _iter_csv_column(path, "ingredients", limit=limit)


# --- Labeled dataset (the shared yardstick) ---------------------------------


def load_whats_cooking(path: str) -> tuple[list[list[str]], list[str]]:
    """Load What's Cooking as ``(recipes_of_tokens, cuisine_labels)`` — the labeled eval.

    Small enough (~40K recipes) to hold in memory. Returned recipes are normalized the
    same way as the training corpora so the vocabularies line up at scoring time.
    """
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    recipes: list[list[str]] = []
    labels: list[str] = []
    for row in rows:
        tokens = _normalize_row([str(t) for t in row.get("ingredients", [])])
        cuisine = row.get("cuisine")
        if tokens and cuisine:
            recipes.append(tokens)
            labels.append(str(cuisine))
    return recipes, labels


# --- Synthetic corpus (offline smoke test) ----------------------------------


def synthetic_corpus() -> list[list[str]]:
    """A tiny deterministic corpus so the whole pipeline can be proven with no download.

    Three latent 'cuisines' whose ingredients co-occur, so a correctly-trained model must
    place within-group ingredients closer than across-group ones — enough to smoke-test
    training, export, and evaluation end to end.
    """
    indian = ["paneer", "garam masala", "cumin", "turmeric", "ginger", "onion", "tomato"]
    italian = ["pasta", "parmesan", "basil", "olive oil", "garlic", "tomato", "oregano"]
    dessert = ["flour", "sugar", "butter", "eggs", "vanilla", "baking powder", "milk"]
    corpus: list[list[str]] = []
    # Repeat with rotating subsets so pairs recur (co-occurrence needs repetition).
    for base in (indian, italian, dessert):
        for start in range(len(base)):
            corpus.append([base[(start + i) % len(base)] for i in range(5)])
    return corpus


# --- Dispatch ---------------------------------------------------------------

_STREAMERS: dict[str, Callable[..., Iterator[list[str]]]] = {
    "recipenlg": iter_recipenlg,
    "foodcom": iter_foodcom,
}


def stream_corpus(name: str, path: str, *, limit: int | None = None) -> Iterator[list[str]]:
    """Return the ingredient-token stream for a named training corpus."""
    try:
        streamer = _STREAMERS[name]
    except KeyError:
        raise ValueError(f"Unknown corpus '{name}'. Options: {sorted(_STREAMERS)}") from None
    return streamer(path, limit=limit)
