"""Train ingredient embeddings at scale with gensim Word2Vec (skip-gram + negatives).

Why this exists alongside ``build_embeddings.py``: that script implements word2vec from
scratch with a **full softmax**, which is O(vocabulary) per training pair — perfect for
teaching and fine for 49 ingredients, but hopelessly slow once a corpus has thousands of
ingredients across hundreds of thousands of recipes. gensim uses **negative sampling**
(a handful of contrastive updates per pair instead of a full-vocabulary pass), so the
same skip-gram idea trains on millions of recipes in minutes.

The output is byte-for-byte the shape the app already reads
(``{"dim": int, "vectors": {token: [floats]}}``, vectors L2-normalized), so a freshly
trained file drops straight into ``data/ingredient_embeddings.json`` with no app change.

gensim is a BUILD-TIME dependency only (``requirements-train.txt``); the deployed app
never imports it. Run examples::

    python -m training.train_word2vec --corpus recipenlg --input data/raw/full_dataset.csv \
        --output data/embeddings_recipenlg.json
    python -m training.train_word2vec --synthetic --output /tmp/smoke.json   # no download
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from training import corpora

# Defaults tuned for a real (100K+) corpus; overridable on the CLI.
DEFAULT_DIM = 100
DEFAULT_EPOCHS = 8
DEFAULT_WINDOW = 10        # large enough to span a typical recipe's ingredient set
DEFAULT_MIN_COUNT = 5      # drop ingredients seen in <5 recipes (noise / typos)
DEFAULT_NEGATIVE = 10
SEED = 1234


class _ReIterable:
    """Wrap a 'make a fresh iterator' thunk so gensim can stream the corpus repeatedly.

    gensim walks the corpus once to build the vocabulary and again on every epoch, so a
    one-shot generator would be exhausted after the first pass. This re-opens the source
    each time it is iterated.
    """

    def __init__(self, make_iterator: Callable[[], Iterator[list[str]]]) -> None:
        self._make_iterator = make_iterator

    def __iter__(self) -> Iterator[list[str]]:
        return self._make_iterator()


def _corpus_iterable(name: str, path: str, *, limit: int | None) -> Iterable[list[str]]:
    if name == "synthetic":
        rows = corpora.synthetic_corpus()
        return _ReIterable(lambda: iter(rows))
    return _ReIterable(lambda: corpora.stream_corpus(name, path, limit=limit))


def train(
    sentences: Iterable[list[str]],
    *,
    dim: int = DEFAULT_DIM,
    epochs: int = DEFAULT_EPOCHS,
    window: int = DEFAULT_WINDOW,
    min_count: int = DEFAULT_MIN_COUNT,
    negative: int = DEFAULT_NEGATIVE,
) -> Any:
    """Train and return a gensim ``Word2Vec`` model (skip-gram, negative sampling)."""
    from gensim.models import Word2Vec  # lazy import — build-time dependency only

    return Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,                 # skip-gram (predict context ingredients from a center one)
        negative=negative,    # negative sampling instead of full softmax
        epochs=epochs,
        seed=SEED,
        workers=os.cpu_count() or 4,
    )


def export(model: Any, *, source: str, method_extra: str = "") -> dict[str, Any]:
    """Serialize a trained model into the app's embeddings JSON shape (L2-normalized)."""
    wv = model.wv
    dim = int(wv.vector_size)
    vectors: dict[str, list[float]] = {}
    for token in wv.index_to_key:
        unit = wv.get_vector(token, norm=True)  # gensim returns the L2-normalized vector
        vectors[token] = [round(float(x), 6) for x in unit]
    method = "skip-gram (gensim word2vec, negative sampling)"
    if method_extra:
        method = f"{method} — {method_extra}"
    return {
        "method": method,
        "dim": dim,
        "epochs": int(model.epochs),
        "window": int(model.window),
        "min_count": int(model.min_count),
        "negative": int(model.negative),
        "seed": SEED,
        "vocab_size": len(vectors),
        "source": source,
        "vectors": vectors,
    }


def write_json(payload: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=0)


def _sanity_report(payload: dict[str, Any]) -> None:
    """Print a couple of nearest-neighbour spot checks so a run is obviously not garbage."""
    vectors = payload["vectors"]
    dim = payload["dim"]

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))  # both already unit-normalized

    for probe in ("onion", "sugar", "paneer", "basil"):
        if probe not in vectors:
            continue
        base = vectors[probe]
        ranked = sorted(
            ((cosine(base, v), t) for t, v in vectors.items() if t != probe),
            reverse=True,
        )[:5]
        neighbours = ", ".join(f"{t} {s:.2f}" for s, t in ranked)
        print(f"  {probe:<14} -> {neighbours}")
    print(f"  (vocab {payload['vocab_size']} tokens, {dim}-dim)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ingredient embeddings at scale.")
    parser.add_argument("--corpus", choices=["recipenlg", "foodcom", "synthetic"], required=True)
    parser.add_argument("--input", help="Path to the raw dataset file (not needed for --corpus synthetic)")
    parser.add_argument("--output", required=True, help="Where to write the embeddings JSON")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of recipes (quick trials)")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--negative", type=int, default=DEFAULT_NEGATIVE)
    args = parser.parse_args()

    if args.corpus != "synthetic" and not args.input:
        parser.error("--input is required unless --corpus synthetic")

    source = "synthetic corpus" if args.corpus == "synthetic" else f"{args.corpus}:{args.input}"
    print(f"Training on {source} (dim={args.dim}, epochs={args.epochs}, min_count={args.min_count})...")

    sentences = _corpus_iterable(args.corpus, args.input or "", limit=args.limit)
    model = train(
        sentences,
        dim=args.dim,
        epochs=args.epochs,
        window=args.window,
        min_count=args.min_count,
        negative=args.negative,
    )
    extra = f"trained on {args.corpus}" + (f", first {args.limit} recipes" if args.limit else "")
    payload = export(model, source=source, method_extra=extra)
    write_json(payload, args.output)
    print(f"Wrote {payload['vocab_size']} vectors to {args.output}. Spot checks:")
    _sanity_report(payload)


if __name__ == "__main__":
    main()
