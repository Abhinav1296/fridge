# Ingredient-embedding bake-off

Train the recommender's ingredient embeddings on a **real** recipe corpus instead of the
49-ingredient seed set, then pick the winner with numbers instead of vibes — and ship a
production file that drops straight into the app.

Two candidate corpora are trained **separately** and scored on **one shared benchmark**.
The winning *corpus* is then retrained together with the app's own recipes into the
production embeddings the app loads. Nothing here ships in the deployed app — gensim,
pandas and scikit-learn are build-time only (`requirements-train.txt`); the app reads the
resulting JSON with pure Python.

## 1. Install the build-time tools

```bash
pip install -r requirements-train.txt
```

## 2. Get the datasets (all free, no Kaggle account needed)

Download each file and drop it under `data/raw/` as shown. `data/raw/` is git-ignored, so
nothing large is committed. The adapters (`training/corpora.py`) read the columns noted
below, so any mirror with the same schema works.

| Corpus | Free source | Column used | Put it at |
|---|---|---|---|
| **RecipeNLG** (2.2M recipes) | Official site <https://recipenlg.cs.put.poznan.pl/> (click through the license) — also mirrored on Hugging Face Datasets | `NER` (pre-cleaned ingredient names) | `data/raw/recipenlg_full.csv` |
| **Food.com** (231K recipes) | Hugging Face Datasets mirror of the Food.com `RAW_recipes` dump | `ingredients` | `data/raw/RAW_recipes.csv` |
| **What's Cooking** (labels, for scoring) | Hugging Face / GitHub mirror of the Kaggle "What's Cooking" `train.json` | `cuisine` + `ingredients` | `data/raw/whats_cooking_train.json` |

> Verify a mirror before a multi-GB download: the loaders only need the one column above,
> so a quick header read (`pandas.read_csv(path, nrows=5)`) confirms the schema. One mirror
> we tried had an `ingredients` column that was non-null for < 0.1 % of rows — check first.

## 3. Train both candidates

```bash
python -m training.train_word2vec --corpus recipenlg \
    --input data/raw/recipenlg_full.csv --output data/embeddings_recipenlg.json
python -m training.train_word2vec --corpus foodcom \
    --input data/raw/RAW_recipes.csv --output data/embeddings_foodcom.json
```

Add `--limit 50000` for a quick trial before committing to the full corpus. gensim uses
skip-gram with **negative sampling** (not the full softmax of the teaching-oriented
`build_embeddings.py`), so millions of recipes train in minutes.

## 4. Score them on the shared benchmark

```bash
python -m training.evaluate \
    --embeddings data/embeddings_recipenlg.json data/embeddings_foodcom.json \
    --whats-cooking data/raw/whats_cooking_train.json \
    --json-out data/eval_results.json
```

One row per model:

- **recall@10 / MRR** — recipe completion: hold out one ingredient per benchmark recipe and
  rank it against a **clean, shared candidate pool** (the benchmark's own ingredients that
  *every* model knows), so noisy training-vocab fragments can't distort the score and the
  models face identical distractors.
- **cuisine acc** — downstream logistic-regression cuisine classification on a fixed split,
  comparable to the published What's Cooking baseline of ~78–81 %.
- **app cov** — fraction of the app's own ingredients this vocabulary covers.

**Result (full corpora):** RecipeNLG wins every axis — recall@10 0.018 vs 0.004, cuisine
accuracy 69.5 % vs 67.4 %, app coverage 93.9 % vs 85.7 %. RecipeNLG's `NER` column is
cleaner and its 2.2M recipes give far broader ingredient coverage.

## 5. Build the production file (ship the winner)

Do **not** just copy a candidate over `data/ingredient_embeddings.json` — the app looks up
*underscore-joined* keys (`green_chili`, not `green chili`), and a couple of app ingredients
(`poha`, `rajma`) don't exist in RecipeNLG's Western-leaning corpus. `build_production`
fixes both: it retrains the winning corpus **unioned with the app's own recipes** (so every
app ingredient gets a real vector in the same space), rewrites keys to the app's form, and
exports a bounded vocabulary (all app ingredients + the top-N most frequent RecipeNLG
ingredients) so the committed file stays a few MB.

```bash
python -m training.build_production --recipenlg data/raw/recipenlg_full.csv --top 2500
```

This writes `data/ingredient_embeddings.json` directly and prints app-coverage (expect
49/49) plus nearest-neighbour spot checks. Restart the app — semantic search and the
recommender now run on the upgraded model; if the vectors were ever missing, both degrade
gracefully to lexical ranking rather than breaking.

## Smoke test (no download needed)

Proves the whole pipeline end to end on a tiny built-in synthetic corpus:

```bash
python -m training.train_word2vec --corpus synthetic --output /tmp/smoke.json
```
