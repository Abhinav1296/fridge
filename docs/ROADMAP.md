# SmartFridge → **Zero-Waste Kitchen Agent** — Capstone Roadmap

A plan to turn the current SmartFridge Vision app into something **unique, sophisticated,
and defensible in front of faculty** — without breaking the clean software app you already
have. Working name for the upgraded product: **PantryPilot** (rename optional).

> **The one-line pitch:** *Most "smart fridge" projects just detect food. Ours sees your
> fridge, tracks what's about to spoil, and then acts like a chef — it plans a week of meals
> with an optimization engine that mathematically minimizes food waste and grocery cost while
> hitting your nutrition goals, and explains every decision.*

That sentence is the "unique" part. Nobody else's fridge app does operations-research meal
planning driven by an AI agent. Everything below builds toward it.

---

## Guiding principles (so "complicated" never means "broken")

1. **The current app always works.** Every phase is additive. Upload/webcam → inventory +
   freshness → history → suggestions keeps running at all times.
2. **Graceful degradation everywhere** (you already do this with vision fallbacks and the
   lexical-ranking fallback). Solver missing → heuristic planner. Nutrition API down → cached
   estimates. Vector DB absent → today's JSON embeddings. Nothing hard-fails.
3. **Core modules stay framework-agnostic** — `agent.py`, `optimizer.py`, `recommender.py`,
   `storage.py`, `vision_service.py` have **no Flask dependency**. This clean seam is exactly
   what impresses reviewers, and it's already your architecture.
4. **Secrets only in `.env`, never committed.** Any new key (nutrition API, LLM) follows the
   existing pattern; you add the value yourself.
5. **You only need Phase 1 to already be unique and impressive.** Phases 2–3 are optional
   depth you add if time allows. Don't let the size of this doc stress you — it's a menu.

---

## Architecture at a glance (target state)

```
                       ┌─────────────────────────────────────────┐
   Browser  ──────────▶│  app.py  (Flask web layer)              │
   (upload / webcam /  │  routes, auth, async job dispatch        │
    barcode / chat)    └───────────────┬─────────────────────────┘
                                        │  (all framework-agnostic below)
        ┌───────────────┬───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼               ▼              ▼
  vision_service   recommender     optimizer.py       agent.py      nutrition.py
  (unchanged)      (embeddings)    zero-waste ILP     LLM + tools   OpenFoodFacts/USDA
        │               │               │               │              │
        └───────────────┴───────────────┴───────┬───────┴──────────────┘
                                                 ▼
                                     storage.py  (libSQL/Turso → Postgres later)
                                     inventory · perishables · nutrition ·
                                     meal plans · users · households · waste log
```

---

## Phase 0 — Foundation (small, do first)

Sets up the ground so later phases don't fight the codebase.

- **Add a test suite** (`tests/`) with `pytest`: unit tests for `recommender`, `storage`,
  and (soon) `optimizer`; a couple of endpoint smoke tests for `app.py`. This is cheap and
  makes the "engineering rigor" story real from day one.
- **Nutrition data table** in `storage.py`: `nutrition(item, kcal, protein, carbs, fat,
  source, updated_at)` — a local cache so the optimizer has calorie/macro data to work with.
- **Config plumbing**: add `LLM_*` env vars (reuse the vision provider pattern) so the agent
  has its own model config independent of the vision model.

**New/changed:** `tests/`, `requirements-dev.txt` (pytest), `storage.py` (+1 table), `.env.example`.

---

## Phase 1 — The unique hook: **Zero-Waste Agent + Optimizer** ⭐ START HERE

This alone makes the project stand out. Two new core modules + a UI.

### 1A. `optimizer.py` — the mathematically serious "built by us" piece

The centerpiece. Frame meal planning as a **constrained optimization problem**:

- **Decision:** which recipes to cook across the next *N* days.
- **Objective:** minimize a weighted sum of
  - **predicted waste** — ingredients likely to expire unused, weighted by how soon they
    expire (uses your use-by dates + freshness), and
  - **shopping cost** — ingredients you'd have to buy that you don't already have.
- **Constraints:** rough daily **calorie/macro targets**, **no ingredient over-use** (can't
  use more than you have + can buy), and **variety** (don't repeat the same dish).
- **Output:** a day-by-day meal plan, the resulting shopping list, and metrics
  (**kg of waste avoided**, **₹ saved**) — great demo numbers.

Implementation:
- Primary: an **ILP solver** via **PuLP** (pip-installable, ships the CBC solver).
- **Fallback:** a pure-Python **greedy + local-search** planner so it always runs even where
  the solver binary isn't available (e.g. Vercel serverless) — same graceful-degradation
  pattern as your recommender.
- Framework-agnostic: `optimizer.plan(inventory, perishables, targets, days) -> MealPlan`.

*Why faculty like it:* it's genuine operations research (linear programming), it's clearly
"built by us," and it produces explainable, quantified results.

### 1B. `agent.py` — the AI agent (reasoning + tool use)

A tool-using LLM agent that turns natural-language requests into actions:

- **Tools it can call:** `get_inventory()`, `get_perishables()`, `rank_recipes()`
  (your recommender), `plan_meals()` (the optimizer), `lookup_nutrition()`, `get_history()`.
- **What it does:** answers things like *"what should I cook tonight to use up the paneer?"*,
  *"plan my meals for the week under 1800 kcal/day"*, *"what will I waste this week and how do
  I avoid it?"* — by calling tools and explaining the result in plain language.
- **Robustness:** works with native function-calling models, and falls back to a
  **JSON-action controller** (the agent replies with a structured action, we execute it and
  feed the result back) for free/basic models — mirrors your robust vision-JSON parsing.
- Framework-agnostic; the web layer just exposes it.

### 1C. Web + UI for the spine

- `POST /api/plan` → returns the optimized week plan + metrics.
- `POST /api/agent` → the chat endpoint (agent turn).
- New **"Meal Plan"** tab: the week grid, waste-avoided/₹-saved headline numbers, and a
  **"why this plan"** explanation from the agent.
- New **"Chef"** tab: chat with the agent.

**New:** `optimizer.py`, `agent.py`, endpoints in `app.py`, UI in `templates/` + `static/`.
**Deps:** `pulp` (in `requirements.txt`, with the pure-Python fallback so Vercel is fine).

---

## Phase 2 — Rich product features (breadth + polish)

Layer these on once the spine works. Pick any subset.

- **Nutrition tracking** — pull per-item nutrition from **Open Food Facts** (no key) and/or
  **USDA FoodData Central** (free key), cache in the Phase-0 table. Daily nutrition dashboard;
  feeds the optimizer's calorie/macro constraints. Charts follow the `dataviz` guidance.
- **Barcode scanning** — a new precise input path for packaged goods: browser-side
  `BarcodeDetector` (or a ZXing-js fallback) → exact product, expiry heuristics, and nutrition
  from Open Food Facts. Complements the vision path with hard data.
- **Meal-plan calendar** — a week/month view; mark a meal **cooked** and it **decrements
  inventory** (closing the loop from plan → consumption → new scan).
- **Waste & spend analytics** — a dashboard of what you wasted, estimated **cost of waste**
  over time, most-wasted categories, and **savings from following the plan**. Strong demo
  visuals and a real sustainability story.
- **Multi-user households** — a shared fridge: who added/used each item, simple roles.
  (Pairs with auth in Phase 3.)

**New:** `nutrition.py`, tables for meal plans + waste log, several UI panels.

---

## Phase 3 — Serious engineering & architecture

The "software-engineering rigor" story. **Honest tradeoff up front:** Vercel's serverless
model doesn't fit long-running workers, WebSockets, or Redis. So this phase introduces a
**second deployment path**, and the capstone shows both:

- **"Vercel Lite"** (what you have) — the free public demo of the core app.
- **"Full self-hosted"** (new) — the complete architecture via **Docker Compose**
  (app + Redis + Postgres), which is what you present as the real system.

Components:
- **User accounts & auth** — Flask-Login + hashed passwords (Werkzeug), or OAuth via Authlib;
  per-user/household data isolation. Enables the households feature.
- **Async background jobs** — move vision analysis + optimization to a worker (**RQ + Redis**);
  requests return immediately and the UI polls/streams progress. Real production pattern.
- **Real-time updates** — **Flask-SocketIO** so household members see inventory changes live.
- **Vector database** — move ingredient/recipe embeddings into **Chroma** or
  **Postgres + pgvector** for scalable semantic search and **RAG over recipes**. (Your JSON
  embeddings stay the fallback.)
- **Containerization + CI/CD** — `Dockerfile`, `docker-compose.yml`, and a **GitHub Actions**
  pipeline running tests + lint on every push.
- **Observability** — structured logging, a `/health` check, feature flags.

**New:** `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `auth.py`,
`worker.py`, Postgres migration path in `storage.py`.

---

## What we're deliberately NOT doing (for now)

- **Training our own vision/ML models** (Direction 2) — skipped. Your embedding recommender
  already trains itself, so you have the "we built ML" story without the cost/risk of a
  freshness or waste-prediction model. Easy to add later if a reviewer asks.
- **The "go big / maximal" everything-at-once build** — we start buildable (Phase 1), and
  only push further if time allows. Probably stop after Phase 2 + the parts of Phase 3 that
  demo well (auth + Docker + CI are the high-value ones).

---

## New files & endpoints (summary)

| Phase | New files | New endpoints |
|-------|-----------|---------------|
| 0 | `tests/`, `requirements-dev.txt` | — |
| 1 | `optimizer.py`, `agent.py` | `POST /api/plan`, `POST /api/agent` |
| 2 | `nutrition.py` | `/api/nutrition`, `/api/mealplan`, `/api/analytics` |
| 3 | `auth.py`, `worker.py`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` | `/auth/*`, `/health` |

---

## Suggested demo narrative (what you show reviewers)

1. Scan the fridge → inventory + freshness (the base, already built).
2. Open **Meal Plan** → the agent + optimizer produce a week of meals; headline: *"avoids
   1.2 kg of waste, saves ₹340 vs. shopping blind."*
3. Ask the **Chef** tab: *"I'm vegetarian and want high protein this week"* → it re-plans and
   explains why.
4. Mark a meal cooked → inventory updates; the **analytics** tab shows waste trending down.
5. (If Phase 3) Show the **Docker Compose** stack, the **CI pipeline** going green, and a
   second household member seeing a live inventory update.

---

## Open decisions for you

- **Product name** — keep "SmartFridge Vision", or rename to something distinctive
  (PantryPilot, FridgeMind, ZeroWaste Kitchen)?
- **Agent model** — reuse the OpenRouter free tier (needs the JSON-action fallback), or use a
  function-calling-capable model / the Claude API?
- **Currency/units for the "money saved" and "waste avoided" metrics** — ₹ and kg assumed.
- **How far into Phase 3** — auth + Docker + CI is the sweet spot; full Redis/WebSockets/
  pgvector is optional.

---

*Next step: build **Phase 1** (the optimizer + agent) — that's what makes this unique. Say
the word and I'll start there.*
