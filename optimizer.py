"""Zero-waste meal-plan optimizer for SmartFridge Vision (Phase 1 — the "built by us" core).

Where :mod:`recommender` *ranks* recipes one at a time, this module solves the harder,
combinatorial question a real kitchen faces: **given everything in the fridge right now,
which set of dishes should I cook over the next few days so that as little as possible is
wasted and as little as possible has to be bought?**

That is a genuine optimization problem, not a sort:

* Each on-hand ingredient is a **scarce resource**. If you have one block of paneer, only
  *one* dish in the plan may use it — you cannot suggest five paneer curries and pretend
  they all get made. The optimizer tracks per-ingredient capacity and never over-commits.
* Ingredients expiring soonest should be **consumed first**, so the plan is rewarded for
  each at-risk item it actually uses up before it spoils. This is the "zero-waste" driver.
* Buying more ingredients is a **cost**, so the plan prefers dishes you can already make
  and, when it must shop, reuses the same new ingredient across several dishes.
* The plan must fill a fixed number of **meal slots** (``days × meals_per_day``).

We express this as a small **integer linear program** (ILP) and solve it exactly with
`PuLP <https://coin-or.github.io/pulp/>`_ (the bundled CBC solver). PuLP is an *optional*
dependency: if it is not installed — or the model is trivial — the module falls back to a
**pure-Python greedy + local-search** heuristic that optimizes the *same* objective, so
the feature works everywhere (including size-constrained serverless hosts) and only gets
*better*, never broken, when the solver is present.

Like :mod:`vision_service`, :mod:`storage`, and :mod:`recommender`, this module is
**framework-agnostic**: no Flask, no HTTP. The web layer (or a future autonomous agent)
passes in the latest scan's items, the tracked "use by" dates, and — optionally — a
nutrition lookup, and gets back a plain, JSON-serializable plan. It reuses the recipe
corpus and ingredient-normalization from :mod:`recommender` so the two features always
speak about ingredients in exactly the same canonical vocabulary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import budget
import recommender

logger = logging.getLogger(__name__)

# PuLP is optional. When present we solve the meal plan exactly; when absent we fall back to
# the greedy/local-search heuristic below. Importing is wrapped broadly on purpose — a
# missing package, a broken build, or a missing CBC binary must all degrade gracefully.
try:  # pragma: no cover - import guard is environment-dependent
    import pulp  # type: ignore

    _HAS_PULP = True
except Exception:  # pragma: no cover
    pulp = None  # type: ignore
    _HAS_PULP = False


# --- Objective weights -------------------------------------------------------
# These encode the product priorities, in order: use up expiring food first, then fill the
# requested meals with high-coverage dishes, while penalizing shopping and repetition. The
# ILP and the heuristic share these exact weights and the same scoring function
# (:func:`_evaluate`), so the two engines agree on what "a good plan" means.
W_URGENT = 10.0   # reward per distinct at-risk ingredient the plan actually uses up (top priority)
W_USE = 3.0       # reward per distinct on-hand ingredient the plan clears (the "use what you have" driver)
W_COVER = 1.5     # reward per unit of recipe coverage (prefer dishes you can nearly complete)
W_FILL = 3.0      # reward per meal slot filled — the plan is expected to feed you
W_SHOP = 2.0      # penalty per *distinct* new ingredient (buying okra once serves every okra dish)
W_REPEAT = 1.5    # penalty per repeated dish (variety is nicer than cooking dal thrice)

# Candidate-recipe gates: keep the model small and the plan realistic.
_COVERAGE_FLOOR = 0.34    # skip dishes you have almost nothing for...
_MAX_MISSING = 4          # ...and dishes that would need a big shop, unless they save waste
_MAX_CANDIDATES = 60      # cap the model size (the bundled corpus is well under this)
_MAX_SLOTS = 21          # bound the horizon so the ILP stays tiny and fast

# CBC time limit (seconds). The models here solve in milliseconds; this is just a guard.
_SOLVER_TIME_LIMIT = 5


def plan_meals(
    inventory_items: Sequence[Mapping[str, Any]] | None,
    perishables: Sequence[Mapping[str, Any]] | None = None,
    *,
    now: datetime | None = None,
    days: int = 3,
    meals_per_day: int = 2,
    nutrition: Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any] | None] | None = None,
    use_soon_days: int = recommender.DEFAULT_USE_SOON_DAYS,
    solver: str = "auto",
    recipe_filter: Callable[[Mapping[str, Any]], bool] | None = None,
    cost_of: Callable[[str], float] | None = None,
    max_budget: float | None = None,
) -> dict[str, Any]:
    """Build a zero-waste meal plan for the current fridge state.

    Args:
        inventory_items: Items from the latest scan (dicts with at least ``name``, and
            optionally ``count``/``freshness``), or ``None`` for an empty fridge.
        perishables: User-tracked items with a ``use_by`` date (dicts with ``name`` and
            ``use_by``), or ``None``.
        now: Reference time (UTC) for expiry math; defaults to now. Present for testing.
        days: Length of the plan horizon in days (>= 1).
        meals_per_day: Meal slots to fill per day (>= 1). Total slots = ``days * meals_per_day``.
        nutrition: Optional per-ingredient nutrition lookup — either a mapping of
            ``token -> {kcal, protein_g, ...}`` (e.g. from :func:`storage.list_nutrition`
            keyed by ``item``) or a callable ``token -> row | None``. Used only to *report*
            the plan's estimated calories/macros; it never constrains the solution, so a
            cold/empty cache simply omits the nutrition summary.
        use_soon_days: A dated perishable within this many days counts as "at risk".
        solver: ``"auto"`` (ILP if PuLP is available, else heuristic), ``"ilp"`` (force the
            exact solver; falls back to heuristic if PuLP is missing), or ``"greedy"``
            (force the heuristic — used to exercise the fallback in tests).
        recipe_filter: Optional predicate over a raw recipe dict; rejected recipes are never
            scheduled. Used to honour remembered diet/allergy rules (see :mod:`preferences`).
        cost_of: Optional ``token -> estimated price`` callable (e.g. :func:`budget.price_of`).
            When given, the plan is priced: each meal, the shopping list, and the metrics gain
            estimated costs, and — with ``max_budget`` — the shop is constrained to fit.
        max_budget: Optional ceiling on the plan's estimated *shopping* spend (the distinct
            new ingredients it must buy; on-hand and pantry items are free). Ignored unless
            ``cost_of`` is also provided. The solver will schedule fewer/cheaper new dishes
            rather than exceed it.

    Returns:
        A JSON-serializable dict::

            {
              "generated_at": "2026-09-23T00:00:00Z",
              "horizon": {"days": 3, "meals_per_day": 2, "slots": 6},
              "solver": "ilp" | "greedy",
              "plan": [ {slot, id, title, time_min, tags, uses, uses_expiring, to_buy}, ... ],
              "shopping_list": [ {item, for_recipes: [...], count}, ... ],
              "at_risk_remaining": [ {name, token, severity, reason, days_left}, ... ],
              "metrics": { ... waste_avoided_pct, urgent_used/total, objective, nutrition? },
              "notes": [ "human-readable summary lines" ],
            }
    """
    today = recommender.local_today(now)
    slots = _slot_count(days, meals_per_day)

    capacity, urgent, use_soon = _fridge_state(
        inventory_items or [], perishables or [], today, use_soon_days
    )
    candidates = _build_candidates(capacity, urgent, recipe_filter)

    # Per-token prices for everything the plan might buy (empty when pricing isn't requested).
    prices = _price_table(candidates, cost_of)
    budget_cap = float(max_budget) if (cost_of is not None and max_budget is not None) else None

    chosen: list[int]
    engine: str
    want_ilp = solver in ("auto", "ilp") and _HAS_PULP and bool(candidates)
    if want_ilp:
        try:
            chosen = _solve_ilp(candidates, capacity, urgent, slots, prices, budget_cap)
            engine = "ilp"
        except Exception:  # pragma: no cover - solver runtime failure is environment-specific
            logger.exception("ILP solve failed; falling back to greedy heuristic")
            chosen = _solve_greedy(candidates, capacity, urgent, slots, prices, budget_cap)
            engine = "greedy"
    else:
        if solver == "ilp" and not _HAS_PULP:
            logger.info("PuLP not installed; using greedy heuristic for the meal plan")
        chosen = _solve_greedy(candidates, capacity, urgent, slots, prices, budget_cap)
        engine = "greedy"

    result = _assemble(
        chosen, candidates, capacity, urgent, use_soon, slots, today, engine, nutrition,
        prices, budget_cap, cost_of is not None,
    )
    result["horizon"] = {
        "days": max(1, int(days or 1)),
        "meals_per_day": max(1, int(meals_per_day or 1)),
        "slots": slots,
    }
    return result


# --- Fridge state -----------------------------------------------------------


def _slot_count(days: int, meals_per_day: int) -> int:
    """Total meal slots to fill, clamped to a sane, bounded range."""
    d = max(1, int(days or 1))
    m = max(1, int(meals_per_day or 1))
    return max(1, min(_MAX_SLOTS, d * m))


def _fridge_state(
    inventory_items: Sequence[Mapping[str, Any]],
    perishables: Sequence[Mapping[str, Any]],
    today: date,
    use_soon_days: int,
) -> tuple[dict[str, int], set[str], list[dict[str, Any]]]:
    """Fold the scan + perishables into (capacity, urgent, use_soon).

    * ``capacity`` — ``token -> units on hand`` for cookable ingredients. Pantry staples,
      visually spoiled items, and anything already past its use-by date are excluded (you
      can't build a waste-saving plan out of food that's already gone).
    * ``urgent`` — tokens to consume soon (freshness ``use_soon`` or a use-by within the
      window); the optimizer is rewarded for using these up.
    * ``use_soon`` — the display list of at-risk items (overdue / spoiled / soon), reused
      verbatim from :mod:`recommender` so both features show identical urgency wording.
    """
    # Reuse the recommender's inventory folding so urgency semantics stay in one place.
    have, urgent, use_soon = recommender._build_inventory_state(
        list(inventory_items), list(perishables), today, use_soon_days
    )

    # Derive per-token capacity (how many units we can actually cook with). ``have`` already
    # excludes pantry/spoiled/overdue tokens, so we only count units for tokens it kept.
    capacity: dict[str, int] = {}
    for item in inventory_items:
        token = recommender.normalize_ingredient(str(item.get("name", "")))
        if token in have:
            capacity[token] = capacity.get(token, 0) + _as_int(item.get("count"), 1)
    for entry in perishables:
        token = recommender.normalize_ingredient(str(entry.get("name", "")))
        if token in have:
            capacity[token] = capacity.get(token, 0) + 1
    # Any token that survived folding but had no explicit count gets one unit.
    for token in have:
        capacity.setdefault(token, 1)

    return capacity, urgent, use_soon


def _build_candidates(
    capacity: Mapping[str, int],
    urgent: set[str],
    recipe_filter: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Precompute the schedulable recipes and their per-plan attributes.

    A recipe is a candidate when it has at least one on-hand ingredient and is either
    reasonably complete or uses something at-risk, and doesn't demand a huge shop. Pantry
    staples are treated as always available (never "matched" capacity, never "to buy").
    Recipes rejected by ``recipe_filter`` (household diet/allergy rules) are skipped.
    """
    have = set(capacity)
    candidates: list[dict[str, Any]] = []
    for recipe in recommender._load_recipes():
        if recipe_filter is not None and not recipe_filter(recipe):
            continue
        req = {t.lower() for t in recipe.get("ingredients", [])}
        effective = req - recommender.PANTRY
        if not effective:
            continue
        matched = effective & have
        missing = sorted(effective - have)
        if not matched:
            continue
        coverage = len(matched) / len(effective)
        uses_expiring = sorted(matched & urgent)
        if coverage < _COVERAGE_FLOOR and not uses_expiring:
            continue
        if len(missing) > _MAX_MISSING:
            continue
        candidates.append(
            {
                "id": recipe.get("id"),
                "title": recipe.get("title", recipe.get("id", "Recipe")),
                "time_min": recipe.get("time_min"),
                "tags": recipe.get("tags", []),
                "match": sorted(matched),        # on-hand tokens this dish consumes (1 unit each)
                "missing": missing,               # tokens it would need bought
                "expiring": uses_expiring,        # at-risk tokens it uses up
                "coverage": coverage,
            }
        )

    # A light prescore keeps the strongest candidates if the corpus ever grows past the cap.
    candidates.sort(
        key=lambda c: (len(c["expiring"]), c["coverage"], -len(c["missing"])),
        reverse=True,
    )
    return candidates[:_MAX_CANDIDATES]


def _price_table(
    candidates: Sequence[Mapping[str, Any]],
    cost_of: Callable[[str], float] | None,
) -> dict[str, float]:
    """Price every ingredient the plan could buy or use, or ``{}`` when pricing is off.

    Built once and shared by the solvers (budget constraint), the shopping list, and the
    metrics so a token's price is looked up exactly once and every layer agrees on it.
    """
    if cost_of is None:
        return {}
    tokens: set[str] = set()
    for cand in candidates:
        tokens.update(cand["missing"])
        tokens.update(cand["match"])
    prices: dict[str, float] = {}
    for token in tokens:
        try:
            prices[token] = float(cost_of(token))
        except Exception:  # noqa: BLE001 — a bad price must never break planning
            prices[token] = 0.0
    return prices


# --- Shared objective (used by both engines and for reporting) --------------


def _evaluate(
    chosen: Sequence[int], candidates: Sequence[Mapping[str, Any]], urgent: set[str]
) -> dict[str, Any]:
    """Score a chosen multiset of candidate indices with the shared objective.

    Returns the component metrics plus the scalar ``objective`` both engines maximize, so
    the ILP result and the heuristic result are always measured on the same ruler.
    """
    used_urgent: set[str] = set()
    used_onhand: set[str] = set()
    bought: set[str] = set()
    coverage_sum = 0.0
    for r in chosen:
        cand = candidates[r]
        used_urgent.update(cand["expiring"])
        used_onhand.update(cand["match"])
        bought.update(cand["missing"])
        coverage_sum += cand["coverage"]

    filled = len(chosen)
    repeats = filled - len(set(chosen))
    objective = (
        W_URGENT * len(used_urgent)
        + W_USE * len(used_onhand)
        + W_COVER * coverage_sum
        + W_FILL * filled
        - W_SHOP * len(bought)
        - W_REPEAT * repeats
    )
    return {
        "used_urgent": used_urgent,
        "used_onhand": used_onhand,
        "bought": bought,
        "coverage_sum": coverage_sum,
        "filled": filled,
        "repeats": repeats,
        "objective": round(objective, 4),
    }


# --- Exact solver (ILP via PuLP) --------------------------------------------


def _solve_ilp(
    candidates: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, int],
    urgent: set[str],
    slots: int,
    prices: Mapping[str, float] | None = None,
    max_budget: float | None = None,
) -> list[int]:
    """Solve the meal plan exactly as an integer linear program; return chosen indices.

    Variables:
        x[r]  integer 0..slots  — how many times recipe r is scheduled.
        z[r]  binary            — whether recipe r is scheduled at all (x[r] >= 1).
        u[i]  binary            — whether at-risk ingredient i gets used up.
        b[j]  binary            — whether ingredient j has to be bought.

    Constraints: fill at most ``slots``; link z to x; never consume more of an ingredient
    than is on hand (the scarce-resource rule); u[i] only if some scheduled dish uses i;
    b[j] whenever a scheduled dish is missing j; and — when ``max_budget`` is given — the
    total estimated cost of the distinct bought ingredients stays within it. Objective = the
    shared :data:`W_*` blend.
    """
    prices = prices or {}
    prob = pulp.LpProblem("zero_waste_meal_plan", pulp.LpMaximize)

    n = len(candidates)
    x = [pulp.LpVariable(f"x_{r}", lowBound=0, upBound=slots, cat="Integer") for r in range(n)]
    z = [pulp.LpVariable(f"z_{r}", cat="Binary") for r in range(n)]

    urgent_used = sorted({i for c in candidates for i in c["expiring"]})
    onhand_used = sorted({i for c in candidates for i in c["match"]})
    buyables = sorted({j for c in candidates for j in c["missing"]})
    u = {i: pulp.LpVariable(f"u_{i}", cat="Binary") for i in urgent_used}
    v = {i: pulp.LpVariable(f"v_{i}", cat="Binary") for i in onhand_used}
    b = {j: pulp.LpVariable(f"b_{j}", cat="Binary") for j in buyables}

    # Objective: reward using up at-risk items and clearing on-hand stock, and filling slots
    # with high-coverage dishes; penalize shopping and repeated dishes. (Filling is rewarded
    # directly via W_FILL * x; at-risk items are rewarded twice — via both u and v.)
    prob += (
        W_URGENT * pulp.lpSum(u.values())
        + W_USE * pulp.lpSum(v.values())
        + W_COVER * pulp.lpSum(candidates[r]["coverage"] * x[r] for r in range(n))
        + W_FILL * pulp.lpSum(x)
        - W_SHOP * pulp.lpSum(b.values())
        - W_REPEAT * pulp.lpSum(x[r] - z[r] for r in range(n))
    )

    # Fill at most the available slots.
    prob += pulp.lpSum(x) <= slots

    # Link z[r] to x[r]: z == 1 iff x >= 1.
    for r in range(n):
        prob += x[r] <= slots * z[r]
        prob += z[r] <= x[r]

    # Scarce-resource rule: total uses of each on-hand ingredient <= units in stock.
    for token, cap in capacity.items():
        users = [x[r] for r in range(n) if token in candidates[r]["match"]]
        if users:
            prob += pulp.lpSum(users) <= cap

    # An at-risk ingredient counts as "used" only if some scheduled dish consumes it.
    for i in urgent_used:
        users = [x[r] for r in range(n) if i in candidates[r]["expiring"]]
        prob += u[i] <= pulp.lpSum(users)

    # Likewise, an on-hand ingredient counts as "cleared" only if a scheduled dish uses it.
    for i in onhand_used:
        users = [x[r] for r in range(n) if i in candidates[r]["match"]]
        prob += v[i] <= pulp.lpSum(users)

    # Scheduling a dish forces buying each ingredient it is missing.
    for r in range(n):
        for j in candidates[r]["missing"]:
            prob += b[j] >= z[r]

    # Budget ceiling: the estimated cost of everything the plan buys must fit the cap. Each
    # ingredient is bought once (b[j] is binary), so the same new okra serves every okra dish.
    if max_budget is not None:
        prob += pulp.lpSum(b[j] * float(prices.get(j, 0.0)) for j in buyables) <= max_budget

    solver_cmd = pulp.PULP_CBC_CMD(msg=0, timeLimit=_SOLVER_TIME_LIMIT)
    prob.solve(solver_cmd)

    # Expand x[r] into a flat list of chosen indices (a dish scheduled twice appears twice).
    chosen: list[int] = []
    for r in range(n):
        count = int(round(pulp.value(x[r]) or 0))
        chosen.extend([r] * max(0, count))
    return chosen


# --- Heuristic solver (greedy + local search) -------------------------------


def _solve_greedy(
    candidates: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, int],
    urgent: set[str],
    slots: int,
    prices: Mapping[str, float] | None = None,
    max_budget: float | None = None,
) -> list[int]:
    """Approximate the plan with a greedy fill, then improve it with local search.

    Greedy: repeatedly add the still-feasible dish with the best *marginal* objective gain
    until the slots are full or nothing helps. Local search: try replacing each scheduled
    dish with any other candidate and keep the swap if it raises the shared objective. This
    optimizes exactly the objective in :func:`_evaluate`, so its plans track the ILP's.

    When ``max_budget`` is given, a dish is only feasible if adding it keeps the plan's total
    estimated shopping cost (the distinct ingredients bought across all chosen dishes) within
    the cap — the same budget the ILP enforces, so both engines respect the ceiling.
    """
    prices = prices or {}

    def _shop_cost(indices: Sequence[int]) -> float:
        """Estimated cost of the distinct new ingredients a set of dishes must buy."""
        bought = {t for r in indices for t in candidates[r]["missing"]}
        return sum(float(prices.get(t, 0.0)) for t in bought)

    chosen: list[int] = []
    remaining = dict(capacity)  # units left after the dishes chosen so far

    def _feasible(r: int) -> bool:
        if any(remaining.get(t, 0) < 1 for t in candidates[r]["match"]):
            return False
        if max_budget is not None and _shop_cost([*chosen, r]) > max_budget:
            return False
        return True

    def _consume(r: int) -> None:
        for t in candidates[r]["match"]:
            remaining[t] = remaining.get(t, 0) - 1

    # --- Greedy fill ---
    for _ in range(slots):
        best_r, best_gain = -1, 0.0
        base = _evaluate(chosen, candidates, urgent)["objective"]
        for r in range(len(candidates)):
            if not _feasible(r):
                continue
            gain = _evaluate([*chosen, r], candidates, urgent)["objective"] - base
            if gain > best_gain:
                best_r, best_gain = r, gain
        if best_r < 0:
            break  # no feasible dish improves the plan — leave the remaining slots empty
        chosen.append(best_r)
        _consume(best_r)

    # --- Local search: single-dish replacement while it improves the objective ---
    improved = True
    while improved:
        improved = False
        for pos in range(len(chosen)):
            trial_base = chosen[:pos] + chosen[pos + 1:]
            # Capacity available if we drop the dish at ``pos``.
            avail = dict(capacity)
            for r in trial_base:
                for t in candidates[r]["match"]:
                    avail[t] = avail.get(t, 0) - 1
            current = _evaluate(chosen, candidates, urgent)["objective"]
            best_alt, best_obj = chosen[pos], current
            for r in range(len(candidates)):
                if any(avail.get(t, 0) < 1 for t in candidates[r]["match"]):
                    continue
                if max_budget is not None and _shop_cost([*trial_base, r]) > max_budget:
                    continue
                obj = _evaluate([*trial_base, r], candidates, urgent)["objective"]
                if obj > best_obj:
                    best_alt, best_obj = r, obj
            if best_alt != chosen[pos]:
                chosen[pos] = best_alt
                improved = True
                break  # restart the scan; capacities shifted

    return chosen


# --- Result assembly --------------------------------------------------------


def _assemble(
    chosen: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    capacity: Mapping[str, int],
    urgent: set[str],
    use_soon: list[dict[str, Any]],
    slots: int,
    today: date,
    engine: str,
    nutrition: Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any] | None] | None,
    prices: Mapping[str, float] | None = None,
    budget_cap: float | None = None,
    priced: bool = False,
) -> dict[str, Any]:
    """Turn the chosen indices into the public plan payload with metrics and notes.

    When ``priced`` (i.e. the caller supplied a ``cost_of`` hook), every meal, the shopping
    list, and the metrics gain estimated costs in the household currency, and — if a
    ``budget_cap`` was set — the metrics report the budget and whether the plan fits it.
    """
    prices = prices or {}
    metrics = _evaluate(chosen, candidates, urgent)
    used_urgent: set[str] = metrics["used_urgent"]

    def _sum_price(tokens: Sequence[str]) -> float:
        return round(sum(float(prices.get(t, 0.0)) for t in tokens), 2)

    # Order the schedule so the most waste-saving, quickest dishes come first.
    order = sorted(
        range(len(chosen)),
        key=lambda k: (
            -len(candidates[chosen[k]]["expiring"]),
            candidates[chosen[k]]["time_min"] or 999,
            -candidates[chosen[k]]["coverage"],
        ),
    )
    plan: list[dict[str, Any]] = []
    for slot_no, k in enumerate(order, start=1):
        cand = candidates[chosen[k]]
        meal: dict[str, Any] = {
            "slot": slot_no,
            "id": cand["id"],
            "title": cand["title"],
            "time_min": cand["time_min"],
            "tags": cand["tags"],
            "uses": [recommender._prettify(t) for t in cand["match"]],
            "uses_expiring": [recommender._prettify(t) for t in cand["expiring"]],
            "to_buy": [recommender._prettify(t) for t in cand["missing"]],
        }
        if priced:
            # ``est_cost`` is what this dish adds to the shop (its missing items); ``pantry_value``
            # is the worth of what it uses from the fridge — the money the plan saves by cooking it.
            meal["est_cost"] = _sum_price(cand["missing"])
            meal["pantry_value"] = _sum_price(cand["match"])
        plan.append(meal)

    shopping_list = _shopping_list(chosen, candidates, prices if priced else None)
    at_risk_remaining = [
        row for row in use_soon
        if row["severity"] == "soon" and row["token"] not in used_urgent
    ]

    urgent_total = len(urgent)
    waste_avoided = (
        round(100.0 * len(used_urgent) / urgent_total, 1) if urgent_total else None
    )
    result_metrics: dict[str, Any] = {
        "slots": slots,
        "slots_filled": len(chosen),
        "urgent_total": urgent_total,
        "urgent_used": len(used_urgent),
        "waste_avoided_pct": waste_avoided,
        "distinct_ingredients_used": len({t for k in chosen for t in candidates[k]["match"]}),
        "new_ingredients_to_buy": len(metrics["bought"]),
        "objective": metrics["objective"],
    }
    if priced:
        # The shop's cost counts each distinct new ingredient once (buy okra once for every okra
        # dish); pantry value is the total worth of on-hand stock the plan actually consumes.
        shopping_cost = _sum_price(sorted(metrics["bought"]))
        pantry_value = _sum_price(sorted({t for k in chosen for t in candidates[k]["match"]}))
        result_metrics["currency"] = budget.CURRENCY
        result_metrics["shopping_cost"] = shopping_cost
        result_metrics["pantry_value_used"] = pantry_value
        if budget_cap is not None:
            result_metrics["budget"] = round(float(budget_cap), 2)
            result_metrics["within_budget"] = shopping_cost <= budget_cap + 1e-6
    nutrition_summary = _nutrition_summary(chosen, candidates, nutrition)
    if nutrition_summary:
        result_metrics["nutrition"] = nutrition_summary

    return {
        "generated_at": today.strftime("%Y-%m-%dT00:00:00Z"),
        "horizon": {"days": None, "meals_per_day": None, "slots": slots},
        "solver": engine,
        "plan": plan,
        "shopping_list": shopping_list,
        "at_risk_remaining": at_risk_remaining,
        "metrics": result_metrics,
        "notes": _notes(plan, result_metrics, at_risk_remaining, engine),
    }


def _shopping_list(
    chosen: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    prices: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate everything the plan needs bought, biggest shared ingredient first.

    When ``prices`` is given, each row carries the ingredient's estimated ``est_cost`` so the
    UI can show a priced list without re-looking-up anything.
    """
    by_item: dict[str, list[str]] = {}
    for r in chosen:
        for token in candidates[r]["missing"]:
            by_item.setdefault(token, [])
            if candidates[r]["title"] not in by_item[token]:
                by_item[token].append(candidates[r]["title"])
    ranked = sorted(by_item.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    rows: list[dict[str, Any]] = []
    for token, titles in ranked:
        row: dict[str, Any] = {
            "item": recommender._prettify(token),
            "for_recipes": titles,
            "count": len(titles),
        }
        if prices is not None:
            row["est_cost"] = round(float(prices.get(token, 0.0)), 2)
        rows.append(row)
    return rows


def _nutrition_summary(
    chosen: Sequence[int],
    candidates: Sequence[Mapping[str, Any]],
    nutrition: Mapping[str, Mapping[str, Any]] | Callable[[str], Mapping[str, Any] | None] | None,
) -> dict[str, Any] | None:
    """Estimate the plan's calories/macros from cached nutrition, or None if unavailable.

    Purely informational — sums the per-ingredient rows for every ingredient the plan
    consumes. Missing rows are simply skipped, and if nothing is known the summary is
    omitted entirely (the cache may be empty on a fresh install).
    """
    if nutrition is None:
        return None
    lookup = nutrition if callable(nutrition) else (lambda t: nutrition.get(t))

    totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    known = 0
    for r in chosen:
        for token in candidates[r]["match"]:
            row = lookup(token)
            if not row:
                continue
            known += 1
            for key in totals:
                val = row.get(key)
                if isinstance(val, (int, float)):
                    totals[key] += float(val)
    if not known:
        return None
    return {
        "ingredients_with_data": known,
        "kcal": round(totals["kcal"], 1),
        "protein_g": round(totals["protein_g"], 1),
        "carbs_g": round(totals["carbs_g"], 1),
        "fat_g": round(totals["fat_g"], 1),
        "basis": "sum of cached per-ingredient values (typically per 100 g)",
    }


def _notes(
    plan: list[dict[str, Any]],
    metrics: Mapping[str, Any],
    at_risk_remaining: list[dict[str, Any]],
    engine: str,
) -> list[str]:
    """A few human-readable summary lines for the UI / agent to relay."""
    notes: list[str] = []
    if not plan:
        notes.append("No cookable plan yet — scan the fridge or add a few staples first.")
        return notes

    engine_label = "exact optimizer (ILP)" if engine == "ilp" else "fast heuristic solver"
    notes.append(f"Planned {len(plan)} meal(s) with the {engine_label}.")

    if metrics["urgent_total"]:
        notes.append(
            f"Uses up {metrics['urgent_used']} of {metrics['urgent_total']} at-risk "
            f"item(s) ({metrics['waste_avoided_pct']}% of what's about to spoil)."
        )
    if metrics["new_ingredients_to_buy"]:
        buy_note = f"Needs {metrics['new_ingredients_to_buy']} ingredient(s) bought"
        if "shopping_cost" in metrics:
            buy_note += f" (~{metrics['currency']}{metrics['shopping_cost']:g} to shop)"
        notes.append(buy_note + ".")
    else:
        notes.append("Everything in this plan is already in your kitchen.")
    if metrics.get("budget") is not None:
        cur, cost, cap = metrics["currency"], metrics["shopping_cost"], metrics["budget"]
        if metrics.get("within_budget"):
            notes.append(f"Within your {cur}{cap:g} budget ({cur}{cost:g} spent).")
        else:
            notes.append(
                f"Over the {cur}{cap:g} budget by {cur}{cost - cap:g} — "
                "raise the budget or free up more from the fridge."
            )
    if metrics.get("pantry_value_used"):
        notes.append(
            f"Uses ~{metrics['currency']}{metrics['pantry_value_used']:g} of food you "
            "already have."
        )
    if at_risk_remaining:
        names = ", ".join(row["name"] for row in at_risk_remaining[:3])
        notes.append(f"Still at risk (no recipe fit): {names}.")
    return notes


# --- Small helper -----------------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    """Coerce a value to a non-negative int, falling back to ``default``."""
    try:
        n = int(value)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default
