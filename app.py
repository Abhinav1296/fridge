"""SmartFridge Vision MVP — Flask web layer.

A thin HTTP wrapper around :mod:`vision_service` and :mod:`storage`:

* ``GET  /``            serves the single-page UI.
* ``POST /analyze``     accepts an image (multipart file upload OR base64 JSON from the
  webcam), validates and downscales it, hands the bytes to the vision service, records
  the result to the history database, and returns the parsed inventory as JSON.
* ``GET  /api/current`` returns the current fridge inventory (the most recent scan).
* ``GET  /api/history`` returns recent scans (timestamped inventory history) as JSON.
* ``GET  /api/recommendations`` returns recipe / use-soon / shopping suggestions built by
  :mod:`recommender` from the current inventory plus tracked "use by" dates.
* ``POST /api/plan`` returns a zero-waste meal plan from :mod:`optimizer` (which dishes to
  cook so the least food is wasted and the least must be bought).
* ``POST /api/agent`` answers a natural-language message with the Chef agent (:mod:`agent`):
  it looks things up with tools, remembers household preferences, keeps the shopping list,
  and *proposes* fridge changes that wait for the user's Confirm. Without an LLM (or when
  the provider is down) a rules-based fallback (:mod:`offline_agent`) answers instead.
  Progress streams live over Socket.IO (``agent_step``).
* ``/api/agent/actions`` lists proposed changes; ``.../<id>/confirm`` runs one through
  :mod:`kitchen`, ``.../<id>/cancel`` drops it.
* ``/api/agent/memory`` (GET/POST/DELETE) shows and edits what Chef remembers;
  ``/api/shopping`` (GET/POST/PATCH/DELETE) is the shared shopping list.
* ``/api/agent/briefing`` returns / regenerates the proactive briefing from :mod:`watcher`
  (also refreshed after every scan, every confirmed change, and on a timer).
* ``GET  /api/nutrition`` returns the nutrition dashboard for the current fridge (cached
  per-ingredient calories/macros plus per-100 g totals and coverage).
* ``POST /api/nutrition/refresh`` fills the nutrition cache for the current inventory from
  Open Food Facts (:mod:`nutrition`).
* ``POST /api/barcode`` looks up a scanned packaged product by barcode (Open Food Facts)
  and caches its nutrition.
* ``/api/perishables`` (GET/POST/DELETE) manages the user's tracked "use by" dates.

A Socket.IO layer pushes a ``state_changed`` hint to every open tab whenever a mutating
route changes the fridge, so views refresh live without polling (see :func:`_emit_state`).

The web layer owns request/response concerns only. All vision logic lives in
:mod:`vision_service`, recommendation logic in :mod:`recommender`, meal-plan optimization
in :mod:`optimizer`, the conversational agent in :mod:`agent`, and persistence in
:mod:`storage`, so a future agent layer can bypass HTTP entirely.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import re
import threading
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from PIL import Image, UnidentifiedImageError

import agent
import analytics
import budget
import cookmode
import kitchen
import nutrition
import offline_agent
import optimizer
import preferences
import receipts
import recommender
import restock
import storage
import vectorstore
import vision_service
import watcher
from agent import AgentAPIError, AgentConfigError
from nutrition import NutritionError
from vision_service import (
    RateLimitError,
    VisionAPIError,
    VisionConfigError,
    VisionParseError,
)

# Load .env before anything reads configuration.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smartfridge.app")

# --- Limits -----------------------------------------------------------------

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # Reject decoded images larger than 5 MB.
MAX_DIMENSION = 1600  # Downscale so the longest side is at most this many pixels.
JPEG_QUALITY = 85

# Hard backstop on the raw request body. A 5 MB image base64-encoded inside a JSON
# body is ~6.7 MB, so allow headroom above MAX_IMAGE_BYTES; oversized bodies are
# turned into a friendly 413 by the error handler below.
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Real-time layer. When the fridge changes on the server (a new scan, a cooked meal, a
# tracked item resolved, waste logged, nutrition refreshed) we push a lightweight
# "state_changed" hint to every open browser tab so they refresh the affected views live —
# no polling, no manual reload. ``async_mode="threading"`` runs on the plain Werkzeug/gthread
# server (no eventlet/gevent needed); the default same-origin CORS policy is kept. Broadcasts
# fan out from a single process — to scale past one worker, back this with the Redis message
# queue that arrives with the async-jobs phase (``SocketIO(..., message_queue=...)``).
socketio = SocketIO(app, async_mode="threading", logger=False, engineio_logger=False)


def _emit_state(*scopes: str, source: str | None = None) -> None:
    """Broadcast a "these views changed, please refresh" hint to all connected clients.

    Best-effort and never in the request's critical path: the HTTP mutation has already
    succeeded by the time this runs, so a socket failure is logged and swallowed rather
    than turned into an error the user sees. ``scopes`` name the client views to refresh
    (e.g. "inventory", "analytics", "perishables", "nutrition").
    """
    try:
        socketio.emit("state_changed", {"scopes": list(scopes), "source": source})
    except Exception:  # noqa: BLE001 — realtime is additive; it must never break a response
        logger.exception("Failed to emit realtime state update")


@socketio.on("connect")
def _on_connect() -> None:
    """Log new real-time subscribers. No server state is pushed on connect — the page
    loads its own initial data over HTTP; sockets carry only subsequent change hints."""
    logger.info("Realtime client connected")
    _ensure_watch_timer()


# --- Proactive watcher ------------------------------------------------------

_watch_started = False
_watch_guard = threading.Lock()


def _auto_briefings() -> bool:
    """Scan/action-triggered briefings are on unless ``AGENT_AUTO_BRIEFING=0``."""
    return os.getenv("AGENT_AUTO_BRIEFING", "1").strip() != "0"


def _refresh_briefing(trigger: str) -> None:
    """Let the watcher re-check the fridge off the request thread (never delays a response)."""
    if _auto_briefings():
        socketio.start_background_task(_briefing_task, trigger)


def _briefing_task(trigger: str) -> None:
    try:
        watcher.generate_briefing(trigger)
    except Exception:  # noqa: BLE001 — a background check must never crash the server
        logger.exception("Watcher failed (trigger=%s)", trigger)
        return
    # Also covers "nothing new" / "stale briefing dismissed": the client just re-reads.
    _emit_state("briefing", "agent", source=f"watcher-{trigger}")


def _ensure_watch_timer() -> None:
    """Start the periodic watcher once per process (lazily, on the first live client)."""
    global _watch_started
    minutes = watcher.watch_minutes()
    if minutes <= 0 or not _auto_briefings():
        return
    with _watch_guard:
        if _watch_started:
            return
        _watch_started = True
    socketio.start_background_task(_watch_loop, minutes)


def _watch_loop(minutes: int) -> None:
    logger.info("Watcher timer started (every %d min)", minutes)
    socketio.sleep(3)  # first look right after the first client connects (cheap: deduped)
    while True:
        _briefing_task("timer")
        socketio.sleep(minutes * 60)


# Ensure the history database exists before serving. A DB problem must not take the
# whole app down — analysis still works without history — so failures are logged, not
# raised.
try:
    storage.init_db()
except Exception:  # noqa: BLE001 — degrade gracefully if the DB can't be opened
    logger.exception("Could not initialize the history database; history disabled")

_DATA_URL_PREFIX = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)


class ImageInputError(Exception):
    """A client-side problem with the submitted image (bad/missing/oversized)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.get("/")
def index() -> str:
    """Serve the single-page UI."""
    return render_template("index.html")


@app.get("/health")
@app.get("/healthz")
def health():
    """Liveness/readiness probe for containers, load balancers, and CI.

    Reports database reachability plus which optional services are configured (no network
    calls to them). Returns 200 when the database answers, 503 when it doesn't — so an
    orchestrator can tell a degraded instance from a healthy one. Never leaks secrets: only
    the backend *kind* and whether keys are present, never URLs, paths, or values.
    """
    db_ok = storage.ping()
    body = {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "down",
        "backend": storage.backend(),
        "vision_configured": bool(os.getenv("VISION_API_KEY", "").strip()),
        "agent_configured": agent.configured(),
    }
    return jsonify(body), (200 if db_ok else 503)


@app.post("/analyze")
def analyze():
    """Analyze an uploaded or captured image and return the structured inventory."""
    source = _request_source()
    try:
        raw_bytes = _extract_image_bytes()
        prepared = _prepare_image(raw_bytes)
        result = vision_service.analyze_image(prepared)
    except ImageInputError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except VisionConfigError as exc:
        logger.error("Vision service misconfigured: %s", exc)
        return (
            jsonify(
                {
                    "error": "The vision service isn't configured on the server. "
                    "Check the VISION_API_KEY setting."
                }
            ),
            500,
        )
    except RateLimitError as exc:
        return jsonify({"error": str(exc)}), 429
    except VisionParseError as exc:
        return jsonify({"error": str(exc)}), 502
    except VisionAPIError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:  # noqa: BLE001 — last-resort guard so we never leak a stack trace
        logger.exception("Unexpected error while analyzing image")
        return (
            jsonify({"error": "Something went wrong while analyzing the image."}),
            500,
        )

    # Record the scan to history. This is additive and must never break the response,
    # so a storage failure is logged and the analysis is still returned to the user.
    try:
        storage.save_scan(result, source)
    except Exception:  # noqa: BLE001 — history is best-effort, analysis is what matters
        logger.exception("Failed to record scan to history (analysis still returned)")
    else:
        # A new scan replaces the fridge snapshot — nudge open tabs to refresh inventory,
        # and let the watcher look for anything about to go off.
        _emit_state("inventory", source="scan")
        _refresh_briefing("scan")

    return jsonify(result)


@app.get("/api/current")
def api_current():
    """Return the current fridge inventory — the most recent scan — for the Fridge tab.

    A scan is a full snapshot of the fridge; the latest one — with anything cooked since
    then subtracted (see :func:`storage.get_current_inventory`) — is "what's in the fridge
    now". Responds ``{"scan": null}`` when there is no history yet.
    """
    try:
        scan = _current_inventory()
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to read current inventory")
        return jsonify({"error": "Couldn't load your fridge right now."}), 500
    return jsonify({"scan": scan})


@app.get("/api/history")
def api_history():
    """Return recent scans (timestamped inventory history) as JSON for the History tab."""
    try:
        scans = storage.list_scans(limit=50)
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to read history")
        return jsonify({"error": "Couldn't load history right now."}), 500
    return jsonify({"scans": scans})


@app.get("/api/recommendations")
def api_recommendations():
    """Return recipe / use-soon / shopping suggestions for the current fridge.

    Combines the latest scan (what's visibly in the fridge) with the user's tracked
    "use by" dates, and runs the embedding-based recommender over both. The household
    rules Chef remembers (diet, allergies, dislikes) filter the recipes. Returns empty
    lists (not an error) when there's nothing to suggest yet.
    """
    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
        # Read with the rest: if the rules can't be loaded we must not suggest (say) an
        # allergen, so this fails the request rather than silently dropping the filter.
        rules = preferences.build_filter(storage.get_memory())
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load data for recommendations")
        return jsonify({"error": "Couldn't load suggestions right now."}), 500

    items = (scan or {}).get("items", [])
    try:
        result = recommender.recommend(items, perishables, recipe_filter=rules)
    except Exception:  # noqa: BLE001 — recommender must never take the page down
        logger.exception("Recommender failed")
        return jsonify({"error": "Couldn't build suggestions right now."}), 500
    return jsonify(result)


@app.post("/api/plan")
def api_plan():
    """Return a zero-waste, budget-aware meal plan for the current fridge.

    Body (all optional):
        ``days``          int 1-7   — horizon length (default 3).
        ``meals_per_day`` int 1-4   — slots per day (default 2).
        ``budget``        number    — a ceiling on the plan's estimated *shopping* spend, in
                                      the household currency. Omit (or send ``null``) for no cap.

    Combines the latest scan, tracked "use by" dates, and the cached nutrition table, then runs
    :func:`optimizer.plan_meals` (exact ILP when PuLP is available, else the built-in
    heuristic). The plan is always **priced** — every meal, the shopping list, and the metrics
    carry estimated costs (see :mod:`budget`) — and grouped into per-day buckets under
    ``by_day`` so the Autopilot view can show one card per day. When ``budget`` is given the
    shop is constrained to fit and ``metrics.within_budget`` reports whether it did. Returns an
    empty plan (not an error) when there's nothing to cook yet.
    """
    payload = request.get_json(silent=True) or {}
    days = _clamp_int(payload.get("days"), default=3, lo=1, hi=7)
    meals_per_day = _clamp_int(payload.get("meals_per_day"), default=2, lo=1, hi=4)
    max_budget = _clamp_budget(payload.get("budget"))

    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
        nutrition = {row["item"]: row for row in storage.list_nutrition() if row.get("item")}
        rules = preferences.build_filter(storage.get_memory())
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load data for the meal plan")
        return jsonify({"error": "Couldn't load your fridge for planning right now."}), 500

    items = (scan or {}).get("items", [])
    try:
        plan = optimizer.plan_meals(
            items,
            perishables,
            days=days,
            meals_per_day=meals_per_day,
            nutrition=nutrition or None,
            recipe_filter=rules,
            cost_of=budget.price_of,
            max_budget=max_budget,
        )
        plan["by_day"] = budget.group_by_day(plan["plan"], meals_per_day)
    except Exception:  # noqa: BLE001 — the planner must never take the page down
        logger.exception("Meal-plan optimizer failed")
        return jsonify({"error": "Couldn't build a meal plan right now."}), 500
    return jsonify(plan)


# --- Semantic recipe search --------------------------------------------------

# Built once from the recipe corpus + the recommender's embedding math, then
# reused across requests. Rebuilt only if explicitly reset (tests / data reload).
_recipe_searcher: vectorstore.RecipeSearcher | None = None


def _get_recipe_searcher() -> vectorstore.RecipeSearcher:
    """Return the process-wide recipe search index, building it on first use."""
    global _recipe_searcher
    if _recipe_searcher is None:
        _recipe_searcher = vectorstore.RecipeSearcher(
            recommender.all_recipes(),
            embed=recommender.embed_ingredients,
            normalize=recommender.normalize_ingredient,
            prettify=recommender._prettify,
        )
    return _recipe_searcher


@app.get("/api/recipes/search")
def api_recipe_search():
    """Semantic + lexical search over the whole recipe corpus.

    Query params: ``q`` (the search text) and optional ``k`` (1-12, default 6).
    Ranks by meaning (ingredient-embedding similarity) blended with word overlap
    on titles/tags, so it works for both ingredient and descriptor queries.
    Returns an empty result set (not an error) for a blank query.
    """
    query = (request.args.get("q") or "").strip()
    k = _clamp_int(request.args.get("k"), default=6, lo=1, hi=12)
    if not query:
        return jsonify({"query": "", "results": [], "count": 0, "semantic": False})
    try:
        searcher = _get_recipe_searcher()
        results = searcher.search(query, k=k)
    except Exception:  # noqa: BLE001 — search must never take the page down
        logger.exception("Recipe search failed")
        return jsonify({"error": "Couldn't search recipes right now."}), 500
    return jsonify(
        {"query": query, "results": results, "count": len(results), "semantic": searcher.embedded}
    )


@app.get("/api/recipes/<recipe_id>/similar")
def api_recipe_similar(recipe_id: str):
    """Return recipes most semantically similar to ``recipe_id`` (nearest neighbours).

    Optional ``k`` query param (1-8, default 4). 404 if the recipe id is unknown.
    """
    k = _clamp_int(request.args.get("k"), default=4, lo=1, hi=8)
    try:
        searcher = _get_recipe_searcher()
        results = searcher.similar(recipe_id, k=k)
    except Exception:  # noqa: BLE001 — a lookup failure shouldn't 500 the page
        logger.exception("Similar-recipe lookup failed")
        return jsonify({"error": "Couldn't find similar recipes right now."}), 500
    if not results and recipe_id not in searcher.index:
        return jsonify({"error": "Unknown recipe."}), 404
    return jsonify({"id": recipe_id, "results": results, "count": len(results)})


@app.get("/api/cook/<recipe_id>")
def api_cook_session(recipe_id: str):
    """Return a Cook Mode session for one recipe: synthesised steps + ingredient swap ideas.

    Each ingredient is marked have/missing against the current fridge (scan + tracked
    perishables), and swap ideas never include anything the household avoids (diet, allergy
    or dislike). Read-only — cooking the dish is a separate, confirm-gated agent action.
    404 if the recipe id is unknown.
    """
    try:
        recipe = next(
            (r for r in recommender.all_recipes() if r.get("id") == recipe_id), None
        )
        if recipe is None:
            return jsonify({"error": "Unknown recipe."}), 404
        scan = _current_inventory()
        perishables = storage.list_perishables()
        memory = storage.get_memory()
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load data for cook mode")
        return jsonify({"error": "Couldn't open cook mode right now."}), 500

    items = (scan or {}).get("items", [])
    have = nutrition.tokens_from_inventory(items, perishables)
    exclude = set(preferences.excluded_tokens(memory))
    try:
        session = cookmode.build_cook_session(
            recipe, have_tokens=have, exclude_tokens=exclude
        )
    except Exception:  # noqa: BLE001 — cook mode must never take the page down
        logger.exception("Cook Mode session build failed for %s", recipe_id)
        return jsonify({"error": "Couldn't build the cooking guide right now."}), 500
    return jsonify(session)


@app.get("/api/nutrition")
def api_nutrition():
    """Return the nutrition dashboard for the current fridge.

    For every ingredient in the latest scan (plus tracked perishables), reports the cached
    per-100 g calories/macros (``null`` where not yet fetched), the summed totals across
    ingredients that have data, and how much of the fridge is covered. Empty (not an error)
    when there's no inventory or nothing is cached yet.
    """
    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
        cached = {row["item"]: row for row in storage.list_nutrition() if row.get("item")}
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load data for the nutrition dashboard")
        return jsonify({"error": "Couldn't load nutrition right now."}), 500

    items = (scan or {}).get("items", [])
    tokens = nutrition.tokens_from_inventory(items, perishables)

    dashboard: list[dict] = []
    totals = {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
    with_data = 0
    for token in tokens:
        row = cached.get(token)
        entry = {
            "token": token,
            "name": recommender._prettify(token),
            "kcal": (row or {}).get("kcal"),
            "protein_g": (row or {}).get("protein_g"),
            "carbs_g": (row or {}).get("carbs_g"),
            "fat_g": (row or {}).get("fat_g"),
            "source": (row or {}).get("source"),
        }
        dashboard.append(entry)
        if row and any(entry[k] is not None for k in totals):
            with_data += 1
            for key in totals:
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += float(value)

    return jsonify(
        {
            "items": dashboard,
            "totals": {k: round(v, 1) for k, v in totals.items()},
            "coverage": {"tracked": len(tokens), "with_data": with_data},
            "cached_count": len(cached),
            "basis": "per 100 g, summed across ingredients with data",
        }
    )


@app.post("/api/nutrition/refresh")
def api_nutrition_refresh():
    """Fetch missing nutrition for the current inventory from Open Food Facts.

    Body (optional): ``{"force": bool}`` to re-fetch even already-cached items. Enriches the
    tokens in the latest scan + tracked perishables and returns a per-token outcome summary.
    Individual lookup failures are reported, not fatal; the cache keeps whatever it had.
    """
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force"))

    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load inventory for nutrition refresh")
        return jsonify({"error": "Couldn't load your fridge right now."}), 500

    items = (scan or {}).get("items", [])
    tokens = nutrition.tokens_from_inventory(items, perishables)
    if not tokens:
        return jsonify({"updated": [], "cached": [], "not_found": [], "failed": [], "source": nutrition.SOURCE})

    try:
        summary = nutrition.enrich(
            tokens,
            get_cached=storage.get_nutrition,
            upsert=storage.upsert_nutrition,
            force=force,
        )
    except Exception:  # noqa: BLE001 — enrichment must never take the page down
        logger.exception("Nutrition refresh failed")
        return jsonify({"error": "Couldn't refresh nutrition right now."}), 500
    if summary.get("updated"):
        _emit_state("nutrition", source="nutrition-refresh")
    return jsonify(summary)


@app.post("/api/barcode")
def api_barcode():
    """Look up a scanned packaged product by barcode and cache its nutrition.

    Body: ``{"code": "<digits>"}``. Returns the product name/brand and per-100 g macros
    from Open Food Facts, caching them under the product's canonical token so the optimizer
    and dashboard can use them. Returns 404 when the barcode is unknown, 502 when OFF is
    unreachable.
    """
    payload = request.get_json(silent=True) or {}
    code = "".join(ch for ch in str(payload.get("code", "")) if ch.isdigit())
    if not (8 <= len(code) <= 14):
        return jsonify({"error": "Please provide a valid barcode (8–14 digits)."}), 400

    try:
        product = nutrition.fetch_by_barcode(code)
    except NutritionError as exc:
        logger.warning("Barcode lookup failed: %s", exc)
        return jsonify({"error": "Couldn't reach the product database. Try again."}), 502
    except Exception:  # noqa: BLE001 — never leak a stack trace
        logger.exception("Unexpected error during barcode lookup")
        return jsonify({"error": "Something went wrong looking up that barcode."}), 500

    if not product or not product.get("product_name"):
        return jsonify({"error": "That barcode wasn't found in the product database."}), 404

    name = product["product_name"]
    token = recommender.normalize_ingredient(name)
    has_nutrition = any(product.get(k) is not None for k in ("kcal", "protein_g", "carbs_g", "fat_g"))
    if token and has_nutrition:
        try:
            storage.upsert_nutrition(
                token,
                kcal=product.get("kcal"),
                protein_g=product.get("protein_g"),
                carbs_g=product.get("carbs_g"),
                fat_g=product.get("fat_g"),
                source=product.get("source", nutrition.SOURCE),
            )
        except Exception:  # noqa: BLE001 — caching is best-effort
            logger.exception("Failed to cache nutrition for scanned product")
        else:
            _emit_state("nutrition", source="barcode")

    return jsonify(
        {
            "product": {
                "code": product.get("code", code),
                "name": name,
                "brands": product.get("brands"),
                "token": token,
                "kcal": product.get("kcal"),
                "protein_g": product.get("protein_g"),
                "carbs_g": product.get("carbs_g"),
                "fat_g": product.get("fat_g"),
            },
            "cached": bool(token and has_nutrition),
            "has_nutrition": has_nutrition,
        }
    )


# --- Chef agent ---------------------------------------------------------------

_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
ACTION_MAX_AGE = timedelta(hours=24)   # an unconfirmed proposal older than this is stale


def _agent_tools(*, source: str = "chat") -> dict[str, agent.Tool]:
    """Chef's full toolset, wired to the real storage / recommender / search / kitchen."""
    return agent.build_default_tools(
        get_latest_scan=_current_inventory,
        list_perishables=storage.list_perishables,
        list_nutrition=storage.list_nutrition,
        get_memory=storage.get_memory,
        set_memory=storage.set_memory,
        delete_memory=storage.delete_memory,
        search_recipes=lambda query, k: _get_recipe_searcher().search(query, k=k),
        waste_summary=lambda: analytics.summarize(storage.list_waste_events()),
        list_shopping=storage.list_shopping,
        add_shopping=kitchen.add_shopping,
        propose=lambda tool, args, summary: storage.create_action(
            tool, args, summary, source=source
        ),
    )


@app.post("/api/agent")
def api_agent():
    """Talk to Chef, the kitchen agent.

    Body: ``{"message": str, "history": [{"role", "content"}]?, "run_id": str?, "sid": str?}``.

    Chef works in a loop — look things up with tools, decide, act — and returns
    ``{"answer", "steps", "mode", "model", "actions", "changed"}``:

    * it may **save** household preferences and **add** to the shopping list directly
      (reversible, shown in the UI);
    * anything that changes the fridge is only **proposed** (``actions``) and waits for the
      user's Confirm (``POST /api/agent/actions/<id>/confirm``).

    With ``run_id`` + ``sid`` (the caller's Socket.IO id), each step streams to that tab as
    an ``agent_step`` event while Chef works. Without an LLM configured — or if the
    provider is down — a rules-based fallback answers (``mode: "offline"``) instead of an
    error, using the same tools and the same confirm-before-change rule.
    """
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    history = payload.get("history")
    if not message:
        return jsonify({"error": "Please type a question for the Chef."}), 400
    if len(message) > 1000:
        return jsonify({"error": "That message is a bit long — please shorten it."}), 400
    if not isinstance(history, list):
        history = None
    run_id = str(payload.get("run_id") or "")
    sid = str(payload.get("sid") or "")
    stream = bool(_RUN_ID.fullmatch(run_id) and _RUN_ID.fullmatch(sid))

    def on_step(event) -> None:
        if stream:  # to the asking tab only; other tabs get the state_changed hints below
            socketio.emit("agent_step", {"run_id": run_id, **dict(event)}, to=sid)

    try:
        tools = _agent_tools()
        memory = storage.get_memory()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to prepare the agent")
        return jsonify({"error": "Something went wrong while asking the Chef."}), 500

    try:
        if agent.configured():
            try:
                result = agent.run_agent(
                    message, tools=tools, history=history, memory=memory, on_step=on_step
                )
            except (AgentConfigError, AgentAPIError) as exc:
                logger.warning("Agent LLM unavailable, answering offline: %s", exc)
                result = offline_agent.run_offline(
                    message, tools=tools, on_step=on_step,
                    note="My AI brain is unreachable right now, so this is a simpler answer.",
                )
        else:
            result = offline_agent.run_offline(message, tools=tools, on_step=on_step)
    except Exception:  # noqa: BLE001 — never leak a stack trace to the client
        logger.exception("Unexpected error in the agent")
        return jsonify({"error": "Something went wrong while asking the Chef."}), 500

    scopes = list(result.get("changed") or [])
    if result.get("actions"):
        scopes.append("agent")
    if scopes:
        _emit_state(*scopes, source="agent")
    return jsonify(result)


@app.get("/api/agent/actions")
def api_agent_actions():
    """List Chef's proposed changes: ``?status=pending`` (default) or ``all`` (recent 30)."""
    status = (request.args.get("status") or "pending").strip().lower()
    try:
        if status == "all":
            actions = storage.list_actions(limit=30)
        else:
            actions = storage.list_actions(status=storage.ACTION_PENDING, limit=30)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list agent actions")
        return jsonify({"error": "Couldn't load Chef's suggestions right now."}), 500
    return jsonify({"actions": actions})


@app.post("/api/agent/actions/<int:action_id>/confirm")
def api_agent_confirm(action_id: int):
    """Carry out one proposed change — the only way an agent suggestion touches the fridge.

    The stored tool + arguments run through :func:`kitchen.execute` (a fixed whitelist —
    the same code as the manual buttons). Claiming is atomic, so a double-click or two tabs
    confirming at once run it exactly once (the loser gets 409).
    """
    try:
        action = storage.get_action(action_id)
        if action is None:
            return jsonify({"error": "That request was not found."}), 404
        if action.get("status") != storage.ACTION_PENDING:
            return jsonify({"error": "That request was already handled.", "action": action}), 409
        if _is_stale(action):
            storage.resolve_action(action_id, storage.ACTION_EXPIRED)
            _emit_state("agent", source="agent-action")
            return jsonify({"error": "That request is out of date — ask Chef again."}), 409
        claimed = storage.claim_action(action_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load agent action %s", action_id)
        return jsonify({"error": "Couldn't carry that out right now."}), 500
    if claimed is None:
        return jsonify({"error": "That request was already handled."}), 409

    try:
        result = kitchen.execute(claimed["tool"], claimed.get("arguments") or {})
    except kitchen.KitchenError as exc:
        storage.finish_action(action_id, storage.ACTION_FAILED, {"error": exc.message})
        _emit_state("agent", source="agent-action")
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:  # noqa: BLE001
        logger.exception("Agent action %s failed", action_id)
        storage.finish_action(action_id, storage.ACTION_FAILED, {"error": "internal error"})
        _emit_state("agent", source="agent-action")
        return jsonify({"error": "Couldn't carry that out right now."}), 500

    scopes = list(result.pop("scopes", []))
    storage.finish_action(action_id, storage.ACTION_DONE, result)
    _emit_state(*scopes, "agent", source="agent-action")
    if {"inventory", "perishables"} & set(scopes):
        _refresh_briefing("action")
    return jsonify({"action": storage.get_action(action_id), "result": result})


@app.post("/api/agent/actions/<int:action_id>/cancel")
def api_agent_cancel(action_id: int):
    """Drop a proposed change without doing it."""
    try:
        if storage.get_action(action_id) is None:
            return jsonify({"error": "That request was not found."}), 404
        if not storage.resolve_action(action_id, storage.ACTION_CANCELLED):
            return jsonify({"error": "That request was already handled."}), 409
    except Exception:  # noqa: BLE001
        logger.exception("Failed to cancel agent action %s", action_id)
        return jsonify({"error": "Couldn't cancel that right now."}), 500
    _emit_state("agent", source="agent-action")
    return jsonify({"cancelled": True, "id": action_id})


def _is_stale(action) -> bool:
    try:
        created = datetime.strptime(str(action.get("created_at")), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return datetime.now(UTC) - created.replace(tzinfo=UTC) > ACTION_MAX_AGE


@app.get("/api/agent/memory")
def api_agent_memory():
    """What Chef remembers about the household (diet, allergies, dislikes, ...)."""
    try:
        memory = storage.get_memory()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read agent memory")
        return jsonify({"error": "Couldn't load your preferences right now."}), 500
    return jsonify({
        "memory": memory,
        "summary": preferences.summarize(memory),
        "keys": list(preferences.KEYS),
        "diets": list(preferences.DIETS),
    })


@app.post("/api/agent/memory")
def api_agent_remember():
    """Save one preference. Body: ``{"key": str, "value": str}`` (list keys append)."""
    payload = request.get_json(silent=True) or {}
    return _memory_tool("remember_preference", payload)


@app.delete("/api/agent/memory")
def api_agent_forget():
    """Forget ``?key=`` (optionally just one ``&value=`` of a list), or everything if no key."""
    key = (request.args.get("key") or "").strip()
    if not key:
        try:
            cleared = storage.clear_memory()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to clear agent memory")
            return jsonify({"error": "Couldn't update your preferences right now."}), 500
        _emit_state("memory", source="memory")
        return jsonify({"cleared": cleared})
    return _memory_tool("forget_preference", {"key": key, "value": request.args.get("value")})


def _memory_tool(name: str, args):
    """Run Chef's own memory tool, so the form and the chat share one validation path."""
    try:
        result = _agent_tools()[name].run(dict(args))
    except Exception:  # noqa: BLE001
        logger.exception("Memory update failed")
        return jsonify({"error": "Couldn't update your preferences right now."}), 500
    if result.get("error"):
        return jsonify({"error": result["error"]}), 400
    _emit_state("memory", source="memory")
    return jsonify({**result, "memory": storage.get_memory()})


@app.get("/api/shopping")
def api_shopping():
    """The shopping list: open items first, then ticked-off ones."""
    try:
        items = storage.list_shopping()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read the shopping list")
        return jsonify({"error": "Couldn't load your shopping list right now."}), 500
    return jsonify({"items": items})


@app.post("/api/shopping")
def api_shopping_add():
    """Add items. Body: ``{"items": [{"name", "qty"?, "reason"?} | str]}`` or ``{"name"}``.

    Same rules as when Chef adds them: duplicates skipped, household allergens refused.
    """
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    if items is None and payload.get("name"):
        items = [{"name": payload.get("name"), "qty": payload.get("qty")}]
    try:
        result = _agent_tools()["add_to_shopping_list"].run({"items": items})
    except Exception:  # noqa: BLE001
        logger.exception("Failed to add to the shopping list")
        return jsonify({"error": "Couldn't update your shopping list right now."}), 500
    if result.get("error"):
        return jsonify({"error": result["error"]}), 400
    _emit_state("shopping", source="shopping")
    return jsonify(result), 201


@app.patch("/api/shopping/<int:item_id>")
def api_shopping_toggle(item_id: int):
    """Tick an item off (or back on). Body: ``{"done": bool}``."""
    payload = request.get_json(silent=True) or {}
    try:
        found = storage.set_shopping_done(item_id, bool(payload.get("done", True)))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update shopping item %s", item_id)
        return jsonify({"error": "Couldn't update your shopping list right now."}), 500
    if not found:
        return jsonify({"error": "That item was not found."}), 404
    _emit_state("shopping", source="shopping")
    return jsonify({"updated": True, "id": item_id})


@app.delete("/api/shopping/<int:item_id>")
def api_shopping_delete(item_id: int):
    """Remove one item from the list."""
    try:
        removed = storage.delete_shopping_item(item_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete shopping item %s", item_id)
        return jsonify({"error": "Couldn't update your shopping list right now."}), 500
    if not removed:
        return jsonify({"error": "That item was not found."}), 404
    _emit_state("shopping", source="shopping")
    return jsonify({"deleted": True})


@app.delete("/api/shopping")
def api_shopping_clear():
    """Clear the list — or only the ticked-off items with ``?done=1``."""
    done_only = (request.args.get("done") or "").strip() in ("1", "true")
    try:
        cleared = storage.clear_shopping(done_only=done_only)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear the shopping list")
        return jsonify({"error": "Couldn't update your shopping list right now."}), 500
    if cleared:
        _emit_state("shopping", source="shopping")
    return jsonify({"cleared": cleared})


@app.get("/api/agent/briefing")
def api_agent_briefing():
    """The current proactive briefing (``null`` if none / dismissed) with its live actions."""
    try:
        briefing = storage.latest_briefing()
        if briefing:
            pending = {a["id"]: a for a in storage.list_actions(
                status=storage.ACTION_PENDING, limit=30) if a.get("source") == watcher.SOURCE}
            briefing["actions"] = [pending[a["id"]] for a in
                                   (briefing.get("body") or {}).get("actions", [])
                                   if a.get("id") in pending]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to read the briefing")
        return jsonify({"error": "Couldn't load Chef's briefing right now."}), 500
    return jsonify({"briefing": briefing})


@app.post("/api/agent/briefing")
def api_agent_briefing_check():
    """*Check now*: have the watcher look at the fridge right away (always answers)."""
    try:
        watcher.generate_briefing("manual", force=True)
    except Exception:  # noqa: BLE001
        logger.exception("Manual briefing failed")
        return jsonify({"error": "Chef couldn't check the fridge right now."}), 500
    _emit_state("briefing", "agent", source="watcher-manual")
    return api_agent_briefing()


@app.post("/api/agent/briefing/<int:briefing_id>/dismiss")
def api_agent_briefing_dismiss(briefing_id: int):
    """Hide a briefing (its suggested actions expire with it)."""
    try:
        if not storage.dismiss_briefing(briefing_id):
            return jsonify({"error": "That briefing was not found."}), 404
        storage.expire_actions(source=watcher.SOURCE)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to dismiss briefing %s", briefing_id)
        return jsonify({"error": "Couldn't dismiss that right now."}), 500
    _emit_state("briefing", "agent", source="briefing-dismiss")
    return jsonify({"dismissed": True})


@app.get("/api/perishables")
def api_list_perishables():
    """Return the user's tracked perishables (soonest use-by first)."""
    try:
        perishables = storage.list_perishables()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list perishables")
        return jsonify({"error": "Couldn't load your tracked items right now."}), 500
    return jsonify({"perishables": perishables})


@app.post("/api/perishables")
def api_add_perishable():
    """Track a perishable with a "use by" date. Body: ``{"name": str, "use_by": "YYYY-MM-DD"}``."""
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    use_by = str(payload.get("use_by", "")).strip()

    if not name:
        return jsonify({"error": "Please enter what the item is."}), 400
    if len(name) > 80:
        return jsonify({"error": "That name is too long."}), 400
    if not kitchen.valid_use_by(use_by):
        return jsonify({"error": "Please enter a valid use-by date (YYYY-MM-DD)."}), 400

    try:
        perishable_id = storage.add_perishable(name, use_by)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to add perishable")
        return jsonify({"error": "Couldn't save that item right now."}), 500
    _emit_state("perishables", source="perishable-add")
    _refresh_briefing("action")
    return (
        jsonify({"perishable": {"id": perishable_id, "name": name, "use_by": use_by}}),
        201,
    )


@app.delete("/api/perishables/<int:perishable_id>")
def api_delete_perishable(perishable_id: int):
    """Stop tracking one perishable."""
    try:
        removed = storage.delete_perishable(perishable_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to delete perishable %s", perishable_id)
        return jsonify({"error": "Couldn't remove that item right now."}), 500
    if not removed:
        return jsonify({"error": "That item was not found."}), 404
    _emit_state("perishables", source="perishable-delete")
    return jsonify({"deleted": True})


@app.delete("/api/perishables")
def api_clear_perishables():
    """Stop tracking all perishables."""
    try:
        cleared = storage.clear_perishables()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear perishables")
        return jsonify({"error": "Couldn't clear your tracked items right now."}), 500
    if cleared:
        _emit_state("perishables", source="perishable-clear")
    return jsonify({"cleared": cleared})


@app.post("/api/perishables/<int:perishable_id>/resolve")
def api_resolve_perishable(perishable_id: int):
    """Close out a tracked perishable as eaten or thrown away, then stop tracking it.

    Body: ``{"event": "used" | "wasted", "est_cost": number?}``. Logs the outcome to the
    waste log (feeding the Analytics dashboard) and removes the perishable so it no longer
    shows up under "use soon". ``est_cost`` is an optional rough price for spend analytics.
    """
    payload = request.get_json(silent=True) or {}
    est_cost, cost_error = kitchen.parse_est_cost(payload.get("est_cost"))
    if cost_error:
        return jsonify({"error": cost_error}), 400

    try:
        result = kitchen.resolve_perishable(
            perishable_id, str(payload.get("event", "")), est_cost=est_cost
        )
    except kitchen.KitchenError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:  # noqa: BLE001
        logger.exception("Failed to resolve perishable %s", perishable_id)
        return jsonify({"error": "Couldn't update that item right now."}), 500
    # Resolving a tracked item logs a waste/used event and drops it from the at-risk list.
    _emit_state(*result.pop("scopes"), source="perishable-resolve")
    _refresh_briefing("action")
    return jsonify(result)


@app.post("/api/waste")
def api_log_waste():
    """Manually log that an item was used or wasted (not tied to a tracked perishable).

    Body: ``{"name": str, "event": "used" | "wasted", "est_cost": number?}``.
    """
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    event = str(payload.get("event", "")).strip().lower()
    if not name:
        return jsonify({"error": "Please enter what the item is."}), 400
    if len(name) > 80:
        return jsonify({"error": "That name is too long."}), 400
    if event not in (storage.WASTE_USED, storage.WASTE_WASTED):
        return jsonify({"error": "Mark the item as either 'used' or 'wasted'."}), 400
    est_cost, cost_error = kitchen.parse_est_cost(payload.get("est_cost"))
    if cost_error:
        return jsonify({"error": cost_error}), 400

    try:
        event_id = storage.log_waste_event(
            name,
            event,
            token=recommender.normalize_ingredient(name),
            est_cost=est_cost,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to log waste event")
        return jsonify({"error": "Couldn't save that right now."}), 500
    _emit_state("analytics", source="waste")
    return jsonify({"logged": True, "event": event, "id": event_id, "name": name}), 201


@app.get("/api/analytics")
def api_analytics():
    """Summarize the waste log into the numbers behind the Analytics dashboard."""
    try:
        events = storage.list_waste_events()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load waste events")
        return jsonify({"error": "Couldn't load your analytics right now."}), 500
    return jsonify(analytics.summarize(events))


@app.get("/api/restock")
def api_restock():
    """Predict which staples are running low, from usage history + what's on hand.

    Combines four read-only sources — the append-only waste log (long-term usage history),
    the current fridge, tracked perishables, and the open shopping list — and hands them to
    :func:`restock.suggest_restock`, which decides what's worth topping up. Everything the UI
    needs to render the Restock card comes back in one round-trip.
    """
    try:
        events = storage.list_waste_events()
        on_hand = (_current_inventory() or {}).get("items", [])
        perishables = storage.list_perishables()
        on_list = storage.list_shopping(include_done=False)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load restock inputs")
        return jsonify({"error": "Couldn't work out your restock list right now."}), 500
    return jsonify(
        restock.suggest_restock(
            events,
            on_hand,
            perishables=perishables,
            on_list=on_list,
        )
    )


@app.post("/api/cook")
def api_cook():
    """Mark a planned meal cooked: decrement its ingredients from the fridge and log them used.

    Body: ``{"uses": ["Paneer", "Tomato", ...], "title": "Paneer Bhurji"?}`` — typically the
    ``uses`` list of one entry from a :func:`optimizer.plan_meals` plan. (An ``items`` list of
    ``{"name", "qty"?}`` objects is also accepted for explicit quantities.)

    This closes the plan → cook → inventory loop by doing two things:

    * records each ingredient in the **consumption ledger**, so the current-inventory view
      (fridge, recommender, optimizer, nutrition) reflects what's actually left; and
    * logs each ingredient as **used** in the waste log, so the Analytics dashboard shows
      waste trending down as the user follows the plan.

    Returns the updated current inventory so the caller can refresh in one round-trip.
    """
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()[:120]
    try:
        result = kitchen.cook_meal(kitchen.cook_entries(payload), title=title)
    except kitchen.KitchenError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:  # noqa: BLE001 — never leak a stack trace
        logger.exception("Failed to record a cooked meal")
        return jsonify({"error": "Couldn't record that meal right now."}), 500

    try:
        result["scan"] = _current_inventory()
    except Exception:  # noqa: BLE001 — the write succeeded; a read failure shouldn't 500
        logger.exception("Cooked meal recorded but couldn't reload inventory")
        result["scan"] = None

    # Cooking touches three views at once: the fridge shrinks, waste analytics tick up,
    # and any tracked perishables it used up are cleared.
    _emit_state(*result.pop("scopes"), source="cook")
    _refresh_briefing("action")
    return jsonify(result), 201


# Longest receipt text we'll parse in one go — generous for a full grocery bill, but a guard
# against someone pasting a novel.
MAX_RECEIPT_CHARS = 20000


@app.post("/api/receipt/parse")
def api_receipt_parse():
    """Preview a pasted receipt as a list of grocery items. Body: ``{"text": str}``.

    Read-only: nothing is written. Returns :func:`receipts.parse_receipt`'s reviewable
    preview so the shopper can confirm before committing (see ``/api/receipt/import``).
    """
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", ""))
    if not text.strip():
        return jsonify({"error": "Paste the text of your receipt first."}), 400
    if len(text) > MAX_RECEIPT_CHARS:
        return jsonify({"error": "That receipt is very long — paste just the item lines."}), 400
    try:
        preview = receipts.parse_receipt(text)
    except Exception:  # noqa: BLE001 — parsing must never leak a stack trace
        logger.exception("Failed to parse a receipt")
        return jsonify({"error": "Couldn't read that receipt right now."}), 500
    return jsonify(preview)


@app.post("/api/receipt/import")
def api_receipt_import():
    """Fold reviewed receipt items into the fridge. Body: ``{"items": [{"name", "qty"?}]}``.

    Writes a new inventory snapshot (see :func:`kitchen.import_receipt`) and returns the
    updated current inventory so the caller can refresh in one round-trip.
    """
    payload = request.get_json(silent=True) or {}
    try:
        result = kitchen.import_receipt(payload.get("items") or [])
    except kitchen.KitchenError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:  # noqa: BLE001 — never leak a stack trace
        logger.exception("Failed to import a receipt")
        return jsonify({"error": "Couldn't add those items right now."}), 500

    try:
        result["scan"] = _current_inventory()
    except Exception:  # noqa: BLE001 — the write succeeded; a read failure shouldn't 500
        logger.exception("Receipt imported but couldn't reload inventory")
        result["scan"] = None

    _emit_state(*result.pop("scopes"), source="receipt-import")
    _refresh_briefing("action")
    return jsonify(result), 201


@app.post("/api/leftovers")
def api_save_leftovers():
    """Track cooked leftovers as a perishable. Body: ``{"name": str, "days": int?}``."""
    payload = request.get_json(silent=True) or {}
    try:
        result = kitchen.save_leftovers(
            str(payload.get("name", "")), payload.get("days", kitchen.DEFAULT_LEFTOVER_DAYS)
        )
    except kitchen.KitchenError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception:  # noqa: BLE001 — never leak a stack trace
        logger.exception("Failed to save leftovers")
        return jsonify({"error": "Couldn't save those leftovers right now."}), 500

    _emit_state(*result.pop("scopes"), source="leftovers")
    _refresh_briefing("action")
    return jsonify(result), 201


def _current_inventory():
    """The current fridge: the latest scan with any "cooked since" consumption subtracted.

    Every inventory-reading path (fridge view, recommender, optimizer, nutrition, agent)
    goes through this so they all agree on what's *actually* left after meals are cooked.
    The recommender's normalizer is injected so scan item names line up with the canonical
    tokens stored in the consumption ledger.
    """
    return storage.get_current_inventory(normalize=recommender.normalize_ingredient)


def _clamp_int(value, *, default: int, lo: int, hi: int) -> int:
    """Coerce a request value to an int clamped to [lo, hi], else ``default``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clamp_budget(value) -> float | None:
    """Coerce an optional budget to a non-negative float, or ``None`` for 'no ceiling'.

    Blanks, ``None``, and unparseable values all mean "don't constrain the shop". A negative
    number is treated as zero (buy nothing new) rather than rejected.
    """
    if value is None or value == "":
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, n)


@app.errorhandler(413)
def handle_too_large(_error):
    """Turn Flask's raw 413 (body exceeds MAX_CONTENT_LENGTH) into a friendly JSON error."""
    return (
        jsonify({"error": "That image is too large. Please use one under 5 MB."}),
        413,
    )


def _request_source() -> str:
    """Classify how the image arrived, for the history record ('upload' vs 'webcam')."""
    uploaded = request.files.get("image")
    if uploaded is not None and uploaded.filename:
        return storage.SOURCE_UPLOAD
    return storage.SOURCE_WEBCAM


def _extract_image_bytes() -> bytes:
    """Pull raw image bytes from either a multipart upload or a base64 JSON body.

    Raises:
        ImageInputError: If no image is present, it is empty, it is not decodable
            base64, or it exceeds the size limit.
    """
    # Path 1: multipart/form-data file upload.
    uploaded = request.files.get("image")
    if uploaded is not None and uploaded.filename:
        data = uploaded.read()
        _check_size(data)
        if not data:
            raise ImageInputError("The uploaded file was empty.")
        return data

    # Path 2: JSON body with a base64 (optionally data-URL) image, e.g. from the webcam.
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        b64 = payload.get("image_base64")
        if isinstance(b64, str) and b64.strip():
            data = _decode_base64_image(b64)
            _check_size(data)
            return data

    raise ImageInputError("No image was provided. Upload a file or capture one from the webcam.")


def _decode_base64_image(value: str) -> bytes:
    """Decode a base64 string (with or without a ``data:image/...;base64,`` prefix)."""
    stripped = _DATA_URL_PREFIX.sub("", value.strip())
    try:
        data = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError("The captured image could not be decoded.") from exc
    if not data:
        raise ImageInputError("The captured image was empty.")
    return data


def _check_size(data: bytes) -> None:
    """Reject images larger than the configured limit with a 413."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError(
            "That image is too large. Please use one under 5 MB.", status_code=413
        )


def _prepare_image(data: bytes) -> bytes:
    """Validate the image and downscale/re-encode it to a reasonable JPEG.

    Any image whose longest side exceeds :data:`MAX_DIMENSION` is scaled down to cut
    payload size and latency. All images are re-encoded to JPEG so the vision service
    always receives a clean, known format.

    Raises:
        ImageInputError: If the bytes are not a decodable image.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError(
            "That file doesn't look like a valid image. Please try a JPEG or PNG."
        ) from exc

    # Flatten transparency / palette modes onto white so JPEG encoding is safe.
    if image.mode not in ("RGB", "L"):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

    if max(image.size) > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


if __name__ == "__main__":
    # Convenience for `python app.py`. Use ``socketio.run`` (not ``app.run``) so the
    # Socket.IO/WebSocket transport is served alongside the HTTP routes.
    # ``allow_unsafe_werkzeug`` opts the dev server in for local use — production runs
    # under gunicorn (see the Dockerfile).
    socketio.run(app, host="127.0.0.1", port=5000, debug=True, allow_unsafe_werkzeug=True)
