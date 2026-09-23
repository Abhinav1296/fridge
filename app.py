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
* ``POST /api/agent`` answers a natural-language question with the tool-using Chef agent
  (:mod:`agent`), which calls the inventory / recommender / optimizer functions itself.
* ``GET  /api/nutrition`` returns the nutrition dashboard for the current fridge (cached
  per-ingredient calories/macros plus per-100 g totals and coverage).
* ``POST /api/nutrition/refresh`` fills the nutrition cache for the current inventory from
  Open Food Facts (:mod:`nutrition`).
* ``POST /api/barcode`` looks up a scanned packaged product by barcode (Open Food Facts)
  and caches its nutrition.
* ``/api/perishables`` (GET/POST/DELETE) manages the user's tracked "use by" dates.

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
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import agent
import analytics
import nutrition
import optimizer
import recommender
import storage
import vision_service
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
    "use by" dates, and runs the embedding-based recommender over both. Returns empty
    lists (not an error) when there's nothing to suggest yet.
    """
    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
    except Exception:  # noqa: BLE001 — a read failure shouldn't 500 the whole page
        logger.exception("Failed to load data for recommendations")
        return jsonify({"error": "Couldn't load suggestions right now."}), 500

    items = (scan or {}).get("items", [])
    try:
        result = recommender.recommend(items, perishables)
    except Exception:  # noqa: BLE001 — recommender must never take the page down
        logger.exception("Recommender failed")
        return jsonify({"error": "Couldn't build suggestions right now."}), 500
    return jsonify(result)


@app.post("/api/plan")
def api_plan():
    """Return a zero-waste meal plan for the current fridge.

    Body (all optional): ``{"days": int (1-7), "meals_per_day": int (1-4)}``. Combines the
    latest scan, tracked "use by" dates, and the cached nutrition table, then runs
    :func:`optimizer.plan_meals` (exact ILP when PuLP is available, else the built-in
    heuristic). Returns an empty plan (not an error) when there's nothing to cook yet.
    """
    payload = request.get_json(silent=True) or {}
    days = _clamp_int(payload.get("days"), default=3, lo=1, hi=7)
    meals_per_day = _clamp_int(payload.get("meals_per_day"), default=2, lo=1, hi=4)

    try:
        scan = _current_inventory()
        perishables = storage.list_perishables()
        nutrition = {row["item"]: row for row in storage.list_nutrition() if row.get("item")}
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
        )
    except Exception:  # noqa: BLE001 — the planner must never take the page down
        logger.exception("Meal-plan optimizer failed")
        return jsonify({"error": "Couldn't build a meal plan right now."}), 500
    return jsonify(plan)


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


@app.post("/api/agent")
def api_agent():
    """Answer a natural-language question with the tool-using Chef agent.

    Body: ``{"message": str, "history": [{"role": "user"|"assistant", "content": str}]?}``.
    The agent calls the inventory / recommender / optimizer tools itself and returns
    ``{"answer", "steps", "mode", "model"}``. Returns 503 when no chat model is configured.
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

    if not agent.configured():
        return (
            jsonify(
                {
                    "error": "The Chef isn't configured on the server. Add LLM_API_KEY "
                    "(or VISION_API_KEY) to your .env to enable it."
                }
            ),
            503,
        )

    tools = agent.build_default_tools(
        get_latest_scan=_current_inventory,
        list_perishables=storage.list_perishables,
        list_nutrition=storage.list_nutrition,
    )
    try:
        result = agent.run_agent(message, tools=tools, history=history)
    except AgentConfigError as exc:
        logger.error("Agent misconfigured: %s", exc)
        return jsonify({"error": "The Chef isn't configured on the server."}), 503
    except AgentAPIError as exc:
        logger.warning("Agent API error: %s", exc)
        return jsonify({"error": "The Chef couldn't reach its language model. Try again."}), 502
    except Exception:  # noqa: BLE001 — never leak a stack trace to the client
        logger.exception("Unexpected error in the agent")
        return jsonify({"error": "Something went wrong while asking the Chef."}), 500
    return jsonify(result)


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
    if not _valid_use_by(use_by):
        return jsonify({"error": "Please enter a valid use-by date (YYYY-MM-DD)."}), 400

    try:
        perishable_id = storage.add_perishable(name, use_by)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to add perishable")
        return jsonify({"error": "Couldn't save that item right now."}), 500
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
    return jsonify({"deleted": True})


@app.delete("/api/perishables")
def api_clear_perishables():
    """Stop tracking all perishables."""
    try:
        cleared = storage.clear_perishables()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear perishables")
        return jsonify({"error": "Couldn't clear your tracked items right now."}), 500
    return jsonify({"cleared": cleared})


@app.post("/api/perishables/<int:perishable_id>/resolve")
def api_resolve_perishable(perishable_id: int):
    """Close out a tracked perishable as eaten or thrown away, then stop tracking it.

    Body: ``{"event": "used" | "wasted", "est_cost": number?}``. Logs the outcome to the
    waste log (feeding the Analytics dashboard) and removes the perishable so it no longer
    shows up under "use soon". ``est_cost`` is an optional rough price for spend analytics.
    """
    payload = request.get_json(silent=True) or {}
    event = str(payload.get("event", "")).strip().lower()
    if event not in (storage.WASTE_USED, storage.WASTE_WASTED):
        return jsonify({"error": "Mark the item as either 'used' or 'wasted'."}), 400
    est_cost, cost_error = _parse_est_cost(payload.get("est_cost"))
    if cost_error:
        return jsonify({"error": cost_error}), 400

    try:
        item = storage.get_perishable(perishable_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load perishable %s", perishable_id)
        return jsonify({"error": "Couldn't update that item right now."}), 500
    if item is None:
        return jsonify({"error": "That item was not found."}), 404

    name = str(item.get("name", "")).strip()
    try:
        event_id = storage.log_waste_event(
            name,
            event,
            token=recommender.normalize_ingredient(name),
            est_cost=est_cost,
        )
        storage.delete_perishable(perishable_id)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to resolve perishable %s", perishable_id)
        return jsonify({"error": "Couldn't update that item right now."}), 500
    return jsonify({"resolved": True, "event": event, "id": event_id, "name": name})


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
    est_cost, cost_error = _parse_est_cost(payload.get("est_cost"))
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

    entries = _cook_entries(payload)
    if not entries:
        return jsonify({"error": "Tell me which ingredients were used to cook this."}), 400
    if len(entries) > 50:
        return jsonify({"error": "That's a lot of ingredients — please cook fewer at once."}), 400

    note = f"Cooked: {title}" if title else "Cooked a planned meal"
    # How many units of each canonical ingredient this meal consumed — used to clear the
    # right number of tracked perishables (not every entry that happens to share a name).
    consumed_qty: dict[str, int] = {}
    for e in entries:
        token = recommender.normalize_ingredient(e["name"])
        consumed_qty[token] = consumed_qty.get(token, 0) + e["qty"]
    cleared_perishables = 0
    try:
        # Ledger rows drive the inventory decrement...
        recorded = storage.record_consumption(
            [
                {
                    "name": e["name"],
                    "token": recommender.normalize_ingredient(e["name"]),
                    "qty": e["qty"],
                }
                for e in entries
            ],
            note=note,
        )
        # ...and a "used" event per ingredient feeds the waste/spend analytics.
        for e in entries:
            storage.log_waste_event(
                e["name"],
                storage.WASTE_USED,
                token=recommender.normalize_ingredient(e["name"]),
            )
        # Cooking a meal also clears the tracked perishables it used up, so the at-risk
        # list and future plans stop nagging about an ingredient that's now eaten. Clear
        # only as many as were consumed (soonest use-by first, the order list_perishables
        # returns), so tracking two blocks of cheese and cooking one meal clears one. The
        # "used" event was already logged above, so delete without re-logging (which would
        # double-count it in the analytics).
        remaining = dict(consumed_qty)
        for perishable in storage.list_perishables():
            token = recommender.normalize_ingredient(perishable.get("name", ""))
            if remaining.get(token, 0) > 0 and storage.delete_perishable(perishable["id"]):
                remaining[token] -= 1
                cleared_perishables += 1
    except Exception:  # noqa: BLE001 — never leak a stack trace
        logger.exception("Failed to record a cooked meal")
        return jsonify({"error": "Couldn't record that meal right now."}), 500

    try:
        scan = _current_inventory()
    except Exception:  # noqa: BLE001 — the write succeeded; a read failure shouldn't 500
        logger.exception("Cooked meal recorded but couldn't reload inventory")
        scan = None

    return (
        jsonify(
            {
                "cooked": True,
                "title": title or None,
                "consumed": [e["name"] for e in entries],
                "recorded": recorded,
                "cleared_perishables": cleared_perishables,
                "scan": scan,
            }
        ),
        201,
    )


def _cook_entries(payload: dict) -> list[dict]:
    """Normalize a cook request body into a de-duplicated ``[{name, qty}]`` list.

    Accepts either ``uses`` (a list of ingredient names, one unit each) or ``items`` (a list
    of ``{"name", "qty"}`` objects). Blank names and names over 80 chars are dropped; repeated
    names are merged, summing their quantities.
    """
    raw: list[tuple[str, int]] = []
    for name in payload.get("uses") or []:
        text = str(name).strip()
        if text:
            raw.append((text, 1))
    for obj in payload.get("items") or []:
        if not isinstance(obj, dict):
            continue
        text = str(obj.get("name", "")).strip()
        if not text:
            continue
        try:
            qty = int(obj.get("qty", 1))
        except (TypeError, ValueError):
            qty = 1
        raw.append((text, max(1, qty)))

    merged: dict[str, dict] = {}
    for name, qty in raw:
        if len(name) > 80:
            continue
        key = name.lower()
        if key in merged:
            merged[key]["qty"] += qty
        else:
            merged[key] = {"name": name, "qty": qty}
    return list(merged.values())


def _current_inventory():
    """The current fridge: the latest scan with any "cooked since" consumption subtracted.

    Every inventory-reading path (fridge view, recommender, optimizer, nutrition, agent)
    goes through this so they all agree on what's *actually* left after meals are cooked.
    The recommender's normalizer is injected so scan item names line up with the canonical
    tokens stored in the consumption ledger.
    """
    return storage.get_current_inventory(normalize=recommender.normalize_ingredient)


def _parse_est_cost(value) -> tuple[float | None, str | None]:
    """Validate an optional cost from a request body.

    Returns ``(cost, None)`` on success (cost may be ``None`` when omitted/blank), or
    ``(None, message)`` when the value is present but not a non-negative number.
    """
    if value is None or value == "":
        return None, None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None, "Enter a cost as a number, or leave it blank."
    if cost < 0:
        return None, "Cost can't be negative."
    return cost, None


def _valid_use_by(value: str) -> bool:
    """True if ``value`` is a real calendar date in ``YYYY-MM-DD`` form."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _clamp_int(value, *, default: int, lo: int, hi: int) -> int:
    """Coerce a request value to an int clamped to [lo, hi], else ``default``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


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
    # Convenience for `python app.py`. `flask run` is the documented entry point.
    app.run(host="127.0.0.1", port=5000, debug=True)
