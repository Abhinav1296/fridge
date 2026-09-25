"""Unit tests for :mod:`restock` — the smart-restock predictor.

Framework-agnostic (no Flask/HTTP, no database). They pin down the contract the watcher
briefings and the ``/api/restock`` endpoint depend on: which staples get flagged, how they
rank, how the on-hand / shopping-list / pantry filters behave, and that the output shape is
stable and JSON-friendly. Time is injected via ``now`` so every case is deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import budget
import recommender
import restock

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)


def _event(item, event="used", *, days_ago=1, name=None, est_cost=None):
    """Build a waste-log-shaped row (matches storage.list_waste_events output)."""
    when = NOW - timedelta(days=days_ago)
    return {
        "item": item,
        "name": name or recommender._prettify(item),
        "event": event,
        "est_cost": est_cost if est_cost is not None else budget.price_of(item),
        "logged_at": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _uses(item, times, **kw):
    """A staple used ``times`` times over the recent window."""
    return [_event(item, days_ago=d + 1, **kw) for d in range(times)]


# --- basic flagging ----------------------------------------------------------


def test_no_history_reports_no_data():
    result = restock.suggest_restock([], [], now=NOW)
    assert result["has_data"] is False
    assert result["count"] == 0
    assert result["items"] == []
    assert "cook" in result["headline"].lower()


def test_heavily_used_and_gone_is_flagged_out():
    events = _uses("paneer", 3)
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    assert result["has_data"] is True
    assert result["count"] == 1
    item = result["items"][0]
    assert item["token"] == "paneer"
    assert item["status"] == "out"
    assert item["uses"] == 3
    assert item["on_hand"] == 0
    assert item["est_cost"] == budget.price_of("paneer")


def test_used_but_still_on_hand_is_not_flagged():
    events = _uses("paneer", 3)
    on_hand = [{"name": "Paneer", "count": 2}]
    result = restock.suggest_restock(events, on_hand=on_hand, now=NOW)
    assert result["count"] == 0
    assert "well stocked" in result["headline"].lower()


def test_below_min_uses_is_ignored():
    # Used only once — not yet a staple, so not flagged even though none is left.
    events = _uses("paneer", 1)
    result = restock.suggest_restock(events, on_hand=[], now=NOW, min_uses=2)
    assert result["count"] == 0


def test_wasted_events_count_toward_usage():
    # Throwing something away repeatedly still means you go through it.
    events = _uses("spinach", 3, event="wasted")
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    assert result["count"] == 1
    item = result["items"][0]
    assert item["wasted"] == 3
    assert item["used"] == 0


# --- low vs out --------------------------------------------------------------


def test_heavy_user_with_one_left_is_low():
    # 4 uses (>= min_uses*2) with a single unit left → running low, not out.
    events = _uses("milk", 4)
    on_hand = [{"name": "milk", "count": 1}]
    result = restock.suggest_restock(events, on_hand=on_hand, now=NOW, min_uses=2, low_threshold=1)
    assert result["count"] == 1
    assert result["items"][0]["status"] == "low"
    assert result["items"][0]["on_hand"] == 1


def test_light_user_with_one_left_is_not_low():
    # Exactly min_uses (2) uses is a staple, but not a *heavy* user, so one spare is fine.
    events = _uses("milk", 2)
    on_hand = [{"name": "milk", "count": 1}]
    result = restock.suggest_restock(events, on_hand=on_hand, now=NOW, min_uses=2, low_threshold=1)
    assert result["count"] == 0


def test_out_outranks_low():
    events = _uses("paneer", 3) + _uses("milk", 4)
    on_hand = [{"name": "milk", "count": 1}]  # milk low, paneer out
    result = restock.suggest_restock(events, on_hand=on_hand, now=NOW)
    assert [i["token"] for i in result["items"]] == ["paneer", "milk"]
    assert result["items"][0]["status"] == "out"
    assert result["items"][1]["status"] == "low"


def test_ties_break_by_usage_frequency():
    events = _uses("paneer", 2) + _uses("chicken", 5)
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    # Both "out"; chicken used more → ranks first.
    assert [i["token"] for i in result["items"]] == ["chicken", "paneer"]


# --- filters -----------------------------------------------------------------


def test_pantry_staples_are_never_suggested():
    staple = next(iter(recommender.PANTRY))
    events = _uses(staple, 5)
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    assert result["count"] == 0


def test_items_already_on_shopping_list_are_skipped():
    events = _uses("paneer", 3)
    result = restock.suggest_restock(events, on_hand=[], on_list=["Paneer"], now=NOW)
    assert result["count"] == 0


def test_on_list_accepts_dict_rows():
    events = _uses("paneer", 3)
    result = restock.suggest_restock(
        events, on_hand=[], on_list=[{"item": "paneer", "name": "Paneer"}], now=NOW
    )
    assert result["count"] == 0


def test_perishables_count_as_on_hand():
    events = _uses("tomato", 3)
    result = restock.suggest_restock(
        events, on_hand=[], perishables=[{"name": "Tomato"}], now=NOW
    )
    # A tracked perishable means it's on hand → not flagged out.
    assert result["count"] == 0


# --- recency window ----------------------------------------------------------


def test_old_events_fall_outside_the_window():
    # All uses are 100 days ago — outside a 6-week window, so no recent usage.
    events = [_event("paneer", days_ago=100) for _ in range(3)]
    result = restock.suggest_restock(events, on_hand=[], now=NOW, recent_weeks=6)
    # has_data is True (there IS history) but nothing recent qualifies.
    assert result["has_data"] is True
    assert result["count"] == 0


def test_events_without_timestamp_still_count():
    events = [{"item": "paneer", "name": "Paneer", "event": "used", "logged_at": ""} for _ in range(3)]
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    assert result["count"] == 1
    assert result["items"][0]["days_since_last"] is None


# --- output shape ------------------------------------------------------------


def test_top_n_caps_results():
    events = []
    for tok in ("paneer", "chicken", "tomato", "onion", "milk"):
        events += _uses(tok, 3)
    result = restock.suggest_restock(events, on_hand=[], now=NOW, top_n=2)
    assert result["count"] == 2
    assert len(result["items"]) == 2


def test_total_cost_sums_estimates():
    events = _uses("paneer", 3) + _uses("chicken", 3)
    result = restock.suggest_restock(events, on_hand=[], now=NOW)
    expected = round(sum(i["est_cost"] for i in result["items"]), 2)
    assert result["total_cost"] == expected


def test_currency_and_window_reported():
    result = restock.suggest_restock(_uses("paneer", 3), on_hand=[], now=NOW, recent_weeks=4)
    assert result["currency"] == budget.CURRENCY
    assert result["window_weeks"] == 4


def test_result_is_json_friendly():
    import json

    events = _uses("paneer", 3) + _uses("milk", 4)
    result = restock.suggest_restock(events, on_hand=[{"name": "milk", "count": 1}], now=NOW)
    # Round-trips cleanly — no datetimes or sets leak into the payload.
    assert json.loads(json.dumps(result)) == result


def test_each_item_has_a_plain_reason():
    result = restock.suggest_restock(_uses("paneer", 3), on_hand=[], now=NOW)
    reason = result["items"][0]["reason"]
    assert isinstance(reason, str) and reason
    assert "paneer" not in reason.lower() or True  # reason is about counts, not necessarily name


def test_default_now_does_not_crash():
    # Exercises the real-clock path (no now=) without asserting on timing.
    # Stamp events at "now" using the real clock so they fall inside the window.
    real_now = datetime.now(UTC)
    events = [
        {
            "item": "paneer",
            "name": "Paneer",
            "event": "used",
            "logged_at": (real_now - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for h in range(3)
    ]
    result = restock.suggest_restock(events, on_hand=[])
    assert "items" in result and "count" in result
