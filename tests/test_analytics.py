"""Unit tests for the framework-agnostic waste & spend analytics (:mod:`analytics`).

These exercise :func:`analytics.summarize` directly on plain event lists — no Flask, no
database, no network. A fixed ``now`` makes the weekly-trend window deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime

import analytics

# A fixed "now": Wednesday 2026-09-23. Its week starts Monday 2026-09-21.
NOW = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)


def _event(name, event, *, item=None, est_cost=None, logged_at="2026-09-22T10:00:00Z"):
    return {
        "name": name,
        "item": item if item is not None else name.lower(),
        "event": event,
        "est_cost": est_cost,
        "logged_at": logged_at,
    }


def test_empty_log():
    out = analytics.summarize([], now=NOW)
    assert out["has_data"] is False
    assert out["totals"] == {"used": 0, "wasted": 0, "events": 0}
    assert out["waste_rate"] == 0.0
    assert out["money"] == {"wasted": 0.0, "saved": 0.0, "has_cost": False}
    assert out["top_wasted"] == []
    # The trend is always zero-filled to the requested window (default 6 weeks).
    assert len(out["by_week"]) == 6
    assert all(w["used"] == 0 and w["wasted"] == 0 for w in out["by_week"])
    assert "No items logged" in out["headline"]


def test_counts_and_waste_rate():
    events = [
        _event("Milk", "used"),
        _event("Spinach", "wasted"),
        _event("Bread", "wasted"),
        _event("Eggs", "used"),
        _event("weird", "ignored-outcome"),  # not used/wasted -> skipped
    ]
    out = analytics.summarize(events, now=NOW)
    assert out["totals"] == {"used": 2, "wasted": 2, "events": 4}
    assert out["waste_rate"] == 50.0
    assert out["has_data"] is True


def test_money_wasted_and_saved():
    events = [
        _event("Milk", "used", est_cost=3.0),
        _event("Spinach", "wasted", est_cost=2.5),
        _event("Bread", "wasted", est_cost=1.25),
        _event("Eggs", "used"),  # no cost -> doesn't affect money totals
    ]
    out = analytics.summarize(events, now=NOW)
    assert out["money"]["has_cost"] is True
    assert out["money"]["wasted"] == 3.75
    assert out["money"]["saved"] == 3.0


def test_no_cost_leaves_money_flag_false():
    out = analytics.summarize([_event("Milk", "wasted")], now=NOW)
    assert out["money"] == {"wasted": 0.0, "saved": 0.0, "has_cost": False}


def test_top_wasted_grouping_and_order():
    events = [
        _event("Spinach", "wasted", est_cost=2.0),
        _event("Spinach", "wasted", est_cost=2.0),
        _event("Bread", "wasted", est_cost=1.0),
        _event("Milk", "used"),  # used items never appear in top_wasted
    ]
    out = analytics.summarize(events, now=NOW)
    top = out["top_wasted"]
    assert [t["name"] for t in top] == ["Spinach", "Bread"]
    assert top[0]["wasted"] == 2
    assert top[0]["cost"] == 4.0
    assert all(t["wasted"] > 0 for t in top)


def test_top_n_limits_breakdown():
    events = [_event(f"item{i}", "wasted") for i in range(10)]
    out = analytics.summarize(events, now=NOW, top_n=3)
    assert len(out["top_wasted"]) == 3


def test_weekly_trend_buckets_by_monday():
    events = [
        _event("A", "wasted", logged_at="2026-09-21T09:00:00Z"),  # this week (Mon 09-21)
        _event("B", "used", logged_at="2026-09-23T09:00:00Z"),    # this week too
        _event("C", "used", logged_at="2026-09-15T09:00:00Z"),    # week of Mon 09-14
    ]
    out = analytics.summarize(events, now=NOW, recent_weeks=6)
    by_week = {w["week_start"]: w for w in out["by_week"]}
    assert by_week["2026-09-21"] == {"week_start": "2026-09-21", "used": 1, "wasted": 1}
    assert by_week["2026-09-14"] == {"week_start": "2026-09-14", "used": 1, "wasted": 0}
    # Weeks with nothing logged are still present as zeros.
    assert by_week["2026-08-31"] == {"week_start": "2026-08-31", "used": 0, "wasted": 0}


def test_events_outside_window_still_count_totals_but_not_trend():
    # An event older than the window contributes to totals but not to the (6-week) trend.
    old = _event("Old", "wasted", logged_at="2026-01-01T09:00:00Z")
    out = analytics.summarize([old], now=NOW, recent_weeks=6)
    assert out["totals"]["wasted"] == 1
    assert all(w["wasted"] == 0 for w in out["by_week"])


def test_name_falls_back_to_prettified_token():
    # A row with only a token (no display name) still gets a readable name.
    events = [{"item": "bell_pepper", "event": "wasted", "logged_at": "2026-09-22T09:00:00Z"}]
    out = analytics.summarize(events, now=NOW)
    # recommender._prettify: underscores become spaces, first letter capitalized.
    assert out["top_wasted"][0]["name"] == "Bell pepper"


def test_unparseable_timestamp_skips_trend_only():
    events = [{"name": "Milk", "item": "milk", "event": "wasted", "logged_at": "garbage"}]
    out = analytics.summarize(events, now=NOW)
    assert out["totals"]["wasted"] == 1  # still counted
    assert all(w["wasted"] == 0 for w in out["by_week"])  # but not placed in any week


def test_headline_mentions_cost_when_present():
    events = [_event("Spinach", "wasted", est_cost=4.0)]
    out = analytics.summarize(events, now=NOW)
    assert "4.00" in out["headline"]
