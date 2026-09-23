"""Waste & spend analytics (Phase 2 — turning the waste log into insight).

Every time the user resolves a tracked item as eaten (``used``) or thrown away
(``wasted``), the web layer appends a row to the waste log (see
:func:`storage.log_waste_event`). This module turns that raw log into the numbers the
Analytics dashboard shows: how much food is wasted vs used, the money lost and rescued,
which ingredients are wasted most, and the trend over recent weeks.

Like :mod:`optimizer`, :mod:`recommender`, :mod:`nutrition`, and :mod:`storage`, this
module is **framework-agnostic**: it takes the event rows as a plain list (the web layer
reads them from storage and passes them in) and returns plain dicts, so it needs no Flask
and no database and is trivially testable. Pass ``now`` for deterministic output in tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import recommender

# The two outcomes the log records; mirrors storage.WASTE_USED / WASTE_WASTED.
USED = "used"
WASTED = "wasted"


def summarize(
    events: Iterable[Mapping[str, Any]],
    *,
    now: datetime | None = None,
    recent_weeks: int = 6,
    top_n: int = 5,
) -> dict[str, Any]:
    """Aggregate raw waste-log rows into the analytics dashboard payload.

    Args:
        events: Rows like :func:`storage.list_waste_events` returns —
            ``{item, name, event, est_cost, logged_at}`` (any order; extras ignored).
        now: "Current" time (UTC) for the weekly trend window; defaults to real now.
        recent_weeks: How many trailing weeks to include in the trend (Monday-started).
        top_n: How many of the most-wasted ingredients to list.

    Returns:
        A dict with ``totals`` (used/wasted/events), ``waste_rate`` (percent of resolved
        items thrown away), ``money`` (wasted vs rescued, and whether any cost was given),
        ``top_wasted`` (ingredient breakdown), ``by_week`` (trend), ``headline`` (a plain
        one-line summary), and ``has_data``.
    """
    now = now or datetime.now(UTC)

    used = 0
    wasted = 0
    money_wasted = 0.0
    money_saved = 0.0
    have_cost = False
    # token -> aggregate for the "most wasted" breakdown.
    per_item: dict[str, dict[str, Any]] = {}
    # Monday date (YYYY-MM-DD) -> {used, wasted} for the trend.
    weekly: dict[str, dict[str, int]] = {}

    for event in events:
        outcome = str(event.get("event", "")).strip().lower()
        if outcome not in (USED, WASTED):
            continue
        name = _clean(event.get("name")) or recommender._prettify(str(event.get("item", "")))
        token = _clean(event.get("item")) or (name or "").strip().lower()
        cost = _as_cost(event.get("est_cost"))

        bucket = per_item.setdefault(
            token, {"token": token, "name": name or token, "used": 0, "wasted": 0, "cost": 0.0}
        )
        bucket[outcome] += 1

        if outcome == WASTED:
            wasted += 1
            if cost is not None:
                money_wasted += cost
                bucket["cost"] += cost
                have_cost = True
        else:
            used += 1
            if cost is not None:
                money_saved += cost
                have_cost = True

        # Weekly trend bucket (skip rows with an unparseable timestamp).
        week = _week_start(event.get("logged_at"))
        if week is not None:
            wb = weekly.setdefault(week, {"used": 0, "wasted": 0})
            wb[outcome] += 1

    resolved = used + wasted
    waste_rate = round(wasted / resolved * 100, 1) if resolved else 0.0

    # Most-wasted ingredients: only those actually wasted, worst first (count, then cost).
    top_wasted = sorted(
        (b for b in per_item.values() if b["wasted"] > 0),
        key=lambda b: (b["wasted"], b["cost"]),
        reverse=True,
    )[:top_n]
    top_wasted = [
        {"name": b["name"], "token": b["token"], "wasted": b["wasted"], "cost": round(b["cost"], 2)}
        for b in top_wasted
    ]

    by_week = _trend(weekly, now, recent_weeks)

    return {
        "totals": {"used": used, "wasted": wasted, "events": resolved},
        "waste_rate": waste_rate,
        "money": {
            "wasted": round(money_wasted, 2),
            "saved": round(money_saved, 2),
            "has_cost": have_cost,
        },
        "top_wasted": top_wasted,
        "by_week": by_week,
        "headline": _headline(used, wasted, waste_rate, money_wasted, have_cost),
        "has_data": resolved > 0,
    }


# --- Helpers ----------------------------------------------------------------


def _trend(
    weekly: Mapping[str, Mapping[str, int]], now: datetime, recent_weeks: int
) -> list[dict[str, Any]]:
    """Build a zero-filled, oldest→newest list of the last ``recent_weeks`` weeks."""
    weeks = max(1, int(recent_weeks))
    this_monday = _monday(now.date())
    out: list[dict[str, Any]] = []
    for i in range(weeks - 1, -1, -1):
        monday = this_monday - timedelta(days=7 * i)
        key = monday.isoformat()
        counts = weekly.get(key, {})
        out.append(
            {
                "week_start": key,
                "used": int(counts.get("used", 0)),
                "wasted": int(counts.get("wasted", 0)),
            }
        )
    return out


def _headline(
    used: int, wasted: int, waste_rate: float, money_wasted: float, have_cost: bool
) -> str:
    """A plain-language one-liner summarizing the log."""
    resolved = used + wasted
    if resolved == 0:
        return "No items logged yet — mark tracked items as used or wasted to see your trends."
    noun = "item" if wasted == 1 else "items"
    base = f"You've thrown away {wasted} of {resolved} tracked {noun} ({waste_rate}%)."
    if have_cost and money_wasted > 0:
        return f"{base} That's about {money_wasted:.2f} in food lost."
    return base


def _week_start(value: Any) -> str | None:
    """Return the Monday (YYYY-MM-DD) of the week containing ``logged_at``, or None."""
    dt = _parse_dt(value)
    if dt is None:
        return None
    return _monday(dt.date()).isoformat()


def _monday(day):
    """The Monday on or before ``day`` (a date)."""
    return day - timedelta(days=day.weekday())


def _parse_dt(value: Any) -> datetime | None:
    """Parse a stored ``YYYY-MM-DDTHH:MM:SSZ`` timestamp (tolerant), or None."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        # Fall back to a bare date.
        try:
            return datetime.fromisoformat(str(value).strip()[:10])
        except ValueError:
            return None


def _clean(value: Any) -> str | None:
    """Return a stripped string, or None for missing/empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_cost(value: Any) -> float | None:
    """Coerce to a non-negative float cost, or None for missing/invalid values."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None
