"""Smart restock — predict the staples you're about to run out of.

Every time a tracked item is resolved as eaten (``used``) or thrown away (``wasted``), the
web layer appends a row to the waste log (see :func:`storage.log_waste_event`). Over time
that log becomes a record of *how often you actually go through each ingredient*. This
module reads that history, compares it with what's on hand right now, and works out which
staples are worth putting back on the shopping list — before you notice they're gone.

The rule is deliberately simple and explainable (no opaque model):

* an ingredient you've gone through at least ``min_uses`` times in the recent window is a
  **staple**;
* if none is left on hand it's **out** (top priority); if only a little is left and you
  use it heavily it's **running low**;
* anything already on the shopping list, or a pantry staple that's always assumed on hand,
  is skipped.

Like :mod:`analytics`, :mod:`optimizer`, :mod:`budget` and :mod:`recommender`, this module
is **framework-agnostic**: it takes plain rows and returns plain dicts, so it needs no
Flask and no database and is trivially testable. Pass ``now`` for deterministic output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import budget
import recommender

# Mirrors storage.WASTE_USED / WASTE_WASTED — both mean the item left the fridge, so both
# count towards "how fast you go through it".
USED = "used"
WASTED = "wasted"

# Priority weight per status, so "out of it entirely" always outranks "running low".
_STATUS_WEIGHT = {"out": 2, "low": 1}


def suggest_restock(
    events: Iterable[Mapping[str, Any]],
    on_hand: Iterable[Mapping[str, Any]] | None = None,
    *,
    perishables: Iterable[Mapping[str, Any]] = (),
    on_list: Iterable[Any] = (),
    now: datetime | None = None,
    recent_weeks: int = 6,
    min_uses: int = 2,
    low_threshold: int = 1,
    top_n: int = 8,
) -> dict[str, Any]:
    """Work out which staples to restock from usage history + what's on hand.

    Args:
        events: Waste-log rows like :func:`storage.list_waste_events` returns —
            ``{item, name, event, est_cost, logged_at}`` (any order; extras ignored).
        on_hand: Current inventory item dicts (``{name, count}``) — typically
            ``storage.get_current_inventory()["items"]``. Counts default to 1.
        perishables: Tracked-perishable rows (``{name}``); each counts as one unit on hand,
            so an item the user is actively tracking isn't suggested for restock.
        on_list: Ingredients already on the shopping list (names or tokens) — skipped.
        now: "Current" time (UTC) for the recency window; defaults to real now.
        recent_weeks: Trailing weeks of history that count toward a staple's usage.
        min_uses: Times gone through in the window before an item counts as a staple.
        low_threshold: On-hand units at or below which a heavily-used staple is "running low".
        top_n: Maximum suggestions to return.

    Returns:
        A dict with ``items`` (ranked suggestions), ``count``, ``total_cost`` (sum of the
        estimates), ``currency``, ``window_weeks``, ``has_data`` (any usage history at all),
        and ``headline`` (a plain one-line summary).
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(weeks=max(1, int(recent_weeks)))
    min_uses = max(1, int(min_uses))
    low_threshold = max(0, int(low_threshold))

    have = _on_hand_totals(on_hand, perishables)
    listed = {
        recommender.normalize_ingredient(str(x if not isinstance(x, Mapping) else x.get("item") or x.get("name") or ""))
        for x in on_list
    }
    listed.discard("")

    # token -> usage aggregate over the recent window.
    usage: dict[str, dict[str, Any]] = {}
    any_history = False
    for event in events or []:
        outcome = str(event.get("event", "")).strip().lower()
        if outcome not in (USED, WASTED):
            continue
        any_history = True
        when = _parse_ts(event.get("logged_at"))
        if when is not None and when < cutoff:
            continue  # too old to count toward "recent" usage
        name = _clean(event.get("name")) or recommender._prettify(str(event.get("item", "")))
        token = _clean(event.get("item")) or recommender.normalize_ingredient(name)
        if not token:
            continue
        bucket = usage.setdefault(
            token,
            {"token": token, "name": name or token, "used": 0, "wasted": 0, "uses": 0,
             "last_seen": None},
        )
        bucket[outcome] += 1
        bucket["uses"] += 1
        if when is not None and (bucket["last_seen"] is None or when > bucket["last_seen"]):
            bucket["last_seen"] = when
        # Prefer a real display name if a later row has one.
        if name and (not bucket["name"] or bucket["name"] == token):
            bucket["name"] = name

    suggestions: list[dict[str, Any]] = []
    for token, agg in usage.items():
        if agg["uses"] < min_uses:
            continue
        if token in recommender.PANTRY:
            continue  # always-on-hand staple — never nags to restock
        if token in listed:
            continue  # already on the shopping list
        stock = have.get(token, 0)
        status = _status(stock, agg["uses"], min_uses, low_threshold)
        if status is None:
            continue  # well enough stocked for how much it's used
        days_since = _days_since(agg["last_seen"], now)
        suggestions.append(
            {
                "token": token,
                "name": agg["name"],
                "uses": agg["uses"],
                "used": agg["used"],
                "wasted": agg["wasted"],
                "on_hand": stock,
                "status": status,
                "days_since_last": days_since,
                "reason": _reason(status, agg["uses"], stock, days_since),
                "est_cost": budget.price_of(token),
            }
        )

    suggestions.sort(key=_rank, reverse=True)
    suggestions = suggestions[: max(0, int(top_n))]

    total_cost = round(sum(float(s["est_cost"]) for s in suggestions), 2)
    return {
        "items": suggestions,
        "count": len(suggestions),
        "total_cost": total_cost,
        "currency": budget.CURRENCY,
        "window_weeks": max(1, int(recent_weeks)),
        "has_data": any_history,
        "headline": _headline(suggestions, any_history),
    }


# --- Helpers ----------------------------------------------------------------------


def _on_hand_totals(
    on_hand: Iterable[Mapping[str, Any]] | None,
    perishables: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Fold inventory items + tracked perishables into a ``token -> units`` map."""
    totals: dict[str, int] = {}
    for item in on_hand or []:
        token = recommender.normalize_ingredient(str(item.get("name", "")))
        if not token:
            continue
        totals[token] = totals.get(token, 0) + max(1, _as_int(item.get("count"), 1))
    for row in perishables or []:
        token = recommender.normalize_ingredient(str(row.get("name", "")))
        if not token:
            continue
        totals[token] = totals.get(token, 0) + 1
    return totals


def _status(stock: int, uses: int, min_uses: int, low_threshold: int) -> str | None:
    """Classify a staple by how much is left: ``out``, ``low`` or well-stocked (``None``)."""
    if stock <= 0:
        return "out"
    # "Low" only fires for heavy users (used at least twice the staple threshold), so a
    # single spare of something used occasionally isn't flagged.
    if stock <= low_threshold and uses >= min_uses * 2:
        return "low"
    return None


def _rank(s: Mapping[str, Any]) -> tuple[int, int, int]:
    """Sort key: out before low, then most-used, then most-recently used."""
    recency = -(s["days_since_last"] if s["days_since_last"] is not None else 9999)
    return (_STATUS_WEIGHT.get(s["status"], 0), int(s["uses"]), recency)


def _reason(status: str, uses: int, stock: int, days_since: int | None) -> str:
    """A short, plain-language 'why' for one suggestion."""
    times = f"{uses}×" if uses != 1 else "once"
    recent = f" (last about {days_since} day{'s' if days_since != 1 else ''} ago)" if days_since else ""
    if status == "out":
        return f"Used {times} lately{recent} — none left."
    left = f"{stock} left" if stock != 1 else "1 left"
    return f"Used {times} lately{recent} — only {left}."


def _headline(suggestions: Sequence[Mapping[str, Any]], any_history: bool) -> str:
    """One-line summary for the top of the restock panel."""
    if not any_history:
        return "Cook a few meals and I'll learn what you run low on."
    if not suggestions:
        return "You're well stocked on the things you use most. Nothing to restock."
    names = [str(s["name"]) for s in suggestions[:3]]
    out = sum(1 for s in suggestions if s["status"] == "out")
    if out:
        return f"Running low — worth restocking {_join(names)}."
    return f"Getting low on {_join(names)} — top up when you can."


def _join(names: Sequence[str]) -> str:
    """Human 'a, b and c' join."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _parse_ts(value: Any) -> datetime | None:
    """Parse a stored ``YYYY-MM-DDTHH:MM:SSZ`` timestamp into an aware UTC datetime."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _days_since(when: datetime | None, now: datetime) -> int | None:
    if when is None:
        return None
    return max(0, (now - when).days)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
