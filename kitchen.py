"""Kitchen actions — the operations that *change* the fridge.

These used to live inside the Flask routes. They're pulled out here so that two callers
share one implementation:

* the HTTP routes (``POST /api/cook``, ``POST /api/perishables/<id>/resolve``, ...), and
* the Chef agent's **confirmed actions**: when the agent proposes "mark Palak Paneer as
  cooked" and the user presses *Confirm*, :func:`execute` runs exactly the same code the
  Cook button runs — no second, subtly different write path.

Every function returns a JSON-serializable result plus the realtime "scopes" it touched,
so the web layer can push ``state_changed`` hints without knowing the details. Framework-
agnostic: no Flask imports; storage errors propagate to the caller.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import recommender
import storage

logger = logging.getLogger(__name__)

MAX_NAME = 80              # longest ingredient / item name accepted
MAX_COOK_ENTRIES = 50      # most ingredients one cook can consume
MAX_RECEIPT_ITEMS = 80     # most line-items one receipt import can add
DEFAULT_LEFTOVER_DAYS = 3  # how long cooked leftovers stay good, by default
MAX_LEFTOVER_DAYS = 14     # sanity ceiling on a leftover's use-by


class KitchenError(ValueError):
    """A client-side problem with an action's arguments (maps to HTTP 400/404)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --- Validation helpers -----------------------------------------------------


def cook_entries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
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

    merged: dict[str, dict[str, Any]] = {}
    for name, qty in raw:
        if len(name) > MAX_NAME:
            continue
        key = name.lower()
        if key in merged:
            merged[key]["qty"] += qty
        else:
            merged[key] = {"name": name, "qty": qty}
    return list(merged.values())


def valid_use_by(value: str) -> bool:
    """True if ``value`` is a real calendar date in ``YYYY-MM-DD`` form."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def parse_est_cost(value: Any) -> tuple[float | None, str | None]:
    """Validate an optional cost.

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


# --- Actions ------------------------------------------------------------------


def cook_meal(entries: Sequence[Mapping[str, Any]], *, title: str = "") -> dict[str, Any]:
    """Record a cooked meal: decrement the fridge, log each ingredient used, clear perishables.

    ``entries`` is the output of :func:`cook_entries`. This closes the plan → cook →
    inventory loop by

    * recording each ingredient in the **consumption ledger**, so the current-inventory view
      (fridge, recommender, optimizer, nutrition) reflects what's actually left;
    * logging each ingredient as **used** in the waste log, so Analytics shows waste falling
      as the user follows the plan; and
    * clearing as many tracked perishables as were consumed (soonest use-by first).

    Raises:
        KitchenError: no ingredients, or too many.
    """
    if not entries:
        raise KitchenError("Tell me which ingredients were used to cook this.")
    if len(entries) > MAX_COOK_ENTRIES:
        raise KitchenError("That's a lot of ingredients — please cook fewer at once.")
    title = str(title or "").strip()[:120]
    note = f"Cooked: {title}" if title else "Cooked a planned meal"

    # How many units of each canonical ingredient this meal consumed — used to clear the
    # right number of tracked perishables (not every entry that happens to share a name).
    consumed_qty: dict[str, int] = {}
    for e in entries:
        token = recommender.normalize_ingredient(e["name"])
        consumed_qty[token] = consumed_qty.get(token, 0) + int(e["qty"])

    # Ledger rows drive the inventory decrement...
    recorded = storage.record_consumption(
        [
            {
                "name": e["name"],
                "token": recommender.normalize_ingredient(e["name"]),
                "qty": int(e["qty"]),
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
    # Cooking a meal also clears the tracked perishables it used up, so the at-risk list and
    # future plans stop nagging about an ingredient that's now eaten.
    cleared = _clear_tracked(consumed_qty)

    return {
        "cooked": True,
        "title": title or None,
        "consumed": [e["name"] for e in entries],
        "recorded": recorded,
        "cleared_perishables": cleared,
        "scopes": ["inventory", "analytics", "perishables"],
    }


def discard_items(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Throw food away: take it out of the fridge, log it as wasted, stop tracking it.

    ``entries`` is the output of :func:`cook_entries`. Mirrors :func:`cook_meal` (same
    consumption ledger, so the fridge view shrinks), but logs each item as **wasted**.

    Raises:
        KitchenError: no items, or too many.
    """
    if not entries:
        raise KitchenError("Tell me what was thrown away.")
    if len(entries) > MAX_COOK_ENTRIES:
        raise KitchenError("That's a lot of items — please log fewer at once.")
    consumed_qty: dict[str, int] = {}
    rows = []
    for e in entries:
        token = recommender.normalize_ingredient(e["name"])
        consumed_qty[token] = consumed_qty.get(token, 0) + int(e["qty"])
        rows.append({"name": e["name"], "token": token, "qty": int(e["qty"])})
    recorded = storage.record_consumption(rows, note="Thrown away")
    for row in rows:
        storage.log_waste_event(row["name"], storage.WASTE_WASTED, token=row["token"])
    cleared = _clear_tracked(consumed_qty)
    return {
        "discarded": [e["name"] for e in entries],
        "recorded": recorded,
        "cleared_perishables": cleared,
        "scopes": ["inventory", "analytics", "perishables"],
    }


def _clear_tracked(consumed_qty: Mapping[str, int]) -> int:
    """Stop tracking as many perishables as were consumed, soonest use-by first.

    Tracking two blocks of cheese and cooking one meal clears one. The used/wasted event
    was already logged by the caller, so this deletes without re-logging (which would
    double-count it).
    """
    cleared = 0
    remaining = dict(consumed_qty)
    for perishable in storage.list_perishables():  # soonest use-by first
        token = recommender.normalize_ingredient(perishable.get("name", ""))
        if remaining.get(token, 0) > 0 and storage.delete_perishable(perishable["id"]):
            remaining[token] -= 1
            cleared += 1
    return cleared


def resolve_perishable(
    perishable_id: int, event: str, *, est_cost: float | None = None
) -> dict[str, Any]:
    """Close out a tracked perishable as eaten or thrown away, then stop tracking it.

    Raises:
        KitchenError: bad ``event`` (400) or unknown perishable (404).
    """
    event = str(event or "").strip().lower()
    if event not in (storage.WASTE_USED, storage.WASTE_WASTED):
        raise KitchenError("Mark the item as either 'used' or 'wasted'.")
    item = storage.get_perishable(int(perishable_id))
    if item is None:
        raise KitchenError("That item was not found.", status_code=404)

    name = str(item.get("name", "")).strip()
    event_id = storage.log_waste_event(
        name, event, token=recommender.normalize_ingredient(name), est_cost=est_cost
    )
    storage.delete_perishable(int(perishable_id))
    return {
        "resolved": True, "event": event, "id": event_id, "name": name,
        "scopes": ["perishables", "analytics"],
    }


def track_expiry(name: str, use_by: str) -> dict[str, Any]:
    """Start tracking a perishable's "use by" date.

    Raises:
        KitchenError: blank / over-long name or an invalid date.
    """
    name = str(name or "").strip()
    use_by = str(use_by or "").strip()
    if not name:
        raise KitchenError("Please enter what the item is.")
    if len(name) > MAX_NAME:
        raise KitchenError("That name is too long.")
    if not valid_use_by(use_by):
        raise KitchenError("Please enter a valid use-by date (YYYY-MM-DD).")
    perishable_id = storage.add_perishable(name, use_by)
    return {
        "perishable": {"id": perishable_id, "name": name, "use_by": use_by},
        "scopes": ["perishables"],
    }


def add_shopping(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Add items to the shopping list (skipping ones already on it)."""
    entries = []
    for obj in items or []:
        name = str((obj or {}).get("name", "")).strip()
        if not name or len(name) > MAX_NAME:
            continue
        entries.append({
            "name": name,
            "token": recommender.normalize_ingredient(name) or name.lower(),
            "qty": obj.get("qty"),
            "reason": obj.get("reason"),
        })
    if not entries:
        raise KitchenError("Tell me what to add to the shopping list.")
    added = storage.add_shopping_items(entries)
    return {"added": added, "skipped": len(entries) - len(added), "scopes": ["shopping"]}


def import_receipt(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold items bought on a shopping trip into the fridge.

    ``items`` is the shopper-reviewed list from :func:`receipts.parse_receipt` — each a
    ``{"name", "qty"?}`` object. The items are merged into the current inventory (matching
    on the canonical ingredient token so "Amul Milk" bumps the existing "Milk" rather than
    duplicating it) and the result is persisted as a fresh inventory snapshot. Because the
    whole fridge — fridge view, recommender, optimizer, nutrition — reads from the latest
    snapshot, everything picks up the new stock in one write.

    Raises:
        KitchenError: nothing usable to import.
    """
    parsed = _receipt_entries(items)
    if not parsed:
        raise KitchenError("Tell me which items to add from the receipt.")

    # Start from what's actually left (latest scan minus anything cooked since), then bump.
    current = storage.get_current_inventory(normalize=recommender.normalize_ingredient)
    merged_items = _merge_into_inventory(current, parsed)

    # A snapshot is ground truth, so saving one resets the "cooked since" ledger — correct
    # here because the current-inventory reads above already baked that consumption in.
    scan_id = storage.save_scan(
        {"items": merged_items, "unidentified": []}, storage.SOURCE_RECEIPT
    )
    return {
        "imported": [e["name"] for e in parsed],
        "added": len(parsed),
        "units": sum(e["qty"] for e in parsed),
        "scan_id": scan_id,
        "scopes": ["inventory", "nutrition", "scans"],
    }


def save_leftovers(
    name: str, days: int = DEFAULT_LEFTOVER_DAYS, *, now: datetime | None = None
) -> dict[str, Any]:
    """Track cooked leftovers as a perishable so they get eaten before they go off.

    Closes the last gap in the cook loop: after a meal, a portion often goes back in the
    fridge and is then forgotten. This records it as a tracked perishable with a short
    use-by (default 3 days), so it surfaces under "use soon" and in the Chef's briefings.

    Raises:
        KitchenError: blank or over-long name.
    """
    name = str(name or "").strip()
    if not name:
        raise KitchenError("Tell me what the leftovers are.")
    if len(name) > MAX_NAME:
        raise KitchenError("That name is too long.")

    try:
        span = int(days)
    except (TypeError, ValueError):
        span = DEFAULT_LEFTOVER_DAYS
    span = max(1, min(span, MAX_LEFTOVER_DAYS))

    base = now or datetime.now(UTC)
    use_by = (base + timedelta(days=span)).strftime("%Y-%m-%d")
    label = name if name.lower().startswith("leftover") else f"Leftover {name}"
    label = label[:MAX_NAME]

    perishable_id = storage.add_perishable(label, use_by)
    return {
        "perishable": {"id": perishable_id, "name": label, "use_by": use_by},
        "days": span,
        "scopes": ["perishables"],
    }


def _receipt_entries(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize a reviewed receipt selection into ``[{name, token, qty}]`` (deduped)."""
    out: list[dict[str, Any]] = []
    for obj in items or []:
        name = str((obj or {}).get("name", "")).strip()
        if not name or len(name) > MAX_NAME:
            continue
        try:
            qty = int((obj or {}).get("qty", 1))
        except (TypeError, ValueError):
            qty = 1
        out.append({
            "name": name,
            "token": recommender.normalize_ingredient(name) or name.lower(),
            "qty": max(1, qty),
        })
        if len(out) >= MAX_RECEIPT_ITEMS:
            break
    return out


def _merge_into_inventory(
    current: Mapping[str, Any] | None, parsed: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Return a new items list = current inventory with the parsed receipt items folded in.

    An item that matches an existing one (same canonical token) bumps its count; a new item
    is appended with neutral defaults so it renders like any other fridge item.
    """
    items: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for item in (current or {}).get("items", []) or []:
        row = dict(item)
        items.append(row)
        index[recommender.normalize_ingredient(str(row.get("name", "")))] = row

    for entry in parsed:
        row = index.get(entry["token"])
        if row is not None:
            try:
                have = int(row.get("count", 1))
            except (TypeError, ValueError):
                have = 1
            row["count"] = max(0, have) + entry["qty"]
        else:
            new_row = {
                "name": entry["name"],
                "count": entry["qty"],
                "category": "other",
                "freshness": "fresh",
                "freshness_confidence": "low",
                "notes": "Added from receipt",
            }
            items.append(new_row)
            index[entry["token"]] = new_row
    return items


# --- Confirmed-action dispatcher ------------------------------------------------

# Executors the agent may *propose*; nothing outside this table can ever run from the
# action queue, whatever a model (or injected text) asks for.
_EXECUTORS = {
    "cook_meal": lambda a: cook_meal(cook_entries(a), title=str(a.get("title", ""))),
    "discard_items": lambda a: discard_items(cook_entries(a)),
    "resolve_perishable": lambda a: resolve_perishable(
        int(a["perishable_id"]), str(a.get("event", "")), est_cost=a.get("est_cost")
    ),
    "track_expiry": lambda a: track_expiry(str(a.get("name", "")), str(a.get("use_by", ""))),
    "add_shopping": lambda a: add_shopping(a.get("items") or []),
    "import_receipt": lambda a: import_receipt(a.get("items") or []),
    "save_leftovers": lambda a: save_leftovers(
        str(a.get("name", "")), a.get("days", DEFAULT_LEFTOVER_DAYS)
    ),
}

EXECUTABLE = frozenset(_EXECUTORS)


def execute(tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Run one confirmed action. Returns the executor's result (with ``scopes``).

    Raises:
        KitchenError: unknown tool or invalid arguments.
    """
    executor = _EXECUTORS.get(str(tool))
    if executor is None:
        raise KitchenError(f"Unknown action '{tool}'.")
    try:
        return executor(dict(arguments or {}))
    except KitchenError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise KitchenError(f"That action's details are incomplete ({exc}).") from exc
