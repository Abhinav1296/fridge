"""Tests for :mod:`storage` — scans, perishables, and the nutrition cache.

Runs against a temporary local SQLite database (configured in ``conftest.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import recommender
import storage


def test_save_and_get_latest_scan():
    result = {
        "items": [
            {"name": "Tomato", "count": 3, "category": "vegetable", "freshness": "fresh"},
            {"name": "Milk", "count": 1, "category": "dairy", "freshness": "use_soon"},
        ],
        "unidentified": [{"description": "opaque bag", "reason": "occluded"}],
    }
    scan_id = storage.save_scan(result, storage.SOURCE_UPLOAD)
    assert isinstance(scan_id, int) and scan_id > 0

    latest = storage.get_latest_scan()
    assert latest is not None
    assert latest["id"] == scan_id
    assert latest["source"] == storage.SOURCE_UPLOAD
    assert latest["item_count"] == 2
    names = {item["name"] for item in latest["items"]}
    assert names == {"Tomato", "Milk"}
    assert len(latest["unidentified"]) == 1


def test_get_scan_roundtrip_and_missing():
    scan_id = storage.save_scan({"items": [], "unidentified": []}, storage.SOURCE_WEBCAM)
    fetched = storage.get_scan(scan_id)
    assert fetched is not None and fetched["id"] == scan_id
    assert storage.get_scan(999_999) is None


def test_perishable_add_list_delete(clean_db):
    pid = storage.add_perishable("Paneer", "2026-10-01")
    storage.add_perishable("Milk", "2026-09-25")

    rows = storage.list_perishables()
    assert [r["name"] for r in rows] == ["Milk", "Paneer"]  # soonest use_by first

    assert storage.delete_perishable(pid) is True
    assert storage.delete_perishable(pid) is False  # already gone
    assert [r["name"] for r in storage.list_perishables()] == ["Milk"]

    assert storage.clear_perishables() == 1
    assert storage.list_perishables() == []


def test_nutrition_upsert_get_list(clean_db):
    assert storage.get_nutrition("Paneer") is None

    storage.upsert_nutrition(
        "Paneer", kcal=265.0, protein_g=18.0, carbs_g=1.2, fat_g=20.8, source="test"
    )
    row = storage.get_nutrition("paneer")  # case-insensitive key
    assert row is not None
    assert row["kcal"] == 265.0
    assert row["source"] == "test"

    # Re-caching overwrites in place (upsert), not a second row.
    storage.upsert_nutrition("PANEER", kcal=270.0, source="test2")
    row = storage.get_nutrition("paneer")
    assert row["kcal"] == 270.0
    assert row["source"] == "test2"
    assert len(storage.list_nutrition()) == 1


def test_nutrition_upsert_ignores_blank_item(clean_db):
    storage.upsert_nutrition("   ", kcal=100.0)
    assert storage.list_nutrition() == []


def test_created_at_override_is_utc_iso():
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    scan_id = storage.save_scan({"items": [], "unidentified": []}, storage.SOURCE_UPLOAD, created_at=when)
    row = storage.get_scan(scan_id)
    assert row["created_at"] == "2026-01-02T03:04:05Z"


# --- Consumption ledger + reconciled current inventory ----------------------


def test_current_inventory_subtracts_consumption(clean_db):
    storage.save_scan(
        {
            "items": [
                {"name": "Paneer", "count": 2},
                {"name": "Tomato", "count": 3},
                {"name": "Onion", "count": 1},
            ],
            "unidentified": [],
        },
        storage.SOURCE_UPLOAD,
    )
    written = storage.record_consumption(
        [
            {"name": "Paneer", "token": recommender.normalize_ingredient("Paneer"), "qty": 1},
            {"name": "Onion", "token": recommender.normalize_ingredient("Onion"), "qty": 1},
        ],
        note="Cooked: Test Dish",
    )
    assert written == 2

    current = storage.get_current_inventory(normalize=recommender.normalize_ingredient)
    assert current["adjusted_for_consumption"] is True
    by_name = {i["name"]: i["count"] for i in current["items"]}
    assert by_name == {"Paneer": 1, "Tomato": 3}  # Onion fully consumed -> dropped
    assert current["item_count"] == 2

    # The raw scan is immutable — history still shows the full snapshot.
    raw = storage.get_latest_scan()
    assert {i["name"]: i["count"] for i in raw["items"]} == {"Paneer": 2, "Tomato": 3, "Onion": 1}
    assert "adjusted_for_consumption" not in raw


def test_new_scan_resets_consumption_ledger(clean_db):
    storage.save_scan({"items": [{"name": "Paneer", "count": 1}], "unidentified": []}, storage.SOURCE_UPLOAD)
    storage.record_consumption([{"name": "Paneer", "qty": 1}])
    assert len(storage.list_consumption()) == 1

    # A fresh snapshot supersedes the deltas: the ledger is cleared on save.
    storage.save_scan({"items": [{"name": "Milk", "count": 2}], "unidentified": []}, storage.SOURCE_UPLOAD)
    assert storage.list_consumption() == []

    current = storage.get_current_inventory(normalize=recommender.normalize_ingredient)
    assert {i["name"]: i["count"] for i in current["items"]} == {"Milk": 2}
    assert "adjusted_for_consumption" not in current  # nothing consumed since this scan


def test_record_consumption_skips_blanks_and_defaults_qty(clean_db):
    written = storage.record_consumption([{"name": "  "}, {"name": "Eggs"}])
    assert written == 1  # blank-name entry dropped
    rows = storage.list_consumption()
    assert len(rows) == 1
    assert rows[0]["name"] == "Eggs"
    assert rows[0]["qty"] == 1  # default quantity
    assert storage.clear_consumption() == 1
    assert storage.list_consumption() == []
