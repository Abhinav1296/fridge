"""Tests for the receipt parser (receipts.parse_receipt).

The parser is pure and offline, so these tests need no DB or network. They pin the
behaviour that matters for import accuracy: real grocery lines are kept with the right
quantity/price, receipt chrome (headers, totals, tax, payment, barcodes) is filtered out,
duplicates merge, and every output is JSON-serializable for the preview endpoint.
"""
from __future__ import annotations

import json

import receipts

INDIAN_RECEIPT = """BIG BAZAAR
123 MG Road, Bengaluru
GSTIN: 29ABCDE1234F1Z5
Date: 25/09/2026  Time: 18:42
--------------------------------
Item            Qty   Amount
Amul Milk 500ml  2 x 27.00  54.00
Tomato 1kg              40.00
Paneer 200g            80.00
Basmati Rice 5kg      450.00
Carry Bag               5.00
--------------------------------
Subtotal              629.00
CGST 2.5%              15.73
SGST 2.5%              15.73
Total                 660.46
Cash                  700.00
Change                 39.54
Thank you! Visit again"""


def _by_token(result):
    return {item["token"]: item for item in result["items"]}


def test_parses_grocery_items_from_a_full_receipt():
    result = receipts.parse_receipt(INDIAN_RECEIPT)
    items = _by_token(result)
    assert result["has_items"] is True
    assert "milk" in items and "tomato" in items and "paneer" in items
    assert items["milk"]["qty"] == 2
    assert items["milk"]["price"] == 54.0
    assert items["tomato"]["qty"] == 1
    assert items["paneer"]["price"] == 80.0


def test_skips_totals_tax_payment_and_store_chrome():
    result = receipts.parse_receipt(INDIAN_RECEIPT)
    tokens = {item["token"] for item in result["items"]}
    # None of the receipt chrome should have become an item.
    for junk in ("total", "subtotal", "cgst", "sgst", "cash", "change", "gstin", "amount"):
        assert junk not in tokens
    # Address / store-name lines have no price and aren't food → dropped.
    assert not any("bazaar" in t or "road" in t for t in tokens)
    assert result["skipped"] > 0


def test_flags_non_food_lines_as_unknown_but_keeps_priced_ones():
    result = receipts.parse_receipt(INDIAN_RECEIPT)
    items = _by_token(result)
    # A "Carry Bag" line has a price so it's a real line-item, but isn't food.
    assert "carry_bag" in items
    assert items["carry_bag"]["known"] is False
    assert items["milk"]["known"] is True
    assert result["known_count"] == result["count"] - 1


def test_detects_inline_and_leading_quantities():
    inline = receipts.parse_receipt("Onion 3 x 20.00  60.00")
    assert inline["items"][0]["qty"] == 3
    assert inline["items"][0]["price"] == 60.0

    leading = receipts.parse_receipt("2 Milk 54.00")
    assert leading["items"][0]["qty"] == 2

    # A trailing "x4" counts even without a price, because the item is a known food.
    trailing = receipts.parse_receipt("Bread x4")
    assert trailing["items"][0]["qty"] == 4


def test_currency_detection():
    assert receipts.parse_receipt("Milk ₹27.00")["currency"] == "₹"
    assert receipts.parse_receipt("Milk $3.49")["currency"] == "$"
    assert receipts.parse_receipt("Milk Rs.27")["currency"] == "₹"
    # No symbol at all → default to the app's rupee.
    assert receipts.parse_receipt("Paneer")["currency"] == "₹"


def test_merges_duplicate_items():
    result = receipts.parse_receipt("Milk 27.00\nAmul Milk 500ml 54.00")
    items = _by_token(result)
    assert result["count"] == 1
    assert items["milk"]["qty"] == 2
    assert items["milk"]["price"] == 81.0


def test_strips_units_percentages_and_barcodes_from_names():
    result = receipts.parse_receipt("Amul Milk 500ml 2% 8901234567890 27.00")
    name = result["items"][0]["name"]
    assert "500" not in name and "%" not in name
    assert "8901234567890" not in name
    assert result["items"][0]["token"] == "milk"


def test_requires_price_or_known_food():
    # A bare store name — no price, not a food — must not become an item.
    assert receipts.parse_receipt("BIG BAZAAR SUPERSTORE")["count"] == 0
    # A known food with no price is still kept.
    assert receipts.parse_receipt("Tomato")["count"] == 1


def test_empty_and_junk_return_no_items():
    empty = receipts.parse_receipt("")
    assert empty["has_items"] is False
    assert empty["count"] == 0

    junk = receipts.parse_receipt("TOTAL 500\nTHANK YOU\n========")
    assert junk["has_items"] is False
    assert junk["count"] == 0


def test_caps_number_of_items():
    many = "\n".join(f"Item{i} 1{i}.00" for i in range(200))
    result = receipts.parse_receipt(many, max_items=10)
    assert result["count"] <= 10


def test_output_is_json_serializable():
    result = receipts.parse_receipt(INDIAN_RECEIPT)
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["count"] == result["count"]
    expected_keys = {
        "items", "count", "known_count", "total", "currency",
        "lines_seen", "skipped", "has_items", "headline",
    }
    assert expected_keys <= set(round_tripped)
    for item in round_tripped["items"]:
        assert {"name", "token", "qty", "price", "known", "raw"} <= set(item)
