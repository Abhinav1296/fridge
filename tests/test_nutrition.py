"""Tests for the Open Food Facts nutrition layer.

Fully offline: HTTP goes through ``nutrition.requests.get``, which is monkeypatched with a
fake returning canned JSON, and :func:`nutrition.enrich` takes its cache access and (for
some tests) its fetcher as injected callables, so nothing touches the network or a database.
"""

from __future__ import annotations

import pytest
import requests

import nutrition

# --- Test doubles -----------------------------------------------------------


class _FakeResponse:
    def __init__(self, *, json_body=None, status_code=200, raise_json=False):
        self._json_body = json_body
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._json_body


def _patch_get(monkeypatch, response):
    """Point ``nutrition.requests.get`` at a fake returning ``response`` (or raising it)."""

    def fake_get(url, params=None, headers=None, timeout=None):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(nutrition.requests, "get", fake_get)


def _search_body(products):
    return {"count": len(products), "products": products}


# --- _extract_nutriments ----------------------------------------------------


def test_extract_nutriments_reads_per_100g():
    product = {
        "product_name": "Paneer",
        "nutriments": {
            "energy-kcal_100g": 265,
            "proteins_100g": 18.3,
            "carbohydrates_100g": 1.2,
            "fat_100g": 20.8,
        },
    }
    macros = nutrition._extract_nutriments(product)
    assert macros == {"kcal": 265.0, "protein_g": 18.3, "carbs_g": 1.2, "fat_g": 20.8}


def test_extract_nutriments_none_when_empty():
    assert nutrition._extract_nutriments({"nutriments": {}}) is None
    assert nutrition._extract_nutriments({"product_name": "x"}) is None
    assert nutrition._extract_nutriments("not a dict") is None


def test_extract_nutriments_partial_and_negatives():
    # Only protein known; a stray negative energy is treated as unknown.
    product = {"nutriments": {"proteins_100g": 8, "energy-kcal_100g": -5}}
    macros = nutrition._extract_nutriments(product)
    assert macros == {"kcal": None, "protein_g": 8.0, "carbs_g": None, "fat_g": None}


# --- fetch_by_name ----------------------------------------------------------


def test_fetch_by_name_returns_first_hit_with_data(monkeypatch):
    products = [
        {"product_name": "No data", "nutriments": {}},          # skipped
        {"product_name": "Tomato", "nutriments": {"energy-kcal_100g": 18, "proteins_100g": 0.9}},
    ]
    _patch_get(monkeypatch, _FakeResponse(json_body=_search_body(products)))

    macros = nutrition.fetch_by_name("tomato")
    assert macros["kcal"] == 18.0
    assert macros["protein_g"] == 0.9
    assert macros["product_name"] == "Tomato"
    assert macros["source"] == nutrition.SOURCE


def test_fetch_by_name_none_when_no_usable_products(monkeypatch):
    _patch_get(monkeypatch, _FakeResponse(json_body=_search_body([{"nutriments": {}}])))
    assert nutrition.fetch_by_name("obscurity") is None


def test_fetch_by_name_empty_term_short_circuits(monkeypatch):
    # No network call should be needed for a blank term.
    def boom(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("should not hit the network")

    monkeypatch.setattr(nutrition.requests, "get", boom)
    assert nutrition.fetch_by_name("   ") is None


# --- fetch_by_barcode -------------------------------------------------------


def test_fetch_by_barcode_returns_product(monkeypatch):
    body = {
        "status": 1,
        "product": {
            "product_name": "Amul Paneer",
            "brands": "Amul",
            "nutriments": {"energy-kcal_100g": 296, "proteins_100g": 20},
        },
    }
    _patch_get(monkeypatch, _FakeResponse(json_body=body))

    product = nutrition.fetch_by_barcode("8901262010016")
    assert product["product_name"] == "Amul Paneer"
    assert product["brands"] == "Amul"
    assert product["kcal"] == 296.0
    assert product["code"] == "8901262010016"


def test_fetch_by_barcode_none_for_unknown(monkeypatch):
    _patch_get(monkeypatch, _FakeResponse(json_body={"status": 0}))
    assert nutrition.fetch_by_barcode("0000000000000") is None


def test_fetch_by_barcode_strips_non_digits(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["url"] = url
        return _FakeResponse(json_body={"product": {"product_name": "X", "nutriments": {"proteins_100g": 1}}})

    monkeypatch.setattr(nutrition.requests, "get", fake_get)
    nutrition.fetch_by_barcode(" 890-126 ")
    assert "890126" in seen["url"]


# --- _get error handling ----------------------------------------------------


def test_get_raises_on_network_error(monkeypatch):
    _patch_get(monkeypatch, requests.exceptions.ConnectionError("down"))
    with pytest.raises(nutrition.NutritionError):
        nutrition._get(nutrition._SEARCH_URL, {})


def test_get_raises_on_http_error(monkeypatch):
    _patch_get(monkeypatch, _FakeResponse(status_code=500))
    with pytest.raises(nutrition.NutritionError):
        nutrition._get(nutrition._SEARCH_URL, {})


def test_get_raises_on_bad_json(monkeypatch):
    _patch_get(monkeypatch, _FakeResponse(raise_json=True))
    with pytest.raises(nutrition.NutritionError):
        nutrition._get(nutrition._SEARCH_URL, {})


# --- enrich (dependency-injected cache + fetcher) ---------------------------


def test_enrich_updates_missing_and_skips_cached():
    cache = {"paneer": {"item": "paneer", "kcal": 265}}
    writes = []

    def get_cached(token):
        return cache.get(token)

    def upsert(item, **kw):
        writes.append((item, kw))

    def fetch(name):
        return {"kcal": 18, "protein_g": 0.9, "carbs_g": 3.9, "fat_g": 0.2, "source": nutrition.SOURCE}

    summary = nutrition.enrich(
        ["paneer", "tomato", "tomato"],  # duplicate collapses
        get_cached=get_cached, upsert=upsert, fetch=fetch,
    )
    assert summary["cached"] == ["paneer"]
    assert summary["updated"] == ["tomato"]
    assert len(writes) == 1 and writes[0][0] == "tomato"


def test_enrich_force_refetches_cached():
    cache = {"paneer": {"item": "paneer", "kcal": 265}}
    writes = []
    nutrition.enrich(
        ["paneer"],
        get_cached=lambda t: cache.get(t),
        upsert=lambda item, **kw: writes.append(item),
        fetch=lambda name: {"kcal": 300},
        force=True,
    )
    assert writes == ["paneer"]


def test_enrich_records_not_found_and_failed():
    def fetch(name):
        if name == "Ghost":
            return None
        raise nutrition.NutritionError("boom")

    summary = nutrition.enrich(
        ["ghost", "broken"],
        get_cached=lambda t: None,
        upsert=lambda *a, **k: None,
        fetch=fetch,
    )
    assert summary["not_found"] == ["ghost"]
    assert summary["failed"] == ["broken"]


# --- tokens_from_inventory --------------------------------------------------


def test_tokens_from_inventory_normalizes_and_dedupes():
    items = [{"name": "Tomatoes"}, {"name": "Paneer"}, {"name": "tomato"}]
    perishables = [{"name": "Paneer"}, {"name": "Milk"}]
    tokens = nutrition.tokens_from_inventory(items, perishables)
    # 'Tomatoes' and 'tomato' collapse to one token; 'Paneer' appears once.
    assert tokens.count("tomato") == 1
    assert tokens.count("paneer") == 1
    assert "milk" in tokens
