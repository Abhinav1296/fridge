"""Endpoint smoke tests for the Flask web layer.

These exercise the request/response contract without hitting the vision provider (the
``/analyze`` route, which needs a real image + API, is intentionally not called here).
Storage is the hermetic temp database from ``conftest.py``.
"""

from __future__ import annotations

import pytest

import app as app_module


@pytest.fixture()
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_index_serves_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SmartFridge" in resp.data


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"
    assert body["backend"] == "local"  # tests force local-SQLite mode (conftest)
    assert {"vision_configured", "agent_configured"} <= set(body)


def test_healthz_alias(client):
    assert client.get("/healthz").status_code == 200


def test_api_current_ok(client):
    resp = client.get("/api/current")
    assert resp.status_code == 200
    assert "scan" in resp.get_json()


def test_api_history_ok(client):
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert isinstance(resp.get_json()["scans"], list)


def test_api_recommendations_ok(client):
    resp = client.get("/api/recommendations")
    assert resp.status_code == 200
    body = resp.get_json()
    assert {"use_soon", "recipes", "shopping"} <= set(body)


def test_api_plan_ok(client):
    resp = client.post("/api/plan", json={"days": 1, "meals_per_day": 2})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["solver"] in ("ilp", "greedy")
    assert body["horizon"]["slots"] == 2
    assert isinstance(body["plan"], list)
    assert "waste_avoided_pct" in body["metrics"]


def test_api_plan_defaults_and_clamps(client):
    # Out-of-range values are clamped; no body at all is fine.
    resp = client.post("/api/plan", json={"days": 99, "meals_per_day": 0})
    assert resp.status_code == 200
    horizon = resp.get_json()["horizon"]
    assert horizon["days"] == 7 and horizon["meals_per_day"] == 1


def test_api_plan_is_priced_and_grouped_by_day(client, monkeypatch, clean_db):
    # A stocked fridge yields a priced plan with per-day buckets for the Autopilot view.
    monkeypatch.setattr(
        "app.storage.get_current_inventory",
        lambda **_kw: {"items": [
            {"name": "Paneer"}, {"name": "Tomato"}, {"name": "Onion"}, {"name": "Rice"},
        ]},
    )
    resp = client.post("/api/plan", json={"days": 2, "meals_per_day": 2})
    assert resp.status_code == 200
    body = resp.get_json()
    # Costs are present.
    assert "currency" in body["metrics"]
    assert "shopping_cost" in body["metrics"]
    # by_day is a list of buckets whose meals sum to the flat plan.
    assert isinstance(body["by_day"], list)
    assert sum(len(d["meals"]) for d in body["by_day"]) == len(body["plan"])
    if body["plan"]:
        assert "est_cost" in body["plan"][0]


def test_api_plan_honours_a_budget_ceiling(client, monkeypatch, clean_db):
    monkeypatch.setattr(
        "app.storage.get_current_inventory",
        lambda **_kw: {"items": [
            {"name": "Tomato"}, {"name": "Onion"}, {"name": "Rice"}, {"name": "Potato"},
        ]},
    )
    resp = client.post("/api/plan", json={"days": 3, "meals_per_day": 2, "budget": 15})
    assert resp.status_code == 200
    m = resp.get_json()["metrics"]
    assert m["budget"] == 15.0
    assert m["shopping_cost"] <= 15.0 + 1e-6
    assert m["within_budget"] is True


def test_api_plan_bad_budget_is_ignored(client):
    # A non-numeric budget doesn't error — it's treated as "no ceiling".
    resp = client.post("/api/plan", json={"days": 1, "meals_per_day": 2, "budget": "lots"})
    assert resp.status_code == 200
    assert "budget" not in resp.get_json()["metrics"]


def test_api_agent_requires_message(client):
    resp = client.post("/api/agent", json={"message": "   "})
    assert resp.status_code == 400


def test_api_agent_unconfigured_answers_offline(client, monkeypatch, clean_db):
    # No LLM key: Chef still answers, with the rules-based fallback (not a 503).
    monkeypatch.setattr("app.agent.configured", lambda: False)
    resp = client.post("/api/agent", json={"message": "what can I cook?"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["mode"] == "offline"
    assert body["answer"]


def test_api_agent_happy_path_with_stubbed_model(client, monkeypatch):
    monkeypatch.setattr("app.agent.configured", lambda: True)

    def fake_run_agent(message, *, tools, history=None, **kwargs):
        # Confirm the web layer wired the full real toolset to the agent.
        assert {"get_inventory", "get_recommendations", "plan_meals", "autopilot_week",
                "search_recipes", "get_waste_report", "get_shopping_list", "get_preferences",
                "remember_preference", "forget_preference", "add_to_shopping_list",
                "propose_cook_meal", "propose_track_expiry", "propose_log_item"} == set(tools)
        return {"answer": "Try paneer bhurji.", "steps": [], "mode": "json", "model": "test"}

    monkeypatch.setattr("app.agent.run_agent", fake_run_agent)
    resp = client.post("/api/agent", json={"message": "what can I cook tonight?"})
    assert resp.status_code == 200
    assert resp.get_json()["answer"] == "Try paneer bhurji."


def test_api_nutrition_dashboard_ok(client):
    resp = client.get("/api/nutrition")
    assert resp.status_code == 200
    body = resp.get_json()
    assert {"items", "totals", "coverage", "cached_count"} <= set(body)
    assert isinstance(body["items"], list)
    assert {"kcal", "protein_g", "carbs_g", "fat_g"} <= set(body["totals"])


def test_api_nutrition_refresh_enriches(client, monkeypatch, clean_db):
    # Pretend the fridge holds tomato + paneer, and stub the OFF fetcher.
    monkeypatch.setattr(
        "app.storage.get_latest_scan",
        lambda: {"items": [{"name": "Tomato"}, {"name": "Paneer"}]},
    )
    monkeypatch.setattr("app.storage.list_perishables", lambda: [])
    monkeypatch.setattr(
        "app.nutrition.fetch_by_name",
        lambda name: {"kcal": 20, "protein_g": 1, "carbs_g": 4, "fat_g": 0, "source": "openfoodfacts"},
    )
    resp = client.post("/api/nutrition/refresh", json={})
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body["updated"]) == {"tomato", "paneer"}
    # The dashboard now reports coverage for those items.
    dash = client.get("/api/nutrition").get_json()
    assert dash["cached_count"] >= 2


def test_api_barcode_validation(client):
    assert client.post("/api/barcode", json={"code": "123"}).status_code == 400
    assert client.post("/api/barcode", json={"code": ""}).status_code == 400


def test_api_barcode_lookup_and_cache(client, monkeypatch, clean_db):
    monkeypatch.setattr(
        "app.nutrition.fetch_by_barcode",
        lambda code: {
            "product_name": "Amul Paneer", "brands": "Amul", "code": code,
            "kcal": 296, "protein_g": 20, "carbs_g": 6, "fat_g": 22, "source": "openfoodfacts",
        },
    )
    resp = client.post("/api/barcode", json={"code": "8901262010016"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["product"]["name"] == "Amul Paneer"
    assert body["cached"] is True


def test_api_barcode_not_found(client, monkeypatch):
    monkeypatch.setattr("app.nutrition.fetch_by_barcode", lambda code: None)
    resp = client.post("/api/barcode", json={"code": "0000000000000"})
    assert resp.status_code == 404


def test_api_barcode_upstream_error_502(client, monkeypatch):
    def boom(code):
        raise app_module.NutritionError("OFF down")

    monkeypatch.setattr("app.nutrition.fetch_by_barcode", boom)
    resp = client.post("/api/barcode", json={"code": "8901262010016"})
    assert resp.status_code == 502


def test_perishables_crud(client):
    # Reject an invalid date.
    bad = client.post("/api/perishables", json={"name": "Milk", "use_by": "not-a-date"})
    assert bad.status_code == 400

    # Reject a missing name.
    noname = client.post("/api/perishables", json={"name": "", "use_by": "2026-10-01"})
    assert noname.status_code == 400

    # Happy path.
    created = client.post("/api/perishables", json={"name": "Milk", "use_by": "2026-10-01"})
    assert created.status_code == 201
    pid = created.get_json()["perishable"]["id"]

    listed = client.get("/api/perishables").get_json()["perishables"]
    assert any(p["id"] == pid for p in listed)

    deleted = client.delete(f"/api/perishables/{pid}")
    assert deleted.status_code == 200
    assert deleted.get_json()["deleted"] is True

    # Deleting again is a 404.
    assert client.delete(f"/api/perishables/{pid}").status_code == 404


def test_api_analytics_shape(client):
    resp = client.get("/api/analytics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert {"totals", "waste_rate", "money", "top_wasted", "by_week", "headline", "has_data"} <= set(body)
    assert {"used", "wasted", "events"} <= set(body["totals"])


def test_api_waste_validation(client):
    # Missing name.
    assert client.post("/api/waste", json={"name": "", "event": "used"}).status_code == 400
    # Bad event.
    assert client.post("/api/waste", json={"name": "Milk", "event": "eaten"}).status_code == 400
    # Negative cost.
    bad_cost = client.post("/api/waste", json={"name": "Milk", "event": "used", "est_cost": -1})
    assert bad_cost.status_code == 400


def test_api_waste_log_feeds_analytics(client, clean_db):
    used = client.post("/api/waste", json={"name": "Milk", "event": "used", "est_cost": 3})
    assert used.status_code == 201
    wasted = client.post("/api/waste", json={"name": "Spinach", "event": "wasted", "est_cost": 2})
    assert wasted.status_code == 201

    body = client.get("/api/analytics").get_json()
    assert body["totals"] == {"used": 1, "wasted": 1, "events": 2}
    assert body["waste_rate"] == 50.0
    assert body["money"]["has_cost"] is True
    assert body["money"]["wasted"] == 2.0
    assert body["has_data"] is True
    assert body["top_wasted"][0]["name"] == "Spinach"


def test_api_restock_shape_empty(client, clean_db):
    resp = client.get("/api/restock")
    assert resp.status_code == 200
    body = resp.get_json()
    assert {"items", "count", "total_cost", "currency", "window_weeks", "has_data", "headline"} <= set(body)
    assert body["has_data"] is False
    assert body["count"] == 0
    assert body["items"] == []


def test_api_restock_flags_depleted_staple(client, clean_db):
    # A fresh scan without Paneer (latest scan wins; clean_db leaves earlier scans in place).
    app_module.storage.save_scan(
        {"items": [{"name": "Rice", "count": 1}], "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )
    # Go through Paneer repeatedly (logged used) with none in the fridge → flagged "out".
    for _ in range(3):
        assert client.post("/api/waste", json={"name": "Paneer", "event": "used"}).status_code == 201

    body = client.get("/api/restock").get_json()
    assert body["has_data"] is True
    tokens = {i["token"]: i for i in body["items"]}
    assert "paneer" in tokens
    assert tokens["paneer"]["status"] == "out"
    assert tokens["paneer"]["uses"] == 3
    assert body["count"] >= 1


def test_api_restock_skips_what_is_on_hand(client, clean_db):
    # Same usage, but Paneer is sitting in the fridge → not suggested.
    app_module.storage.save_scan(
        {"items": [{"name": "Paneer", "count": 2}], "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )
    for _ in range(3):
        client.post("/api/waste", json={"name": "Paneer", "event": "used"})

    body = client.get("/api/restock").get_json()
    assert all(i["token"] != "paneer" for i in body["items"])


def test_api_restock_skips_items_already_on_shopping_list(client, clean_db):
    app_module.storage.save_scan(
        {"items": [{"name": "Rice", "count": 1}], "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )
    for _ in range(3):
        client.post("/api/waste", json={"name": "Paneer", "event": "used"})
    # Queue Paneer to buy → the nudge shouldn't repeat it.
    assert client.post("/api/shopping", json={"items": [{"name": "Paneer"}]}).status_code in (200, 201)

    body = client.get("/api/restock").get_json()
    assert all(i["token"] != "paneer" for i in body["items"])


def test_api_receipt_parse_previews_items_without_writing(client, clean_db):
    text = "Amul Milk 500ml 2 x 27.00 54.00\nTomato 1kg 40.00\nTotal 94.00\nThank you"
    before = len(app_module.storage.list_scans())
    resp = client.post("/api/receipt/parse", json={"text": text})
    assert resp.status_code == 200
    body = resp.get_json()
    tokens = {i["token"] for i in body["items"]}
    assert "milk" in tokens and "tomato" in tokens
    assert "total" not in tokens
    # Preview is read-only: no scan was written.
    assert len(app_module.storage.list_scans()) == before


def test_api_receipt_parse_rejects_blank(client, clean_db):
    assert client.post("/api/receipt/parse", json={"text": "   "}).status_code == 400


def test_api_receipt_import_adds_items_to_the_fridge(client, clean_db):
    # Authoritative empty fridge (clean_db leaves earlier scans in place).
    app_module.storage.save_scan({"items": [], "unidentified": []}, app_module.storage.SOURCE_UPLOAD)
    resp = client.post(
        "/api/receipt/import",
        json={"items": [{"name": "Tomato", "qty": 2}, {"name": "Paneer", "qty": 1}]},
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["added"] == 2
    names = {i["name"]: i["count"] for i in body["scan"]["items"]}
    assert names.get("Tomato") == 2 and names.get("Paneer") == 1


def test_api_receipt_import_rejects_empty(client, clean_db):
    assert client.post("/api/receipt/import", json={"items": []}).status_code == 400


def test_api_leftovers_tracks_a_perishable(client, clean_db):
    resp = client.post("/api/leftovers", json={"name": "Palak Paneer", "days": 3})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["perishable"]["name"] == "Leftover Palak Paneer"
    names = [p["name"] for p in app_module.storage.list_perishables()]
    assert "Leftover Palak Paneer" in names


def test_api_leftovers_rejects_blank(client, clean_db):
    assert client.post("/api/leftovers", json={"name": ""}).status_code == 400


def test_api_resolve_perishable(client, clean_db):
    created = client.post("/api/perishables", json={"name": "Paneer", "use_by": "2026-10-01"})
    pid = created.get_json()["perishable"]["id"]

    # Bad event is rejected before anything is logged.
    assert client.post(f"/api/perishables/{pid}/resolve", json={"event": "nope"}).status_code == 400

    # Resolving as "wasted" logs the event and stops tracking the item.
    resolved = client.post(f"/api/perishables/{pid}/resolve", json={"event": "wasted"})
    assert resolved.status_code == 200
    assert resolved.get_json()["resolved"] is True

    # It's gone from the tracked list...
    listed = client.get("/api/perishables").get_json()["perishables"]
    assert all(p["id"] != pid for p in listed)

    # ...and shows up in analytics as a wasted item.
    body = client.get("/api/analytics").get_json()
    assert body["totals"]["wasted"] == 1
    assert body["top_wasted"][0]["name"].lower() == "paneer"

    # Resolving a non-existent perishable is a 404.
    assert client.post(f"/api/perishables/{pid}/resolve", json={"event": "used"}).status_code == 404


def test_api_cook_decrements_inventory_and_logs_used(client, clean_db):
    # Seed a scan directly so there's inventory to cook from (no vision call needed).
    app_module.storage.save_scan(
        {"items": [{"name": "Paneer", "count": 2}, {"name": "Onion", "count": 1}], "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )

    resp = client.post("/api/cook", json={"uses": ["Paneer", "Onion"], "title": "Paneer Bhurji"})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["cooked"] is True
    assert set(body["consumed"]) == {"Paneer", "Onion"}
    # The returned inventory is already reconciled: Onion used up (dropped), Paneer down to 1.
    assert {i["name"]: i["count"] for i in body["scan"]["items"]} == {"Paneer": 1}

    # /api/current reflects the same decremented view.
    current = client.get("/api/current").get_json()["scan"]
    assert {i["name"]: i["count"] for i in current["items"]} == {"Paneer": 1}

    # Each cooked ingredient is logged "used", so analytics counts them (waste trends down).
    totals = client.get("/api/analytics").get_json()["totals"]
    assert totals["used"] == 2
    assert totals["wasted"] == 0


def test_api_cook_requires_ingredients(client, clean_db):
    assert client.post("/api/cook", json={"uses": []}).status_code == 400
    assert client.post("/api/cook", json={}).status_code == 400


def test_api_cook_clears_matching_perishable_without_double_logging(client, clean_db):
    # Paneer is in the scan; Cheese is only a tracked (at-risk) perishable.
    app_module.storage.save_scan(
        {"items": [{"name": "Paneer", "count": 1}], "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )
    app_module.storage.add_perishable("Cheese", "2026-10-01")

    resp = client.post("/api/cook", json={"uses": ["Paneer", "Cheese"], "title": "Grilled Cheese"})
    assert resp.status_code == 201
    assert resp.get_json()["cleared_perishables"] == 1

    # The eaten perishable is no longer tracked (loop closed for the at-risk list too).
    assert client.get("/api/perishables").get_json()["perishables"] == []

    # Each ingredient is logged "used" exactly once — clearing the perishable must NOT
    # log a second event for Cheese (no double counting in analytics).
    assert client.get("/api/analytics").get_json()["totals"]["used"] == 2


def test_api_cook_clears_only_the_consumed_quantity_of_a_perishable(client, clean_db):
    # Two separate blocks of cheese are tracked; cooking one meal uses one.
    app_module.storage.add_perishable("Cheese", "2026-10-01")  # soonest use-by
    app_module.storage.add_perishable("Cheese", "2026-10-05")

    resp = client.post("/api/cook", json={"uses": ["Cheese"], "title": "Grilled Cheese"})
    assert resp.status_code == 201
    assert resp.get_json()["cleared_perishables"] == 1  # only one unit was cooked

    # One block of cheese is still tracked (the later use-by; soonest is cleared first).
    remaining = client.get("/api/perishables").get_json()["perishables"]
    assert [p["use_by"] for p in remaining] == ["2026-10-05"]


def test_api_cook_session_returns_a_full_guided_walkthrough(client, clean_db):
    # Seed a fridge with two of the curry's ingredients so the checklist has have/missing marks.
    app_module.storage.save_scan(
        {"items": [{"name": "Paneer", "count": 1}, {"name": "Tomato", "count": 2}],
         "unidentified": []},
        app_module.storage.SOURCE_UPLOAD,
    )

    resp = client.get("/api/cook/paneer_butter_masala")
    assert resp.status_code == 200
    body = resp.get_json()

    # The shape the Cook Mode UI renders.
    assert body["recipe_id"] == "paneer_butter_masala"
    assert body["title"]
    assert body["method"] == "curry"
    assert body["total_count"] == len(body["ingredients"])
    assert body["have_count"] + body["missing_count"] == body["total_count"]

    # Steps are numbered 1..N and the timer total is self-consistent.
    steps = body["steps"]
    assert [s["n"] for s in steps] == list(range(1, len(steps) + 1))
    assert body["total_timer_seconds"] == sum(s["seconds"] for s in steps)

    # The seeded fridge is reflected in the checklist.
    have = {i["token"] for i in body["ingredients"] if i["have"]}
    assert {"paneer", "tomato"} <= have


def test_api_cook_session_unknown_recipe_is_404(client, clean_db):
    resp = client.get("/api/cook/not_a_real_recipe")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


def test_api_cook_session_never_suggests_meat_for_a_vegetarian_dish(client, clean_db):
    # The vegetarian guard must hold through the HTTP layer too, not just in the module.
    body = client.get("/api/cook/paneer_butter_masala").get_json()
    suggested = {s["token"] for i in body["ingredients"] for s in i["substitutes"]}
    assert not (suggested & {"chicken", "egg", "mutton", "fish", "prawn"})


def test_api_cook_session_honours_dietary_exclusions_in_swaps(client, clean_db):
    # A disliked ingredient recorded in memory must never appear as a swap idea.
    app_module.storage.set_memory("dislikes", ["cheese"])
    body = client.get("/api/cook/paneer_butter_masala").get_json()
    suggested = {s["token"] for i in body["ingredients"] for s in i["substitutes"]}
    assert "cheese" not in suggested


# --- Semantic recipe search -------------------------------------------------


def test_api_recipe_search_blank_query(client):
    resp = client.get("/api/recipes/search")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"query": "", "results": [], "count": 0, "semantic": False}


def test_api_recipe_search_returns_ranked_results(client):
    resp = client.get("/api/recipes/search", query_string={"q": "creamy paneer curry"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["query"] == "creamy paneer curry"
    assert body["semantic"] is True  # the trained embeddings ship with the repo
    assert body["count"] == len(body["results"]) > 0
    top = body["results"][0]
    assert "paneer" in top["title"].lower()
    assert {"id", "title", "score", "similarity", "matched", "ingredients"} <= set(top)


def test_api_recipe_search_clamps_k(client):
    resp = client.get("/api/recipes/search", query_string={"q": "curry", "k": "999"})
    assert resp.status_code == 200
    assert resp.get_json()["count"] <= 12  # k is clamped to the 1..12 window


def test_api_recipe_similar_returns_neighbors(client):
    resp = client.get("/api/recipes/paneer_butter_masala/similar")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["id"] == "paneer_butter_masala"
    ids = [r["id"] for r in body["results"]]
    assert ids and "paneer_butter_masala" not in ids  # a recipe is never similar to itself


def test_api_recipe_similar_unknown_returns_404(client):
    resp = client.get("/api/recipes/not-a-real-recipe/similar")
    assert resp.status_code == 404
    assert "error" in resp.get_json()
