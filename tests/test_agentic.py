"""Tests for the agentic layer: memory, shopping list, confirm-gated actions, the offline
Chef, the proactive watcher, and the HTTP endpoints that tie them together.

Everything runs against the throwaway SQLite database from ``conftest`` and never touches
the network: the LLM transport is stubbed where the real agent loop is exercised, and the
watcher's optional LLM headline is disabled (``AGENT_BRIEFING_LLM=0``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

import agent
import app as webapp
import kitchen
import offline_agent
import preferences
import recommender
import storage
import watcher

# --- Fixtures & helpers -----------------------------------------------------------


FRIDGE = [
    {"name": "Spinach", "count": 1, "freshness": "use_soon"},
    {"name": "Paneer", "count": 1, "freshness": "fresh"},
    {"name": "Tomato", "count": 3, "freshness": "fresh"},
    {"name": "Onion", "count": 2, "freshness": "fresh"},
    {"name": "Garlic", "count": 1, "freshness": "fresh"},
    {"name": "Ginger", "count": 1, "freshness": "fresh"},
    {"name": "Green chili", "count": 2, "freshness": "fresh"},
    {"name": "Milk", "count": 1, "freshness": "spoiled"},
]


@pytest.fixture()
def fridge(clean_db):
    """A scanned fridge that can make Palak Paneer, with spoiled milk in it."""
    storage.save_scan({"items": [dict(i) for i in FRIDGE], "unidentified": []}, storage.SOURCE_UPLOAD)
    yield


@pytest.fixture()
def empty_fridge(clean_db):
    """An authoritative empty fridge.

    ``clean_db`` clears most tables but leaves earlier scans in place, so the
    "current inventory" still reflects whatever a prior test scanned. Writing a
    fresh empty scan makes it the latest ground truth, so absolute-count
    assertions in these tests aren't polluted by test ordering.
    """
    storage.save_scan({"items": [], "unidentified": []}, storage.SOURCE_UPLOAD)
    yield


@pytest.fixture()
def client():
    return webapp.app.test_client()


@pytest.fixture()
def offline(monkeypatch):
    """Make the Chef endpoint take the offline (rules) path deterministically."""
    monkeypatch.setattr(agent, "configured", lambda: False)


def _inventory() -> dict[str, int]:
    inv = storage.get_current_inventory(normalize=recommender.normalize_ingredient) or {}
    return {i["name"]: i.get("count") for i in inv.get("items") or []}


def _tools(**overrides):
    tools = webapp._agent_tools()
    tools.update(overrides)
    return tools


def _ask(client, message, **extra):
    resp = client.post("/api/agent", json={"message": message, **extra})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


# --- preferences ------------------------------------------------------------------


def test_diet_aliases_normalize():
    assert preferences.normalize("diet", "pure veg") == ("diet", "vegetarian")
    assert preferences.normalize("diet", "plant-based") == ("diet", "vegan")
    assert preferences.normalize("diet", "non veg") == ("diet", "any")
    with pytest.raises(ValueError):
        preferences.normalize("diet", "carnivore-ish")
    with pytest.raises(ValueError):
        preferences.canonical_key("favourite_colour")


def test_allergy_merge_appends_instead_of_replacing():
    _, first = preferences.merge({}, "allergies", "peanuts")
    _, both = preferences.merge({"allergies": first}, "allergies", "sesame")
    assert set(both) >= {"peanuts", "sesame"}


def test_build_filter_blocks_diet_and_allergens():
    assert preferences.build_filter({}) is None
    rule = preferences.build_filter({"diet": "vegetarian", "allergies": ["dairy"]})
    assert rule({"ingredients": ["rice", "dal"], "tags": []})
    assert not rule({"ingredients": ["chicken", "onion"], "tags": ["non-veg"]})
    assert not rule({"ingredients": ["paneer", "spinach"], "tags": []})  # dairy group
    assert preferences.has_allergen({"ingredients": ["paneer"], "tags": []}, {"allergies": ["dairy"]})
    assert not preferences.has_allergen({"ingredients": ["onion"], "tags": []}, {"dislikes": ["onion"]})


# --- storage: agent tables ----------------------------------------------------------


def test_memory_roundtrip(clean_db):
    storage.set_memory("diet", "vegan")
    storage.set_memory("allergies", ["peanut"])
    assert storage.get_memory() == {"diet": "vegan", "allergies": ["peanut"]}
    assert storage.delete_memory("diet") is True
    assert storage.delete_memory("diet") is False
    assert storage.clear_memory() == 1


def test_shopping_dedupes_open_items_by_token(clean_db):
    added = storage.add_shopping_items([{"name": "Milk"}, {"name": "milk"}, {"name": "Eggs"}])
    assert [row["name"] for row in added] == ["Milk", "Eggs"]
    assert storage.add_shopping_items([{"name": "MILK"}]) == []
    storage.set_shopping_done(added[0]["id"], True)
    # Once bought, the same item can go back on the list.
    assert [row["name"] for row in storage.add_shopping_items([{"name": "Milk"}])] == ["Milk"]
    assert storage.clear_shopping(done_only=True) == 1


def test_claim_action_is_single_winner(clean_db):
    action = storage.create_action("add_shopping", {"items": [{"name": "Rice"}]}, "Add rice")
    assert action["status"] == storage.ACTION_PENDING
    assert storage.claim_action(action["id"])["status"] == storage.ACTION_RUNNING
    assert storage.claim_action(action["id"]) is None  # a double-click can't run it twice
    storage.finish_action(action["id"], storage.ACTION_DONE, {"ok": True})
    assert storage.get_action(action["id"])["status"] == storage.ACTION_DONE


def test_expire_actions_only_touches_one_source(clean_db):
    chat = storage.create_action("add_shopping", {"items": []}, "chat one", source="chat")
    brief = storage.create_action("add_shopping", {"items": []}, "brief one", source=watcher.SOURCE)
    assert storage.expire_actions(source=watcher.SOURCE) == 1
    assert storage.get_action(brief["id"])["status"] == storage.ACTION_EXPIRED
    assert storage.get_action(chat["id"])["status"] == storage.ACTION_PENDING


def test_briefing_latest_and_dismiss(clean_db):
    saved = storage.save_briefing("manual", "Hello", {"lines": ["x"]}, "abc")
    assert storage.latest_briefing()["id"] == saved["id"]
    assert storage.dismiss_briefing(saved["id"]) is True
    assert storage.latest_briefing() is None
    assert storage.latest_briefing(include_dismissed=True)["id"] == saved["id"]


# --- kitchen executors ----------------------------------------------------------------


def test_discard_items_removes_from_fridge_and_logs_waste(fridge):
    result = kitchen.discard_items(kitchen.cook_entries({"items": [{"name": "Milk", "qty": 1}]}))
    assert result["discarded"] == ["Milk"]
    assert "Milk" not in _inventory()
    events = storage.list_waste_events()
    assert [(e["name"], e["event"]) for e in events] == [("Milk", storage.WASTE_WASTED)]


def test_discard_items_rejects_empty(clean_db):
    with pytest.raises(kitchen.KitchenError):
        kitchen.discard_items([])


def test_execute_refuses_unknown_tool(clean_db):
    with pytest.raises(kitchen.KitchenError):
        kitchen.execute("drop_table", {})


# --- kitchen: receipt import + leftovers ----------------------------------------------


def test_import_receipt_adds_new_items_to_an_empty_fridge(empty_fridge):
    result = kitchen.import_receipt([{"name": "Tomato", "qty": 2}, {"name": "Paneer"}])
    assert result["added"] == 2
    assert result["units"] == 3
    inv = _inventory()
    assert inv.get("Tomato") == 2
    assert inv.get("Paneer") == 1


def test_import_receipt_bumps_the_count_of_an_item_already_on_hand(fridge):
    before = _inventory().get("Tomato", 0)  # the fridge fixture starts with 3 tomatoes
    kitchen.import_receipt([{"name": "Tomato", "qty": 2}])
    assert _inventory().get("Tomato") == before + 2


def test_import_receipt_merges_on_canonical_token(empty_fridge):
    # "Amul Milk" and "Milk" are the same ingredient — they should land as one stack.
    kitchen.import_receipt([{"name": "Milk", "qty": 1}, {"name": "Amul Milk", "qty": 2}])
    milk = [c for n, c in _inventory().items() if recommender.normalize_ingredient(n) == "milk"]
    assert milk == [3]


def test_import_receipt_rejects_empty(clean_db):
    with pytest.raises(kitchen.KitchenError):
        kitchen.import_receipt([])


def test_import_receipt_is_reachable_through_the_executor(empty_fridge):
    result = kitchen.execute("import_receipt", {"items": [{"name": "Onion", "qty": 1}]})
    assert "inventory" in result["scopes"]
    assert _inventory().get("Onion") == 1


def test_save_leftovers_tracks_a_perishable_with_a_use_by(clean_db):
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)
    result = kitchen.save_leftovers("Palak Paneer", days=3, now=now)
    assert result["perishable"]["name"] == "Leftover Palak Paneer"
    assert result["perishable"]["use_by"] == "2026-09-28"
    names = [p["name"] for p in storage.list_perishables()]
    assert "Leftover Palak Paneer" in names


def test_save_leftovers_does_not_double_prefix(clean_db):
    result = kitchen.save_leftovers("Leftover Dal", days=2)
    assert result["perishable"]["name"] == "Leftover Dal"


def test_save_leftovers_clamps_days_and_rejects_blank(clean_db):
    assert kitchen.save_leftovers("Rice", days=999)["days"] == kitchen.MAX_LEFTOVER_DAYS
    assert kitchen.save_leftovers("Rice", days=0)["days"] == 1
    with pytest.raises(kitchen.KitchenError):
        kitchen.save_leftovers("")


# --- agent tools (real storage behind them) -------------------------------------------


def test_remember_preference_tool_writes_memory(clean_db):
    tools = _tools()
    result = tools["remember_preference"].run({"key": "allergies", "value": "peanuts"})
    assert result.get("saved")
    assert "peanuts" in storage.get_memory()["allergies"]
    assert tools["remember_preference"].kind == agent.WRITE


def test_shopping_tool_refuses_allergens(clean_db):
    storage.set_memory("allergies", ["peanuts"])
    result = _tools()["add_to_shopping_list"].run({"items": [{"name": "Peanuts"}, {"name": "Rice"}]})
    assert result["refused_for_allergy"] == ["Peanuts"]
    assert [row["name"] for row in storage.list_shopping()] == ["Rice"]


def test_cook_proposal_does_not_touch_the_fridge(fridge):
    before = _inventory()
    result = _tools()["propose_cook_meal"].run({"recipe": "palak paneer"})
    action = result["action"]
    assert action["status"] == storage.ACTION_PENDING and action["tool"] == "cook_meal"
    assert _inventory() == before  # proposing never changes anything


def test_cook_proposal_blocked_for_allergy(fridge):
    storage.set_memory("allergies", ["dairy"])
    result = _tools()["propose_cook_meal"].run({"recipe": "palak paneer"})
    assert result["error"].startswith("BLOCKED")
    assert storage.list_actions(status=storage.ACTION_PENDING) == []


def test_same_proposal_twice_in_one_run_files_once(fridge):
    tools = _tools()
    first = tools["propose_cook_meal"].run({"recipe": "palak paneer"})
    second = tools["propose_cook_meal"].run({"recipe": "palak paneer"})
    assert first["action"]["id"] == second["action"]["id"]
    assert len(storage.list_actions(status=storage.ACTION_PENDING)) == 1


def test_log_item_falls_back_to_scanned_items(fridge):
    tools = _tools()
    wasted = tools["propose_log_item"].run({"name": "milk", "outcome": "wasted"})
    assert wasted["action"]["tool"] == "discard_items"
    used = tools["propose_log_item"].run({"name": "ginger", "outcome": "used"})
    assert used["action"]["tool"] == "cook_meal"
    missing = tools["propose_log_item"].run({"name": "caviar", "outcome": "used"})
    assert "isn't in the fridge" in missing["error"]


def test_log_item_prefers_tracked_perishable(fridge):
    storage.add_perishable("Yogurt", recommender.local_today().isoformat())
    result = _tools()["propose_log_item"].run({"name": "yogurt", "outcome": "wasted"})
    assert result["action"]["tool"] == "resolve_perishable"


# --- run_agent with a stubbed model -----------------------------------------------------


def test_run_agent_files_proposal_and_streams_steps(fridge, monkeypatch):
    replies = [
        {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "propose_cook_meal", "arguments": json.dumps({"recipe": "palak paneer"})}}]},
        {"content": "Ready — tap Confirm.", "tool_calls": None},
    ]
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(agent, "_post_chat",
                        lambda base, key, payload: {"choices": [{"message": replies.pop(0)}]})
    events = []
    result = agent.run_agent("I cooked palak paneer", tools=_tools(), mode="tools",
                             on_step=events.append)

    assert result["answer"] == "Ready — tap Confirm."
    assert [a["tool"] for a in result["actions"]] == ["cook_meal"]
    assert result["changed"] == []
    phases = [(e["phase"], e.get("status")) for e in events]
    assert ("thinking", None) in phases
    assert ("tool", "running") in phases and ("tool", "done") in phases
    assert "Paneer" in _inventory()  # still only a proposal


# --- offline Chef ------------------------------------------------------------------------


def _offline(message):
    return offline_agent.run_offline(message, tools=_tools())


def test_offline_remembers_diet_and_allergy(clean_db):
    result = _offline("I'm vegetarian and allergic to peanuts")
    assert result["mode"] == "offline"
    assert result["changed"] == ["memory"]
    memory = storage.get_memory()
    assert memory["diet"] == "vegetarian" and "peanuts" in memory["allergies"]


def test_offline_shopping_add_and_show(clean_db):
    _offline("add milk and eggs to my shopping list")
    assert [row["name"] for row in storage.list_shopping()] == ["milk", "eggs"]
    assert "milk" in _offline("show my shopping list")["answer"]


@pytest.mark.parametrize(("message", "tool"), [
    ("I threw away the milk", "discard_items"),
    ("I finished the ginger", "cook_meal"),
    ("I cooked palak paneer", "cook_meal"),
    ("track chicken use by in 3 days", "track_expiry"),
])
def test_offline_fridge_changes_are_only_proposed(fridge, message, tool):
    before = _inventory()
    result = _offline(message)
    assert [a["tool"] for a in result["actions"]] == [tool], result["answer"]
    assert "Confirm" in result["answer"]
    assert _inventory() == before


def test_offline_custom_dish_with_ingredients(fridge):
    result = _offline("I made my special curry with tomato and onion")
    assert [a["tool"] for a in result["actions"]] == ["cook_meal"]
    assert _offline("I ate the caviar")["actions"] == []


@pytest.mark.parametrize(("message", "tool"), [
    ("what's expiring?", "get_recommendations"),
    ("plan my meals for 3 days", "plan_meals"),
    ("how much food am I wasting?", "get_waste_report"),
    ("recipes with paneer", "search_recipes"),
    ("find me a quick spicy dinner", "search_recipes"),
    ("what's in my fridge?", "get_inventory"),
    ("what can I cook?", "get_recommendations"),
])
def test_offline_routes_questions(fridge, message, tool):
    result = _offline(message)
    assert tool in [s["tool"] for s in result["steps"]]
    assert result["answer"]


# --- proactive watcher ---------------------------------------------------------------


def test_briefing_files_discard_cook_actions(fridge):
    storage.add_perishable("Yogurt", (recommender.local_today() - timedelta(days=1)).isoformat())
    brief = watcher.generate_briefing("scan")
    assert brief is not None
    tools = {a["tool"] for a in brief["body"]["actions"]}
    assert {"discard_items", "cook_meal"} <= tools
    pending = storage.list_actions(status=storage.ACTION_PENDING)
    assert all(a["source"] == watcher.SOURCE for a in pending)
    assert "Milk" in brief["headline"]
    assert brief["body"]["worded_by"] == "template"


def test_briefing_nudges_restock_for_a_depleted_staple(clean_db):
    # A fresh scan with no Paneer (the latest scan is authoritative), but the waste log
    # shows Paneer is used a lot → nothing on hand, so it's "out".
    storage.save_scan(
        {"items": [{"name": "Rice", "count": 1, "freshness": "fresh"}], "unidentified": []},
        storage.SOURCE_UPLOAD,
    )
    for _ in range(3):
        storage.log_waste_event("Paneer", storage.WASTE_USED)

    brief = watcher.generate_briefing("scan")
    assert brief is not None  # an "out" staple is worth a briefing on its own
    facts = brief["body"]["facts"]
    assert any(r["token"] == "paneer" and r["status"] == "out" for r in facts["restock"])

    # It files a confirm-gated add_shopping action for the depleted staple.
    add = [a for a in brief["body"]["actions"] if a["tool"] == "add_shopping"]
    assert add, "expected a restock add_shopping action"
    pending = storage.list_actions(status=storage.ACTION_PENDING)
    paneer_action = next(
        a for a in pending
        if a["tool"] == "add_shopping"
        and any(i["name"].lower() == "paneer" for i in a["arguments"]["items"])
    )
    assert paneer_action["source"] == watcher.SOURCE


def test_briefing_restock_skips_items_on_hand(clean_db):
    # Paneer is used moderately but well stocked (3 on hand) → no restock nudge.
    storage.save_scan(
        {"items": [{"name": "Paneer", "count": 3, "freshness": "fresh"}], "unidentified": []},
        storage.SOURCE_UPLOAD,
    )
    for _ in range(3):
        storage.log_waste_event("Paneer", storage.WASTE_USED)
    brief = watcher.generate_briefing("manual", force=True)
    facts = brief["body"]["facts"]
    assert all(r["token"] != "paneer" for r in facts["restock"])


def test_briefing_dedupes_unchanged_facts(fridge):
    assert watcher.generate_briefing("timer") is not None
    assert watcher.generate_briefing("timer") is None          # nothing new
    forced = watcher.generate_briefing("manual", force=True)    # Check now always answers
    assert forced is not None
    # The forced refresh expired the old buttons and filed fresh ones.
    assert len({a["id"] for a in forced["body"]["actions"]}) == len(forced["body"]["actions"])
    assert all(a["source"] == watcher.SOURCE
               for a in storage.list_actions(status=storage.ACTION_PENDING))


def test_llm_headline_is_plain_text_and_optional(monkeypatch):
    monkeypatch.setenv("AGENT_BRIEFING_LLM", "1")
    monkeypatch.setattr(agent, "configured", lambda: True)
    monkeypatch.setattr(agent, "complete", lambda prompt, system: '"Use the **spinach** tonight."')
    assert watcher._polish("draft", ["Spinach: starting to turn."]) == "Use the spinach tonight."

    def down(prompt, system):
        raise agent.AgentAPIError("every model failed")

    monkeypatch.setattr(agent, "complete", down)
    assert watcher._polish("draft", ["Spinach: starting to turn."]) is None  # keeps template


def test_briefing_clears_itself_when_nothing_is_urgent(clean_db):
    storage.save_scan({"items": [{"name": "Tomato", "count": 1, "freshness": "use_soon"}],
                       "unidentified": []}, storage.SOURCE_UPLOAD)
    assert watcher.generate_briefing("scan") is not None
    storage.save_scan({"items": [{"name": "Rice", "count": 1, "freshness": "fresh"}],
                       "unidentified": []}, storage.SOURCE_UPLOAD)
    assert watcher.generate_briefing("scan") is None
    assert storage.latest_briefing() is None
    assert storage.list_actions(status=storage.ACTION_PENDING) == []


def test_briefing_respects_allergies(fridge):
    storage.set_memory("allergies", ["dairy"])
    brief = watcher.generate_briefing("manual", force=True)
    facts = brief["body"]["facts"]
    assert (facts["cook"] or {}).get("id") != "palak_paneer"
    for action in storage.list_actions(status=storage.ACTION_PENDING):
        if action["tool"] == "add_shopping":
            names = [i["name"].lower() for i in action["arguments"]["items"]]
            assert not set(names) & preferences.DAIRY


def test_watch_minutes_env(monkeypatch):
    monkeypatch.setenv("AGENT_WATCH_MINUTES", "15")
    assert watcher.watch_minutes() == 15
    monkeypatch.setenv("AGENT_WATCH_MINUTES", "nope")
    assert watcher.watch_minutes() == watcher.DEFAULT_WATCH_MINUTES
    monkeypatch.setenv("AGENT_WATCH_MINUTES", "0")
    assert watcher.watch_minutes() == 0


# --- HTTP: confirm / cancel ----------------------------------------------------------


def test_confirm_applies_cook_once(fridge, client, offline):
    action = _ask(client, "I cooked palak paneer")["actions"][0]
    assert "Paneer" in _inventory()

    resp = client.post(f"/api/agent/actions/{action['id']}/confirm")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["action"]["status"] == storage.ACTION_DONE
    assert body["result"]["cooked"] is True
    inventory = _inventory()
    assert "Paneer" not in inventory and inventory["Tomato"] == 2

    again = client.post(f"/api/agent/actions/{action['id']}/confirm")
    assert again.status_code == 409
    assert _inventory() == inventory


def test_cancel_leaves_fridge_alone(fridge, client, offline):
    action = _ask(client, "I threw away the milk")["actions"][0]
    resp = client.post(f"/api/agent/actions/{action['id']}/cancel")
    assert resp.status_code == 200 and resp.get_json()["cancelled"] is True
    assert "Milk" in _inventory()
    assert client.post(f"/api/agent/actions/{action['id']}/confirm").status_code == 409
    assert client.post(f"/api/agent/actions/{action['id']}/cancel").status_code == 409


def test_confirm_missing_action_is_404(clean_db, client):
    assert client.post("/api/agent/actions/99999/confirm").status_code == 404
    assert client.post("/api/agent/actions/99999/cancel").status_code == 404


def test_stale_action_expires_instead_of_running(fridge, client):
    old = datetime.now(UTC) - webapp.ACTION_MAX_AGE - timedelta(minutes=5)
    action = storage.create_action("discard_items", {"items": [{"name": "Milk", "qty": 1}]},
                                   "Throw out milk", created_at=old)
    resp = client.post(f"/api/agent/actions/{action['id']}/confirm")
    assert resp.status_code == 409
    assert storage.get_action(action["id"])["status"] == storage.ACTION_EXPIRED
    assert "Milk" in _inventory()


def test_confirm_failure_marks_action_failed(clean_db, client):
    action = storage.create_action("discard_items", {"items": []}, "Nothing to throw out")
    resp = client.post(f"/api/agent/actions/{action['id']}/confirm")
    assert resp.status_code == 400
    assert storage.get_action(action["id"])["status"] == storage.ACTION_FAILED


def test_actions_listing(fridge, client, offline):
    _ask(client, "I threw away the milk")
    pending = client.get("/api/agent/actions").get_json()["actions"]
    assert [a["tool"] for a in pending] == ["discard_items"]
    assert client.get("/api/agent/actions?status=all").status_code == 200


# --- HTTP: chat endpoint -----------------------------------------------------------------


def test_agent_endpoint_validates_message(client):
    assert client.post("/api/agent", json={"message": ""}).status_code == 400
    assert client.post("/api/agent", json={"message": "x" * 1001}).status_code == 400


def test_agent_endpoint_falls_back_when_model_is_down(fridge, client, monkeypatch):
    def down(*_a, **_k):
        raise agent.AgentAPIError("provider down")

    monkeypatch.setattr(agent, "configured", lambda: True)
    monkeypatch.setattr(agent, "run_agent", down)
    body = _ask(client, "what's in my fridge?")
    assert body["mode"] == "offline"
    assert body["note"]
    assert "Paneer" in body["answer"]


# --- HTTP: memory, shopping, briefing ------------------------------------------------


def test_memory_endpoints(clean_db, client):
    assert client.post("/api/agent/memory", json={"key": "diet", "value": "veg"}).status_code == 200
    resp = client.post("/api/agent/memory", json={"key": "allergies", "value": "sesame"})
    assert resp.get_json()["memory"]["allergies"] == ["sesame"]
    got = client.get("/api/agent/memory").get_json()
    assert got["memory"]["diet"] == "vegetarian"
    assert got["diets"][0] == "any" and "vegetarian" in got["summary"]
    assert client.post("/api/agent/memory", json={"key": "shoe_size", "value": "9"}).status_code == 400

    client.delete("/api/agent/memory?key=allergies&value=sesame")
    assert "allergies" not in client.get("/api/agent/memory").get_json()["memory"] or \
        client.get("/api/agent/memory").get_json()["memory"]["allergies"] == []
    client.delete("/api/agent/memory")
    assert client.get("/api/agent/memory").get_json()["memory"] == {}


def test_shopping_endpoints(clean_db, client):
    storage.set_memory("allergies", ["peanuts"])
    resp = client.post("/api/shopping", json={"items": ["Rice", {"name": "Peanuts"}]})
    assert resp.status_code == 201
    assert resp.get_json()["refused_for_allergy"] == ["Peanuts"]
    assert client.post("/api/shopping", json={"name": "Dal", "qty": "1 kg"}).status_code == 201

    items = client.get("/api/shopping").get_json()["items"]
    assert [i["name"] for i in items] == ["Rice", "Dal"]
    rice = items[0]["id"]
    assert client.patch(f"/api/shopping/{rice}", json={"done": True}).status_code == 200
    assert client.get("/api/shopping").get_json()["items"][-1]["done"] in (1, True)
    assert client.delete("/api/shopping?done=1").status_code == 200
    assert [i["name"] for i in client.get("/api/shopping").get_json()["items"]] == ["Dal"]
    dal = client.get("/api/shopping").get_json()["items"][0]["id"]
    assert client.delete(f"/api/shopping/{dal}").status_code == 200
    assert client.delete(f"/api/shopping/{dal}").status_code == 404
    assert client.patch("/api/shopping/99999", json={"done": True}).status_code == 404


def test_briefing_endpoints(fridge, client):
    assert client.get("/api/agent/briefing").get_json()["briefing"] is None
    brief = client.post("/api/agent/briefing").get_json()["briefing"]
    assert brief["headline"]
    assert brief["actions"] and all(a["status"] == storage.ACTION_PENDING for a in brief["actions"])

    # Confirming one briefing action takes it off the banner.
    first = brief["actions"][0]
    assert client.post(f"/api/agent/actions/{first['id']}/confirm").status_code == 200
    remaining = client.get("/api/agent/briefing").get_json()["briefing"]["actions"]
    assert first["id"] not in [a["id"] for a in remaining]

    resp = client.post(f"/api/agent/briefing/{brief['id']}/dismiss")
    assert resp.status_code == 200
    assert client.get("/api/agent/briefing").get_json()["briefing"] is None
    assert all(a["status"] != storage.ACTION_PENDING
               for a in storage.list_actions(status=storage.ACTION_PENDING)
               if a["source"] == watcher.SOURCE)
    assert client.post("/api/agent/briefing/99999/dismiss").status_code == 404
