"""Real-time (Socket.IO) integration tests.

These use flask-socketio's in-process test client to confirm that the socket layer connects
cleanly and that mutating HTTP routes broadcast a ``state_changed`` event carrying the right
view scopes. Storage is the hermetic temp database from ``conftest.py``.
"""

from __future__ import annotations

import pytest

import app as app_module


@pytest.fixture()
def clients():
    """A paired HTTP test client + a connected Socket.IO test client on the same app."""
    app_module.app.config.update(TESTING=True)
    http = app_module.app.test_client()
    sio = app_module.socketio.test_client(app_module.app, flask_test_client=http)
    assert sio.is_connected()
    sio.get_received()  # drain anything queued on connect
    yield http, sio
    if sio.is_connected():
        sio.disconnect()


def _scope_sets(received):
    """Every ``state_changed`` event's scopes (as sets) from a get_received() dump."""
    out = []
    for msg in received:
        if msg.get("name") == "state_changed":
            args = msg.get("args") or [{}]
            out.append(set(args[0].get("scopes", [])))
    return out


def test_socket_connects(clients):
    _http, sio = clients
    assert sio.is_connected()


def test_waste_broadcasts_analytics_scope(clients):
    http, sio = clients
    resp = http.post("/api/waste", json={"name": "Spinach", "event": "wasted"})
    assert resp.status_code == 201
    assert any("analytics" in scopes for scopes in _scope_sets(sio.get_received()))


def test_add_perishable_broadcasts_perishables_scope(clients):
    http, sio = clients
    resp = http.post("/api/perishables", json={"name": "Milk", "use_by": "2026-10-10"})
    assert resp.status_code == 201
    assert any("perishables" in scopes for scopes in _scope_sets(sio.get_received()))


def test_cook_broadcasts_inventory_analytics_and_perishables(clients):
    http, sio = clients
    resp = http.post("/api/cook", json={"uses": ["Paneer"], "title": "Paneer Bhurji"})
    assert resp.status_code == 201
    assert any(
        {"inventory", "analytics", "perishables"} <= scopes
        for scopes in _scope_sets(sio.get_received())
    )


def test_read_only_route_does_not_broadcast(clients):
    http, sio = clients
    assert http.get("/api/analytics").status_code == 200
    assert _scope_sets(sio.get_received()) == []  # GETs never push a change hint
