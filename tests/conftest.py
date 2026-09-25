"""Shared pytest setup for the SmartFridge test suite.

This module runs **before any test module is imported**, which matters because importing
:mod:`app` initializes the database at import time. The env vars set here guarantee the
whole suite is hermetic and offline:

* the project ``.env`` is never loaded (``load_dotenv`` is replaced with a no-op), so no
  real API key or fallback provider leaks into a test;
* storage is forced into **local-SQLite** mode (a throwaway temp file), never the user's
  real Turso database. ``TURSO_DATABASE_URL`` is also set to an empty string, so even a
  real ``load_dotenv()`` (which does not override values already present) couldn't
  re-enable the remote target; and
* a dummy ``VISION_API_KEY`` is present so nothing trips the "unconfigured" path. No test
  actually calls the vision provider.
"""

from __future__ import annotations

import os
import tempfile

import dotenv

# --- Force a hermetic, offline environment BEFORE app/storage import anywhere ---
# The developer's real .env (API keys for every fallback provider, the Turso URL) must never
# reach the tests: app.py's ``from dotenv import load_dotenv`` picks up this no-op instead.
dotenv.load_dotenv = lambda *_args, **_kwargs: False
os.environ["TURSO_DATABASE_URL"] = ""
os.environ["TURSO_AUTH_TOKEN"] = ""
os.environ.setdefault("VISION_API_KEY", "test-key-unused")
# The proactive watcher (a background timer) and its optional LLM-written headline are off
# under test: briefings are generated deterministically, on demand, and never hit the network.
os.environ["AGENT_WATCH_MINUTES"] = "0"
os.environ["AGENT_BRIEFING_LLM"] = "0"
os.environ["AGENT_AUTO_BRIEFING"] = "0"  # no background re-checks racing the tests' DB
# "Today" for use-by math follows the household's zone; pin it so fixed-clock tests give the
# same answer on every machine.
os.environ["APP_TIMEZONE"] = "UTC"

_TMP_DIR = tempfile.mkdtemp(prefix="smartfridge-tests-")
os.environ["DATABASE_PATH"] = os.path.join(_TMP_DIR, "test.db")

import pytest  # noqa: E402 — must come after the env is pinned above

import agent  # noqa: E402
import storage  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    """Create the schema once for the whole session against the temp database."""
    storage.init_db()


@pytest.fixture(autouse=True)
def _fresh_model_health():
    """The agent remembers which models failed recently, process-wide; start each test clean."""
    agent._recent_failures.clear()


@pytest.fixture()
def clean_db():
    """Give a test an empty perishables + nutrition + waste-log + consumption slate, plus a
    clean agent state (memory, shopping list, action queue, briefings).

    (Scans stay append-only; ``get_current_inventory`` reconciles against the ledger.)
    """
    storage.clear_perishables()
    storage.clear_waste_events()
    storage.clear_consumption()
    storage.clear_memory()
    storage.clear_shopping()
    storage.clear_actions()
    storage.clear_briefings()
    with storage._connect() as client:  # test-only reach into the module
        client.execute("DELETE FROM nutrition")
    yield
