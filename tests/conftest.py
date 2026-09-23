"""Shared pytest setup for the SmartFridge test suite.

This module runs **before any test module is imported**, which matters because importing
:mod:`app` initializes the database at import time. The env vars set here guarantee the
whole suite is hermetic and offline:

* storage is forced into **local-SQLite** mode (a throwaway temp file), never the user's
  real Turso database — even though the project ``.env`` may point at Turso. We set
  ``TURSO_DATABASE_URL`` to an empty string so ``load_dotenv()`` (which does not override
  values already present in the environment) can't re-enable the remote target; and
* a dummy ``VISION_API_KEY`` is present so nothing trips the "unconfigured" path. No test
  actually calls the vision provider.
"""

from __future__ import annotations

import os
import tempfile

# --- Force a hermetic, offline environment BEFORE app/storage import anywhere ---
os.environ["TURSO_DATABASE_URL"] = ""
os.environ["TURSO_AUTH_TOKEN"] = ""
os.environ.setdefault("VISION_API_KEY", "test-key-unused")

_TMP_DIR = tempfile.mkdtemp(prefix="smartfridge-tests-")
os.environ["DATABASE_PATH"] = os.path.join(_TMP_DIR, "test.db")

import pytest  # noqa: E402 — must come after the env is pinned above

import storage  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_db():
    """Create the schema once for the whole session against the temp database."""
    storage.init_db()


@pytest.fixture()
def clean_db():
    """Give a test an empty perishables + nutrition + waste-log + consumption slate.

    (Scans stay append-only; ``get_current_inventory`` reconciles against the ledger.)
    """
    storage.clear_perishables()
    storage.clear_waste_events()
    storage.clear_consumption()
    with storage._connect() as client:  # test-only reach into the module
        client.execute("DELETE FROM nutrition")
    yield
