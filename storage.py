"""Inventory-history persistence for SmartFridge Vision (Phase 2).

Like :mod:`vision_service`, this module is deliberately **framework-agnostic**: it has
no Flask or HTTP coupling, so a future agent layer can read and write inventory history
directly. It records one *scan* per analysis (every upload/capture is timestamped) plus
the items and unidentified entries that scan produced.

Storage is **libSQL** (a SQLite fork) via the ``libsql-client`` package, which lets the
exact same code run against either:

* a **local SQLite file** for development — the default; or
* a hosted **Turso** database for cloud deploys (e.g. Vercel), where the local
  filesystem is ephemeral/read-only and a real database is needed for data to persist.

Which one is used is decided from the environment at call time:

    TURSO_DATABASE_URL  — if set, connect to that Turso/libSQL database (remote).
    TURSO_AUTH_TOKEN    — auth token for the Turso database (used with the URL above).
    DATABASE_PATH       — local SQLite file path when no Turso URL is set
                          (default: ``smartfridge.db``).

All timestamps are stored as UTC ISO-8601 strings (``YYYY-MM-DDTHH:MM:SSZ``); callers /
the UI convert to local time for display.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import libsql_client

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "smartfridge.db"

# A scan's ``source`` records how the image arrived.
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"

# Schema as discrete statements — libsql-client executes one statement per call (there is
# no ``executescript``), so the DDL is kept as a list rather than one semicolon-joined blob.
_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS scans (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT    NOT NULL,               -- UTC ISO-8601, e.g. 2026-09-15T09:12:33Z
        source      TEXT    NOT NULL,               -- 'upload' | 'webcam'
        item_count  INTEGER NOT NULL DEFAULT 0      -- denormalized convenience count
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS items (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id               INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
        name                  TEXT    NOT NULL,
        count                 INTEGER NOT NULL DEFAULT 1,
        category              TEXT,
        freshness             TEXT,
        freshness_confidence  TEXT,
        notes                 TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS unidentified (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id      INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
        description  TEXT,
        reason       TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_items_scan        ON items(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_unidentified_scan ON unidentified(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_scans_created_at  ON scans(created_at)",
]


def _db_path() -> str:
    """Resolve the local SQLite file path from the environment (read at call time)."""
    return os.getenv("DATABASE_PATH", DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH


def _turso_url() -> str:
    """Return the configured Turso/libSQL URL, or ``""`` to use the local file."""
    return os.getenv("TURSO_DATABASE_URL", "").strip()


def _make_client() -> libsql_client.ClientSync:
    """Create a libSQL client for the configured target (remote Turso or local file)."""
    url = _turso_url()
    if url:
        token = os.getenv("TURSO_AUTH_TOKEN", "").strip() or None
        return libsql_client.create_client_sync(url, auth_token=token)
    # Local file mode: libsql wants a forward-slashed ``file:`` URL, absolute so it doesn't
    # depend on the process's working directory.
    file_url = "file:" + os.path.abspath(_db_path()).replace("\\", "/")
    return libsql_client.create_client_sync(file_url)


def _target_desc() -> str:
    """A log-safe description of the storage target (never includes the auth token)."""
    url = _turso_url()
    if url:
        host = url.split("://", 1)[-1].split("/", 1)[0]
        return f"Turso ({host})"
    return _db_path()


@contextmanager
def _connect() -> Iterator[libsql_client.ClientSync]:
    """Yield a libSQL client, always closing it afterwards."""
    client = _make_client()
    try:
        yield client
    finally:
        client.close()


def _rows_to_dicts(result_set: libsql_client.ResultSet) -> list[dict[str, Any]]:
    """Convert a libSQL result set into a list of plain ``{column: value}`` dicts.

    libSQL ``Row`` objects support name/index access but are not directly ``dict()``-able,
    so we zip them with the result's column names.
    """
    columns = result_set.columns
    return [dict(zip(columns, row)) for row in result_set.rows]


def init_db() -> None:
    """Create the tables and indexes if they don't yet exist (idempotent)."""
    with _connect() as client:
        for statement in _SCHEMA_STATEMENTS:
            client.execute(statement)
    logger.info("SmartFridge history database ready at %s", _target_desc())


def save_scan(
    result: dict[str, Any],
    source: str,
    created_at: datetime | None = None,
) -> int:
    """Persist one analysis result as a timestamped scan and return its new id.

    Args:
        result: The normalized ``{"items": [...], "unidentified": [...]}`` dict from
            :func:`vision_service.analyze_image`.
        source: How the image arrived — :data:`SOURCE_UPLOAD` or :data:`SOURCE_WEBCAM`.
        created_at: Override timestamp (UTC). Defaults to now.

    Returns:
        The auto-generated ``scans.id`` of the recorded scan.
    """
    items = result.get("items") or []
    unidentified = result.get("unidentified") or []
    when = (created_at or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    with _connect() as client:
        # One interactive transaction so a scan and its rows are written atomically.
        tx = client.transaction()
        try:
            head = tx.execute(
                "INSERT INTO scans (created_at, source, item_count) VALUES (?, ?, ?)",
                [when, source, len(items)],
            )
            scan_id = int(head.last_insert_rowid)

            for item in items:
                tx.execute(
                    "INSERT INTO items "
                    "(scan_id, name, count, category, freshness, freshness_confidence, notes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        scan_id,
                        str(item.get("name", "")),
                        _as_int(item.get("count"), default=1),
                        _as_opt_str(item.get("category")),
                        _as_opt_str(item.get("freshness")),
                        _as_opt_str(item.get("freshness_confidence")),
                        _as_opt_str(item.get("notes")),
                    ],
                )

            for entry in unidentified:
                tx.execute(
                    "INSERT INTO unidentified (scan_id, description, reason) VALUES (?, ?, ?)",
                    [
                        scan_id,
                        _as_opt_str(entry.get("description")),
                        _as_opt_str(entry.get("reason")),
                    ],
                )
            tx.commit()
        except Exception:
            tx.rollback()
            raise

    logger.info("Saved scan %d (source=%s, %d items)", scan_id, source, len(items))
    return scan_id


def list_scans(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent scans (newest first), each with its items and unidentified.

    Args:
        limit: Maximum number of scans to return.

    Returns:
        A list of dicts: ``{id, created_at, source, item_count, items, unidentified}``.
    """
    with _connect() as client:
        scans = _rows_to_dicts(
            client.execute(
                "SELECT id, created_at, source, item_count "
                "FROM scans ORDER BY id DESC LIMIT ?",
                [max(1, int(limit))],
            )
        )
        return [_hydrate_scan(client, scan) for scan in scans]


def get_scan(scan_id: int) -> dict[str, Any] | None:
    """Return a single scan (with items + unidentified), or ``None`` if it doesn't exist."""
    with _connect() as client:
        scans = _rows_to_dicts(
            client.execute(
                "SELECT id, created_at, source, item_count FROM scans WHERE id = ?",
                [int(scan_id)],
            )
        )
        return _hydrate_scan(client, scans[0]) if scans else None


def get_latest_scan() -> dict[str, Any] | None:
    """Return the most recent scan (with items + unidentified), or ``None`` if empty.

    Each scan is a full snapshot of the fridge at one moment, so the newest scan *is*
    the current inventory — this is the "what's in my fridge now" view. A future rollup
    that reconciles across scans can replace this without changing callers.
    """
    with _connect() as client:
        scans = _rows_to_dicts(
            client.execute(
                "SELECT id, created_at, source, item_count "
                "FROM scans ORDER BY id DESC LIMIT 1"
            )
        )
        return _hydrate_scan(client, scans[0]) if scans else None


def _hydrate_scan(
    client: libsql_client.ClientSync, scan: dict[str, Any]
) -> dict[str, Any]:
    """Attach a scan row's items and unidentified entries, returning a plain dict."""
    items = _rows_to_dicts(
        client.execute(
            "SELECT name, count, category, freshness, freshness_confidence, notes "
            "FROM items WHERE scan_id = ? ORDER BY id",
            [scan["id"]],
        )
    )
    unidentified = _rows_to_dicts(
        client.execute(
            "SELECT description, reason FROM unidentified WHERE scan_id = ? ORDER BY id",
            [scan["id"]],
        )
    )
    return {
        "id": scan["id"],
        "created_at": scan["created_at"],
        "source": scan["source"],
        "item_count": scan["item_count"],
        "items": items,
        "unidentified": unidentified,
    }


# --- Small coercion helpers -------------------------------------------------


def _as_int(value: Any, default: int) -> int:
    """Coerce a value to int, falling back to ``default`` on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_opt_str(value: Any) -> str | None:
    """Return a stripped string, or ``None`` for missing/empty values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
