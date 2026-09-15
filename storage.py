"""SQLite persistence for SmartFridge Vision (Phase 2).

Like :mod:`vision_service`, this module is deliberately **framework-agnostic**: it has
no Flask or HTTP coupling, so a future agent layer can read and write inventory history
directly. It records one *scan* per analysis (every upload/capture is timestamped) plus
the items and unidentified entries that scan produced.

The database is plain SQLite via the standard-library :mod:`sqlite3` — no server to run
and no extra dependency. The file location is read from the environment at call time::

    DATABASE_PATH   — path to the SQLite file (default: ``smartfridge.db``).

All timestamps are stored as UTC ISO-8601 strings (``YYYY-MM-DDTHH:MM:SSZ``); callers /
the UI convert to local time for display.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "smartfridge.db"

# A scan's ``source`` records how the image arrived.
SOURCE_UPLOAD = "upload"
SOURCE_WEBCAM = "webcam"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,               -- UTC ISO-8601, e.g. 2026-09-15T09:12:33Z
    source      TEXT    NOT NULL,               -- 'upload' | 'webcam'
    item_count  INTEGER NOT NULL DEFAULT 0      -- denormalized convenience count
);

CREATE TABLE IF NOT EXISTS items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id               INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    name                  TEXT    NOT NULL,
    count                 INTEGER NOT NULL DEFAULT 1,
    category              TEXT,
    freshness             TEXT,
    freshness_confidence  TEXT,
    notes                 TEXT
);

CREATE TABLE IF NOT EXISTS unidentified (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id      INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    description  TEXT,
    reason       TEXT
);

CREATE INDEX IF NOT EXISTS idx_items_scan        ON items(scan_id);
CREATE INDEX IF NOT EXISTS idx_unidentified_scan ON unidentified(scan_id);
CREATE INDEX IF NOT EXISTS idx_scans_created_at  ON scans(created_at);
"""


def _db_path() -> str:
    """Resolve the SQLite file path from the environment (read at call time)."""
    return os.getenv("DATABASE_PATH", DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open a connection with row access by name; commit on success, rollback on error."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the tables and indexes if they don't yet exist (idempotent)."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("SmartFridge history database ready at %s", _db_path())


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

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scans (created_at, source, item_count) VALUES (?, ?, ?)",
            (when, source, len(items)),
        )
        scan_id = int(cur.lastrowid)

        for item in items:
            conn.execute(
                "INSERT INTO items "
                "(scan_id, name, count, category, freshness, freshness_confidence, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    scan_id,
                    str(item.get("name", "")),
                    _as_int(item.get("count"), default=1),
                    _as_opt_str(item.get("category")),
                    _as_opt_str(item.get("freshness")),
                    _as_opt_str(item.get("freshness_confidence")),
                    _as_opt_str(item.get("notes")),
                ),
            )

        for entry in unidentified:
            conn.execute(
                "INSERT INTO unidentified (scan_id, description, reason) VALUES (?, ?, ?)",
                (
                    scan_id,
                    _as_opt_str(entry.get("description")),
                    _as_opt_str(entry.get("reason")),
                ),
            )

    logger.info("Saved scan %d (source=%s, %d items)", scan_id, source, len(items))
    return scan_id


def list_scans(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent scans (newest first), each with its items and unidentified.

    Args:
        limit: Maximum number of scans to return.

    Returns:
        A list of dicts: ``{id, created_at, source, item_count, items, unidentified}``.
    """
    with _connect() as conn:
        scan_rows = conn.execute(
            "SELECT id, created_at, source, item_count "
            "FROM scans ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
        return [_hydrate_scan(conn, scan) for scan in scan_rows]


def get_scan(scan_id: int) -> dict[str, Any] | None:
    """Return a single scan (with items + unidentified), or ``None`` if it doesn't exist."""
    with _connect() as conn:
        scan = conn.execute(
            "SELECT id, created_at, source, item_count FROM scans WHERE id = ?",
            (int(scan_id),),
        ).fetchone()
        return _hydrate_scan(conn, scan) if scan is not None else None


def get_latest_scan() -> dict[str, Any] | None:
    """Return the most recent scan (with items + unidentified), or ``None`` if empty.

    Each scan is a full snapshot of the fridge at one moment, so the newest scan *is*
    the current inventory — this is the "what's in my fridge now" view. A future rollup
    that reconciles across scans can replace this without changing callers.
    """
    with _connect() as conn:
        scan = conn.execute(
            "SELECT id, created_at, source, item_count "
            "FROM scans ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _hydrate_scan(conn, scan) if scan is not None else None


def _hydrate_scan(conn: sqlite3.Connection, scan: sqlite3.Row) -> dict[str, Any]:
    """Attach a scan row's items and unidentified entries, returning a plain dict."""
    item_rows = conn.execute(
        "SELECT name, count, category, freshness, freshness_confidence, notes "
        "FROM items WHERE scan_id = ? ORDER BY id",
        (scan["id"],),
    ).fetchall()
    unidentified_rows = conn.execute(
        "SELECT description, reason FROM unidentified WHERE scan_id = ? ORDER BY id",
        (scan["id"],),
    ).fetchall()
    return {
        "id": scan["id"],
        "created_at": scan["created_at"],
        "source": scan["source"],
        "item_count": scan["item_count"],
        "items": [dict(row) for row in item_rows],
        "unidentified": [dict(row) for row in unidentified_rows],
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
