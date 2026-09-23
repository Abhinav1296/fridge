"""Persistence for SmartFridge Vision — inventory history, "use by" dates, nutrition cache.

Like :mod:`vision_service`, this module is deliberately **framework-agnostic**: it has
no Flask or HTTP coupling, so a future agent layer can read and write inventory history
directly. It records one *scan* per analysis (every upload/capture is timestamped) plus
the items and unidentified entries that scan produced. It also stores the user's tracked
"use by" dates and a per-ingredient nutrition cache the meal-plan optimizer reads.

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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

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
    # User-entered "use by" dates for perishables (paneer, milk, ...). This replaces
    # unreliable expiry-date OCR: the user supplies a reliable date, which the recommender
    # turns into "use soon" prompts and recipe timing. Independent of any scan.
    """
    CREATE TABLE IF NOT EXISTS perishables (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,               -- ingredient name the user typed
        use_by      TEXT    NOT NULL,               -- date 'YYYY-MM-DD'
        created_at  TEXT    NOT NULL                -- UTC ISO-8601 when the entry was added
    )
    """,
    # Per-ingredient nutrition cache, keyed by the canonical (lowercased) ingredient
    # token. Populated on demand from an external food-data source (e.g. Open Food Facts)
    # so the meal-plan optimizer can reason about calories/macros without re-fetching. All
    # values are optional — a row may exist with only some macros known. Values are per the
    # cited ``source``'s serving basis (typically per 100 g); callers interpret accordingly.
    """
    CREATE TABLE IF NOT EXISTS nutrition (
        item        TEXT    PRIMARY KEY,            -- canonical ingredient token (lowercased)
        kcal        REAL,
        protein_g   REAL,
        carbs_g     REAL,
        fat_g       REAL,
        source      TEXT,                           -- provenance, e.g. 'openfoodfacts'
        updated_at  TEXT    NOT NULL                -- UTC ISO-8601 when cached/refreshed
    )
    """,
    # Waste & spend log: one row each time the user resolves an item as eaten or thrown
    # away. This is the raw data behind the analytics dashboard (waste rate, money lost vs
    # rescued, most-wasted ingredients, trend over time). ``est_cost`` is optional — the
    # user may enter a rough price so spend can be summed, or leave it null (count-only).
    """
    CREATE TABLE IF NOT EXISTS waste_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        item        TEXT    NOT NULL,               -- canonical ingredient token (lowercased)
        name        TEXT    NOT NULL,               -- display name the user tracked/typed
        event       TEXT    NOT NULL,               -- 'used' | 'wasted'
        est_cost    REAL,                           -- optional user-estimated cost (money)
        logged_at   TEXT    NOT NULL                -- UTC ISO-8601 when the event was recorded
    )
    """,
    # Consumption ledger: one row each time ingredients are used up *after* the latest
    # scan — chiefly when the user marks a planned meal "cooked". This is what closes the
    # loop from plan → consumption → inventory: the current fridge view is the latest scan
    # with these deltas subtracted (see :func:`get_current_inventory`). A fresh scan is a
    # new ground-truth snapshot, so only consumption recorded *after* it still applies;
    # older rows are naturally superseded. ``item`` is the canonical ingredient token so
    # a caller's normalizer can line these up with the scan's (differently-cased) names.
    """
    CREATE TABLE IF NOT EXISTS consumption (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        item        TEXT    NOT NULL,               -- canonical ingredient token (lowercased)
        name        TEXT    NOT NULL,               -- display name consumed
        qty         INTEGER NOT NULL DEFAULT 1,     -- units consumed (>= 1)
        note        TEXT,                           -- optional context, e.g. 'Cooked: Paneer Bhurji'
        created_at  TEXT    NOT NULL                -- UTC ISO-8601 when the consumption happened
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_items_scan        ON items(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_unidentified_scan ON unidentified(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_scans_created_at  ON scans(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_perishables_use_by ON perishables(use_by)",
    "CREATE INDEX IF NOT EXISTS idx_waste_log_logged_at ON waste_log(logged_at)",
    "CREATE INDEX IF NOT EXISTS idx_consumption_created_at ON consumption(created_at)",
]


def _db_path() -> str:
    """Resolve the local SQLite file path from the environment (read at call time)."""
    return os.getenv("DATABASE_PATH", DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH


def _turso_url() -> str:
    """Return the configured Turso/libSQL URL, or ``""`` to use the local file."""
    return os.getenv("TURSO_DATABASE_URL", "").strip()


def _remote_url(url: str) -> str:
    """Normalize a Turso URL to the HTTP transport.

    Turso hands out ``libsql://<host>`` URLs. The client would treat that scheme as a
    WebSocket connection (``wss://``), but we want the **HTTP transport** instead: it is
    the right fit for serverless hosts like Vercel (no long-lived socket) and avoids the
    Hrana-over-WebSocket handshake. So ``libsql://`` is rewritten to ``https://``; any
    explicit ``http(s)://`` the user provides is left untouched.
    """
    if url.startswith("libsql://"):
        return "https://" + url[len("libsql://"):]
    return url


def _make_client() -> libsql_client.ClientSync:
    """Create a libSQL client for the configured target (remote Turso or local file)."""
    url = _turso_url()
    if url:
        token = os.getenv("TURSO_AUTH_TOKEN", "").strip() or None
        return libsql_client.create_client_sync(_remote_url(url), auth_token=token)
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


def backend() -> str:
    """The storage backend in use: ``"turso"`` (remote) or ``"local"`` (SQLite file).

    Safe to expose — it names the kind of backend without leaking the URL, path, or token.
    """
    return "turso" if _turso_url() else "local"


def ping() -> bool:
    """Return True if the database answers a trivial query, False otherwise.

    A cheap liveness probe for the ``/health`` endpoint — never raises, so a health check
    can report "db down" instead of failing.
    """
    try:
        with _connect() as client:
            client.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 — a health probe reports failure, it doesn't raise
        logger.exception("Database ping failed")
        return False


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

    Side effect:
        Clears the consumption ledger. A scan is a fresh, ground-truth snapshot of the
        fridge, so any "cooked since the last scan" deltas are now baked into it and must
        not be subtracted again (see :func:`get_current_inventory`).
    """
    items = result.get("items") or []
    unidentified = result.get("unidentified") or []
    when = (created_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Write the scan and all its rows as one atomic batch. We use batch() rather than an
    # interactive transaction because the HTTP transport used on serverless hosts (Turso)
    # doesn't support interactive transactions — but batch() is atomic on both HTTP and the
    # local file (a failing statement rolls the whole batch back).
    #
    # The child rows can't know the scan's auto-generated id ahead of time, so they resolve
    # it with ``(SELECT MAX(id) FROM scans)``. Inside the batch's single transaction libSQL
    # serializes writers, so that is exactly the scan just inserted above.
    statements: list[tuple[str, list[Any]]] = [
        (
            "INSERT INTO scans (created_at, source, item_count) VALUES (?, ?, ?)",
            [when, source, len(items)],
        )
    ]
    for item in items:
        statements.append(
            (
                "INSERT INTO items "
                "(scan_id, name, count, category, freshness, freshness_confidence, notes) "
                "VALUES ((SELECT MAX(id) FROM scans), ?, ?, ?, ?, ?, ?)",
                [
                    str(item.get("name", "")),
                    _as_int(item.get("count"), default=1),
                    _as_opt_str(item.get("category")),
                    _as_opt_str(item.get("freshness")),
                    _as_opt_str(item.get("freshness_confidence")),
                    _as_opt_str(item.get("notes")),
                ],
            )
        )
    for entry in unidentified:
        statements.append(
            (
                "INSERT INTO unidentified (scan_id, description, reason) "
                "VALUES ((SELECT MAX(id) FROM scans), ?, ?)",
                [
                    _as_opt_str(entry.get("description")),
                    _as_opt_str(entry.get("reason")),
                ],
            )
        )

    # A new snapshot supersedes any "cooked since last scan" deltas — reset the ledger in
    # the same atomic batch so the current-inventory view starts clean from this scan.
    statements.append(("DELETE FROM consumption", []))

    with _connect() as client:
        results = client.batch(statements)
        scan_id = int(results[0].last_insert_rowid)

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


def get_current_inventory(
    normalize: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    """Return the latest scan reconciled with consumption recorded since it.

    The newest scan is a full snapshot of the fridge, but the user may have *cooked* since
    then (see :func:`record_consumption`). This returns that scan with the consumed units
    subtracted: item counts drop, and items consumed entirely disappear — so the fridge
    view, recommender, optimizer, and nutrition dashboard all reflect what's *actually*
    left, closing the plan → cook → inventory loop.

    Only consumption logged *after* the scan's ``created_at`` is applied; a later scan is a
    fresh observation that already accounts for earlier consumption, so those older rows are
    ignored (the baseline resets on every scan).

    Args:
        normalize: A function mapping an ingredient display name to its canonical token, so
            a scan item named "Bell Pepper" lines up with a consumption row keyed
            ``bell_pepper``. Injected by the caller (typically
            ``recommender.normalize_ingredient``) to keep this module dependency-free;
            defaults to a plain stripped-lowercase mapping.

    Returns:
        A scan-shaped dict (same keys as :func:`get_latest_scan`) with adjusted ``items``
        and ``item_count``, or ``None`` when there is no scan yet. Unchanged when nothing
        has been consumed since the scan.
    """
    scan = get_latest_scan()
    if not scan:
        return scan

    consumed = _consumed_totals()
    if not consumed:
        return scan

    norm = normalize or (lambda s: str(s).strip().lower())
    remaining = dict(consumed)  # token -> units still to subtract
    kept: list[dict[str, Any]] = []
    for item in scan.get("items", []):
        token = norm(str(item.get("name", "")))
        have = _as_int(item.get("count"), 1)
        take = remaining.get(token, 0)
        if take > 0:
            used = min(have, take)
            have -= used
            remaining[token] = take - used
        if have > 0:
            row = dict(item)
            row["count"] = have
            kept.append(row)
        # else: fully consumed since the scan — drop it from the current view.

    adjusted = dict(scan)
    adjusted["items"] = kept
    adjusted["item_count"] = len(kept)
    adjusted["adjusted_for_consumption"] = True
    return adjusted


# --- Perishables (user-entered "use by" dates) ------------------------------


def add_perishable(name: str, use_by: str, created_at: datetime | None = None) -> int:
    """Record a perishable the user is tracking with a "use by" date; return its new id.

    Args:
        name: The ingredient name the user typed (stored as-is; the recommender
            normalizes it to a canonical token when matching).
        use_by: A ``YYYY-MM-DD`` date string (validated by the caller).
        created_at: Override timestamp (UTC). Defaults to now.

    Returns:
        The auto-generated ``perishables.id``.
    """
    when = (created_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as client:
        result = client.execute(
            "INSERT INTO perishables (name, use_by, created_at) VALUES (?, ?, ?)",
            [str(name).strip(), str(use_by).strip(), when],
        )
        perishable_id = int(result.last_insert_rowid)
    logger.info("Added perishable %d (%s, use_by=%s)", perishable_id, name, use_by)
    return perishable_id


def list_perishables() -> list[dict[str, Any]]:
    """Return all tracked perishables, soonest ``use_by`` first.

    Returns:
        A list of dicts: ``{id, name, use_by, created_at}``.
    """
    with _connect() as client:
        return _rows_to_dicts(
            client.execute(
                "SELECT id, name, use_by, created_at "
                "FROM perishables ORDER BY use_by ASC, id ASC"
            )
        )


def get_perishable(perishable_id: int) -> dict[str, Any] | None:
    """Return one tracked perishable by id, or ``None`` if it doesn't exist."""
    with _connect() as client:
        rows = _rows_to_dicts(
            client.execute(
                "SELECT id, name, use_by, created_at FROM perishables WHERE id = ?",
                [int(perishable_id)],
            )
        )
    return rows[0] if rows else None


def delete_perishable(perishable_id: int) -> bool:
    """Delete one tracked perishable; return ``True`` if a row was removed."""
    with _connect() as client:
        result = client.execute(
            "DELETE FROM perishables WHERE id = ?", [int(perishable_id)]
        )
        removed = int(result.rows_affected or 0) > 0
    if removed:
        logger.info("Deleted perishable %d", perishable_id)
    return removed


def clear_perishables() -> int:
    """Delete all tracked perishables; return how many rows were removed."""
    with _connect() as client:
        result = client.execute("DELETE FROM perishables")
        removed = int(result.rows_affected or 0)
    logger.info("Cleared %d perishable(s)", removed)
    return removed


# --- Nutrition cache --------------------------------------------------------


_NUTRITION_COLUMNS = "item, kcal, protein_g, carbs_g, fat_g, source, updated_at"


def upsert_nutrition(
    item: str,
    *,
    kcal: float | None = None,
    protein_g: float | None = None,
    carbs_g: float | None = None,
    fat_g: float | None = None,
    source: str | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Insert or refresh cached nutrition for one canonical ingredient token.

    The ``item`` key is normalized to a stripped, lowercased token so lookups are
    stable regardless of how the caller cased it. Re-caching the same item overwrites
    the previous row (SQLite/libSQL ``ON CONFLICT`` upsert), which works identically on
    the local file and on Turso.
    """
    key = str(item).strip().lower()
    if not key:
        return
    when = (updated_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as client:
        client.execute(
            f"INSERT INTO nutrition ({_NUTRITION_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(item) DO UPDATE SET "
            "kcal=excluded.kcal, protein_g=excluded.protein_g, "
            "carbs_g=excluded.carbs_g, fat_g=excluded.fat_g, "
            "source=excluded.source, updated_at=excluded.updated_at",
            [
                key,
                _as_opt_float(kcal),
                _as_opt_float(protein_g),
                _as_opt_float(carbs_g),
                _as_opt_float(fat_g),
                _as_opt_str(source),
                when,
            ],
        )
    logger.info("Cached nutrition for %s (source=%s)", key, source)


def get_nutrition(item: str) -> dict[str, Any] | None:
    """Return the cached nutrition row for one ingredient, or ``None`` if not cached."""
    key = str(item).strip().lower()
    if not key:
        return None
    with _connect() as client:
        rows = _rows_to_dicts(
            client.execute(
                f"SELECT {_NUTRITION_COLUMNS} FROM nutrition WHERE item = ?", [key]
            )
        )
    return rows[0] if rows else None


def list_nutrition() -> list[dict[str, Any]]:
    """Return every cached nutrition row, ordered by ingredient."""
    with _connect() as client:
        return _rows_to_dicts(
            client.execute(f"SELECT {_NUTRITION_COLUMNS} FROM nutrition ORDER BY item")
        )


# --- Waste & spend log ------------------------------------------------------


_WASTE_COLUMNS = "id, item, name, event, est_cost, logged_at"

# The only two outcomes we record for a tracked item.
WASTE_USED = "used"
WASTE_WASTED = "wasted"


def log_waste_event(
    name: str,
    event: str,
    *,
    token: str | None = None,
    est_cost: float | None = None,
    logged_at: datetime | None = None,
) -> int:
    """Record that an item was eaten (``used``) or thrown away (``wasted``); return its id.

    Args:
        name: The display name of the item (as the user tracked/typed it).
        event: :data:`WASTE_USED` or :data:`WASTE_WASTED`.
        token: Canonical ingredient token; derived from ``name`` (stripped/lowercased)
            when not given, so analytics can group by ingredient.
        est_cost: Optional rough price for spend analytics (non-negative), or ``None``.
        logged_at: Override timestamp (UTC). Defaults to now.

    Returns:
        The auto-generated ``waste_log.id``.

    Raises:
        ValueError: If ``event`` is not one of the two allowed outcomes.
    """
    outcome = str(event).strip().lower()
    if outcome not in (WASTE_USED, WASTE_WASTED):
        raise ValueError(f"event must be '{WASTE_USED}' or '{WASTE_WASTED}', got {event!r}")
    display = str(name).strip()
    key = (token or display).strip().lower()
    when = (logged_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cost = _as_opt_float(est_cost)
    if cost is not None and cost < 0:
        cost = None
    with _connect() as client:
        result = client.execute(
            "INSERT INTO waste_log (item, name, event, est_cost, logged_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [key, display, outcome, cost, when],
        )
        event_id = int(result.last_insert_rowid)
    logger.info("Logged %s event %d for %s (cost=%s)", outcome, event_id, display, cost)
    return event_id


def list_waste_events(limit: int = 1000) -> list[dict[str, Any]]:
    """Return recorded waste/use events, newest first.

    Returns:
        A list of dicts: ``{id, item, name, event, est_cost, logged_at}``.
    """
    with _connect() as client:
        return _rows_to_dicts(
            client.execute(
                f"SELECT {_WASTE_COLUMNS} FROM waste_log "
                "ORDER BY id DESC LIMIT ?",
                [max(1, int(limit))],
            )
        )


def clear_waste_events() -> int:
    """Delete all waste/use events; return how many rows were removed."""
    with _connect() as client:
        result = client.execute("DELETE FROM waste_log")
        removed = int(result.rows_affected or 0)
    logger.info("Cleared %d waste event(s)", removed)
    return removed


# --- Consumption ledger (closing the plan → cook → inventory loop) ----------


_CONSUMPTION_COLUMNS = "id, item, name, qty, note, created_at"


def record_consumption(
    entries: Sequence[Mapping[str, Any]],
    *,
    note: str | None = None,
    created_at: datetime | None = None,
) -> int:
    """Record ingredients consumed since the latest scan; return how many rows were written.

    Chiefly called when the user marks a planned meal "cooked" — each ingredient the dish
    used becomes a row here, and :func:`get_current_inventory` subtracts them from the
    current fridge view.

    Args:
        entries: An iterable of ``{"name": str, "token": str?, "qty": int?}`` mappings. The
            canonical ``token`` is derived from ``name`` (stripped/lowercased) when omitted;
            ``qty`` defaults to 1 and is floored at 1. Entries with a blank name are skipped.
        note: Optional shared context stored on every row (e.g. ``"Cooked: Paneer Bhurji"``).
        created_at: Override timestamp (UTC). Defaults to now.

    Returns:
        The number of ledger rows inserted (0 if ``entries`` had nothing usable).
    """
    when = (created_at or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    shared_note = _as_opt_str(note)
    statements: list[tuple[str, list[Any]]] = []
    for entry in entries or []:
        display = str(entry.get("name", "")).strip()
        if not display:
            continue
        token = str(entry.get("token") or display).strip().lower()
        qty = max(1, _as_int(entry.get("qty"), 1))
        statements.append(
            (
                "INSERT INTO consumption (item, name, qty, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [token, display, qty, shared_note, when],
            )
        )
    if not statements:
        return 0
    with _connect() as client:
        client.batch(statements)
    logger.info("Recorded %d consumption row(s) (note=%s)", len(statements), shared_note)
    return len(statements)


def list_consumption(*, since: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
    """Return consumption rows, newest first, optionally only those after an ISO timestamp.

    Args:
        since: If given, keep only rows whose ``created_at`` is strictly greater than this
            UTC ISO-8601 string (string comparison is valid for the fixed format used here).
        limit: Maximum rows to return.

    Returns:
        A list of dicts: ``{id, item, name, qty, note, created_at}``.
    """
    with _connect() as client:
        if since:
            return _rows_to_dicts(
                client.execute(
                    f"SELECT {_CONSUMPTION_COLUMNS} FROM consumption "
                    "WHERE created_at > ? ORDER BY id DESC LIMIT ?",
                    [since, max(1, int(limit))],
                )
            )
        return _rows_to_dicts(
            client.execute(
                f"SELECT {_CONSUMPTION_COLUMNS} FROM consumption "
                "ORDER BY id DESC LIMIT ?",
                [max(1, int(limit))],
            )
        )


def clear_consumption() -> int:
    """Delete all consumption rows; return how many were removed."""
    with _connect() as client:
        result = client.execute("DELETE FROM consumption")
        removed = int(result.rows_affected or 0)
    logger.info("Cleared %d consumption row(s)", removed)
    return removed


def _consumed_totals() -> dict[str, int]:
    """Sum the current consumption ledger into a ``token -> total units`` map.

    The ledger only ever holds consumption since the most recent scan — :func:`save_scan`
    clears it whenever a new snapshot arrives — so every row here still applies to the
    current fridge and no timestamp filtering is needed.
    """
    totals: dict[str, int] = {}
    for row in list_consumption():
        token = str(row.get("item") or "").strip().lower()
        if not token:
            continue
        totals[token] = totals.get(token, 0) + max(1, _as_int(row.get("qty"), 1))
    return totals


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


def _as_opt_float(value: Any) -> float | None:
    """Coerce a value to float, returning ``None`` for missing/uncoercible values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
