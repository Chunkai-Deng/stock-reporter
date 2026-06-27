"""
Database schema, connection management, and DDL for the limit-up backtest system.

Uses SQLite with WAL mode for concurrent read safety.  All write operations
happen single-threaded through the pipeline; WAL enables safe reads from
downstream consumers while a backfill is in progress.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# ── Path resolution ────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_db_path() -> str:
    return str(_PROJECT_ROOT / "data" / "limit_up_backtest" / "limit_up_backtest.db")


def _get_db_path() -> str:
    """Resolve DB path from env or default."""
    return os.environ.get("LIMIT_UP_BACKTEST_DB") or _default_db_path()


# ── Connection management ──────────────────────────────────────────────

_conn: sqlite3.Connection | None = None


def get_connection(readonly: bool = True) -> sqlite3.Connection:
    """Return a connection to the backtest database.

    With readonly=True (default): returns a new read-only connection each call.
        These are safe to use concurrently from multiple threads.
    With readonly=False: returns the singleton write connection.
        Only one writer should exist at a time.
    """
    db_path = _get_db_path()

    if readonly:
        # Read-only connections are created per-call and never cached.
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    # Write connection — singleton, cached.
    global _conn
    if _conn is not None:
        try:
            # Verify connection is still usable
            _conn.execute("SELECT 1")
            return _conn
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            _conn = None  # reconnect below

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _conn = sqlite3.connect(db_path)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode = WAL")
    _conn.execute("PRAGMA foreign_keys = ON")
    _conn.execute("PRAGMA busy_timeout = 5000")
    return _conn


def close_write_connection():
    """Close the singleton write connection if open."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


# ── DDL ────────────────────────────────────────────────────────────────

DDL_STATEMENTS = [
    # -- Meta table -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # -- Limit-up events --------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS limit_up_events (
        event_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date         TEXT    NOT NULL,
        code               TEXT    NOT NULL,
        name               TEXT,
        change_pct         REAL,
        close_price        REAL,
        turnover           REAL,
        float_market_cap   REAL,
        total_market_cap   REAL,
        turnover_rate      REAL,
        board_lock_fund    REAL,
        first_lock_time    TEXT,
        last_lock_time     TEXT,
        blow_count         INTEGER,
        limit_up_count     TEXT,
        consecutive        INTEGER,
        industry           TEXT,
        pre_ma_alignment   TEXT,
        pre_score          INTEGER,
        created_at         TEXT DEFAULT (datetime('now','localtime'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_date
        ON limit_up_events(trade_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_code
        ON limit_up_events(code)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_industry
        ON limit_up_events(industry)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_events_unique
        ON limit_up_events(trade_date, code)
    """,
    # -- Daily indicators -------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS daily_indicators (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id         INTEGER NOT NULL,
        days_before      INTEGER NOT NULL,
        actual_date      TEXT    NOT NULL,
        open             REAL,
        high             REAL,
        low              REAL,
        close            REAL,
        volume           REAL,
        ma5              REAL,
        ma10             REAL,
        ma20             REAL,
        macd             REAL,
        macd_signal      REAL,
        macd_hist        REAL,
        rsi              REAL,
        bb_upper         REAL,
        bb_middle        REAL,
        bb_lower         REAL,
        bb_width_pct     REAL,
        k                REAL,
        d                REAL,
        j                REAL,
        vol_ratio        REAL,
        vol_trend        TEXT,
        adx              REAL,
        plus_di          REAL,
        minus_di         REAL,
        divergence       TEXT,
        macd_cross       TEXT,
        FOREIGN KEY (event_id) REFERENCES limit_up_events(event_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_daily_event
        ON daily_indicators(event_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_daily_days_before
        ON daily_indicators(event_id, days_before)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_daily_actual_date
        ON daily_indicators(actual_date)
    """,
    # -- Event patterns ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS event_patterns (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id         INTEGER NOT NULL,
        pattern_type     TEXT    NOT NULL,
        confidence       REAL    NOT NULL,
        detail           TEXT,
        FOREIGN KEY (event_id) REFERENCES limit_up_events(event_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_patterns_type
        ON event_patterns(pattern_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_patterns_event
        ON event_patterns(event_id)
    """,
    # -- Weight configs ---------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS weight_configs (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        version            INTEGER NOT NULL,
        generated_at       TEXT NOT NULL,
        config_json        TEXT NOT NULL,
        positive_samples   INTEGER,
        control_samples    INTEGER,
        date_range_start   TEXT,
        date_range_end     TEXT,
        is_active          INTEGER DEFAULT 0
    )
    """,
]


def ensure_schema(conn: sqlite3.Connection | None = None):
    """Create all tables and indexes if they don't exist.

    Idempotent — safe to call every time the DB is opened.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection(readonly=False)

    # First run all DDL (idempotent — uses IF NOT EXISTS)
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)

    # Check / set schema version
    cur = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    current_version = int(row["value"]) if row else 0

    if current_version < SCHEMA_VERSION:
        logger.info(
            "Migrating schema from v%d to v%d", current_version, SCHEMA_VERSION
        )
        for stmt in DDL_STATEMENTS:
            conn.execute(stmt)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        logger.info("Schema migration complete")

    if own_conn:
        conn.close()


# ── Simple helpers for the pipeline ────────────────────────────────────

def get_last_processed_date(conn: sqlite3.Connection) -> str | None:
    """Return the last date processed in incremental mode, or None."""
    cur = conn.execute(
        "SELECT value FROM meta WHERE key = 'last_processed_date'"
    )
    row = cur.fetchone()
    return row["value"] if row else None


def set_last_processed_date(conn: sqlite3.Connection, date_str: str):
    """Update the last processed date tracker."""
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_processed_date', ?)",
        (date_str,),
    )


def save_weight_config(
    conn: sqlite3.Connection,
    config_json: str,
    positive_samples: int = 0,
    control_samples: int = 0,
    date_range_start: str = "",
    date_range_end: str = "",
):
    """Save a new weight_config record and mark it active (deactivate others)."""
    import json
    cfg = json.loads(config_json)
    version = cfg.get("version", 1)
    generated_at = cfg.get("generated_at", "")

    # Deactivate existing active configs
    conn.execute("UPDATE weight_configs SET is_active = 0")

    conn.execute(
        """
        INSERT INTO weight_configs
            (version, generated_at, config_json,
             positive_samples, control_samples,
             date_range_start, date_range_end, is_active)
        VALUES (?,?,?,?,?,?,?,1)
        """,
        (
            version, generated_at, config_json,
            positive_samples, control_samples,
            date_range_start, date_range_end,
        ),
    )


def get_active_weight_config(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent active weight config as a Python dict, or None."""
    import json
    cur = conn.execute(
        "SELECT config_json FROM weight_configs WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    return json.loads(row["config_json"]) if row else None
