"""
Downstream query API for the limit-up backtest system.

All functions use read-only connections and are safe for concurrent access.
Other modules (AI agents, screening strategies, reports) import from here.
"""

from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from .schema import get_connection


# ── Event queries ──────────────────────────────────────────────────────

def query_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    codes: Optional[list[str]] = None,
    industry: Optional[str] = None,
    min_consecutive: Optional[int] = None,
    pattern_type: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """Query limit-up events with flexible filters.

    Args:
        start_date: Inclusive start date 'YYYY-MM-DD'.
        end_date: Inclusive end date 'YYYY-MM-DD'.
        codes: Filter by stock codes.
        industry: Filter by industry (substring match).
        min_consecutive: Minimum consecutive limit-up count.
        pattern_type: Filter by pattern type (requires JOIN with event_patterns).
        limit: Max results to return.

    Returns:
        List of event dicts with keys matching limit_up_events columns.
    """
    conn = get_connection(readonly=True)
    try:
        base = """
            SELECT DISTINCT e.* FROM limit_up_events e
        """
        joins = []
        wheres = ["1=1"]
        params: list = []

        if pattern_type:
            joins.append(
                "JOIN event_patterns ep ON e.event_id = ep.event_id"
            )
            wheres.append("ep.pattern_type = ?")
            params.append(pattern_type)

        if start_date:
            wheres.append("e.trade_date >= ?")
            params.append(start_date)
        if end_date:
            wheres.append("e.trade_date <= ?")
            params.append(end_date)
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            wheres.append(f"e.code IN ({placeholders})")
            params.extend(codes)
        if industry:
            wheres.append("e.industry LIKE ?")
            params.append(f"%{industry}%")
        if min_consecutive is not None:
            wheres.append("e.consecutive >= ?")
            params.append(min_consecutive)

        sql = f"{base} {' '.join(joins)} WHERE {' AND '.join(wheres)} ORDER BY e.trade_date DESC, e.consecutive DESC LIMIT ?"
        params.append(limit)

        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_event_detail(event_id: int) -> dict | None:
    """Get full event info including its pattern classifications."""
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            "SELECT * FROM limit_up_events WHERE event_id = ?", (event_id,)
        )
        event_row = cur.fetchone()
        if not event_row:
            return None
        event = dict(event_row)

        # Attach patterns
        cur2 = conn.execute(
            "SELECT pattern_type, confidence, detail FROM event_patterns WHERE event_id = ?",
            (event_id,),
        )
        event["patterns"] = [dict(r) for r in cur2.fetchall()]

        # Attach count of daily indicators
        cur3 = conn.execute(
            "SELECT COUNT(*) as cnt FROM daily_indicators WHERE event_id = ?",
            (event_id,),
        )
        row3 = cur3.fetchone()
        event["daily_bars"] = row3["cnt"] if row3 else 0

        return event
    finally:
        conn.close()


def get_events_for_stock(code: str, limit: int = 50) -> list[dict]:
    """Get all limit-up events for a given stock, most recent first."""
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            """
            SELECT * FROM limit_up_events
            WHERE code = ?
            ORDER BY trade_date DESC
            LIMIT ?
            """,
            (code, limit),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ── Indicator queries ──────────────────────────────────────────────────

def get_pre_limit_up_series(event_id: int) -> pd.DataFrame | None:
    """Return 90-day indicator time series as DataFrame, index=days_before.

    Drops metadata columns (id, event_id) — only indicator values remain.
    """
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            """
            SELECT * FROM daily_indicators
            WHERE event_id = ?
            ORDER BY days_before ASC
            """,
            (event_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        df = df.set_index("days_before")
        # Drop metadata columns for a clean indicator view
        df = df.drop(columns=["id", "event_id"], errors="ignore")
        return df
    finally:
        conn.close()


def get_indicator_snapshot(event_id: int, days_before: int) -> dict | None:
    """Get a single day's indicator snapshot."""
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            """
            SELECT * FROM daily_indicators
            WHERE event_id = ? AND days_before = ?
            """,
            (event_id, days_before),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Aggregate statistics ───────────────────────────────────────────────

def get_pattern_distribution(
    start_date: str, end_date: str
) -> dict[str, int]:
    """Count of events by pattern_type in date range."""
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            """
            SELECT ep.pattern_type, COUNT(DISTINCT ep.event_id) as cnt
            FROM event_patterns ep
            JOIN limit_up_events e ON ep.event_id = e.event_id
            WHERE e.trade_date >= ? AND e.trade_date <= ?
            GROUP BY ep.pattern_type
            ORDER BY cnt DESC
            """,
            (start_date, end_date),
        )
        return {r["pattern_type"]: r["cnt"] for r in cur.fetchall()}
    finally:
        conn.close()


def get_industry_stats(
    start_date: str, end_date: str
) -> pd.DataFrame:
    """Industry-level stats: event count, avg consecutive, top patterns."""
    conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            """
            SELECT
                e.industry,
                COUNT(*) as event_count,
                AVG(e.consecutive) as avg_consecutive,
                AVG(e.change_pct) as avg_change_pct,
                AVG(e.turnover_rate) as avg_turnover_rate
            FROM limit_up_events e
            WHERE e.trade_date >= ? AND e.trade_date <= ?
              AND e.industry IS NOT NULL AND e.industry != ''
            GROUP BY e.industry
            ORDER BY event_count DESC
            """,
            (start_date, end_date),
        )
        rows = [dict(r) for r in cur.fetchall()]
        return pd.DataFrame(rows)
    finally:
        conn.close()


def get_indicator_averages(
    start_date: str, end_date: str, days_before: int = -1
) -> dict:
    """Average indicator values across all events at a given day offset.

    Args:
        days_before: Day offset relative to limit-up (default -1 = day before).
    """
    conn = get_connection(readonly=True)
    try:
        indicator_cols = [
            "ma5", "ma10", "ma20",
            "macd", "macd_signal", "macd_hist",
            "rsi", "bb_width_pct", "k", "d", "j",
            "vol_ratio", "adx", "plus_di", "minus_di",
        ]
        agg_sql = ", ".join(f"AVG(di.{c}) as avg_{c}" for c in indicator_cols)
        sql = f"""
            SELECT {agg_sql}
            FROM daily_indicators di
            JOIN limit_up_events e ON di.event_id = e.event_id
            WHERE e.trade_date >= ? AND e.trade_date <= ?
              AND di.days_before = ?
        """
        cur = conn.execute(sql, (start_date, end_date, days_before))
        row = cur.fetchone()
        if row:
            return {
                k.replace("avg_", ""): round(v, 4) if v is not None else None
                for k, v in dict(row).items()
            }
        return {}
    finally:
        conn.close()


# ── Similarity ─────────────────────────────────────────────────────────

def find_similar_pre_limit_up(
    event_id: int,
    top_n: int = 10,
    metric: str = "correlation",
) -> list[dict]:
    """Find events with most similar pre-limit-up indicator trajectory.

    Uses a lightweight approach: extract a fingerprint from the query event's
    last 5 pre-limit-up days and compare against other events.
    """
    conn = get_connection(readonly=True)
    try:
        # Query event
        q_cur = conn.execute(
            """
            SELECT e.code, e.trade_date FROM limit_up_events e
            WHERE e.event_id = ?
            """,
            (event_id,),
        )
        q_row = q_cur.fetchone()
        if not q_row:
            return []

        # Get query fingerprint (last 5 pre-limit-up days)
        fingerprint_cols = [
            "ma5", "ma10", "ma20", "macd", "rsi", "k", "d", "j",
            "vol_ratio", "adx",
        ]
        q_fp = _get_fingerprint(conn, event_id, fingerprint_cols, days=[-5, -3, -1])
        if not q_fp:
            return []

        # Get all other events in reasonable range (same quarter as query)
        q_date = q_row["trade_date"]
        candidates = conn.execute(
            """
            SELECT e.event_id, e.code, e.trade_date, e.name, e.consecutive
            FROM limit_up_events e
            WHERE e.event_id != ?
              AND e.trade_date >= date(?, '-90 days')
              AND e.trade_date <= date(?, '+90 days')
            LIMIT 500
            """,
            (event_id, q_date, q_date),
        ).fetchall()

        # Score each candidate
        scored = []
        for c in candidates:
            c_fp = _get_fingerprint(conn, c["event_id"], fingerprint_cols, days=[-5, -3, -1])
            if not c_fp:
                continue
            similarity = _cosine_similarity(q_fp, c_fp)
            if similarity is not None:
                scored.append((similarity, dict(c)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"similarity": round(s, 4), **event}
            for s, event in scored[:top_n]
        ]
    finally:
        conn.close()


def _get_fingerprint(
    conn, event_id: int, cols: list[str], days: list[int]
) -> list[float] | None:
    """Extract a fingerprint vector from daily indicators at specific day offsets."""
    placeholders = ",".join(["?"] * len(days))
    cur = conn.execute(
        f"""
        SELECT {', '.join(cols)} FROM daily_indicators
        WHERE event_id = ? AND days_before IN ({placeholders})
        ORDER BY days_before ASC
        """,
        (event_id, *days),
    )
    rows = cur.fetchall()
    if len(rows) < len(days):
        return None
    vec = []
    for row in rows:
        for c in cols:
            v = row[c]
            vec.append(float(v) if v is not None else 0.0)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return None
    import numpy as np
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return None
    return float(dot / (norm_a * norm_b))
