"""
Aggregate statistics and reporting for the limit-up backtest system.

Provides summary views useful after a backfill run or for periodic reports.
"""

from __future__ import annotations

import logging
from typing import Optional

from .schema import get_connection

logger = logging.getLogger(__name__)


def get_summary_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Return a high-level summary of the database contents.

    Returns dict with:
        total_events, date_range_start, date_range_end,
        unique_stocks, unique_industries, top_industries,
        pattern_counts, events_with_indicators
    """
    conn = get_connection(readonly=True)
    try:
        wheres = []
        params = []
        if start_date:
            wheres.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            wheres.append("trade_date <= ?")
            params.append(end_date)
        where_clause = f"WHERE {' AND '.join(wheres)}" if wheres else ""

        result: dict = {}

        # Total events
        cur = conn.execute(
            f"SELECT COUNT(*) as cnt FROM limit_up_events {where_clause}", params
        )
        result["total_events"] = cur.fetchone()["cnt"]

        # Date range
        cur = conn.execute("SELECT MIN(trade_date) as mn, MAX(trade_date) as mx FROM limit_up_events")
        row = cur.fetchone()
        result["date_range_start"] = row["mn"]
        result["date_range_end"] = row["mx"]

        # Unique stocks
        cur = conn.execute(
            f"SELECT COUNT(DISTINCT code) as cnt FROM limit_up_events {where_clause}", params
        )
        result["unique_stocks"] = cur.fetchone()["cnt"]

        # Unique industries
        ind_where = "industry IS NOT NULL AND industry != ''"
        if where_clause:
            ind_where = where_clause + " AND " + ind_where
        cur = conn.execute(
            f"SELECT COUNT(DISTINCT industry) as cnt FROM limit_up_events WHERE {ind_where}",
            params,
        )
        result["unique_industries"] = cur.fetchone()["cnt"]

        # Top industries
        cur = conn.execute(
            f"""
            SELECT industry, COUNT(*) as cnt
            FROM limit_up_events
            WHERE {ind_where}
            GROUP BY industry
            ORDER BY cnt DESC
            LIMIT 10
            """,
            params,
        )
        result["top_industries"] = [dict(r) for r in cur.fetchall()]

        # Pattern counts
        cur = conn.execute(
            f"""
            SELECT ep.pattern_type, COUNT(DISTINCT ep.event_id) as cnt
            FROM event_patterns ep
            JOIN limit_up_events e ON ep.event_id = e.event_id
            {where_clause.replace('trade_date', 'e.trade_date') if where_clause else ''}
            GROUP BY ep.pattern_type
            ORDER BY cnt DESC
            """,
            params if where_clause else [],
        )
        result["pattern_counts"] = [dict(r) for r in cur.fetchall()]

        # Events with indicators coverage
        cur = conn.execute(
            f"""
            SELECT
                COUNT(DISTINCT e.event_id) as total,
                COUNT(DISTINCT di.event_id) as with_indicators
            FROM limit_up_events e
            LEFT JOIN daily_indicators di ON e.event_id = di.event_id
            {where_clause.replace('trade_date', 'e.trade_date') if where_clause else ''}
            """,
            params if where_clause else [],
        )
        row = cur.fetchone()
        result["events_with_indicators"] = row["with_indicators"]

        # Consecutive breakdown
        cur = conn.execute(
            f"""
            SELECT
                CASE
                    WHEN consecutive = 1 THEN '首板'
                    WHEN consecutive = 2 THEN '2连板'
                    WHEN consecutive = 3 THEN '3连板'
                    WHEN consecutive >= 4 THEN '4+连板'
                    ELSE '未知'
                END as board_type,
                COUNT(*) as cnt
            FROM limit_up_events
            {where_clause}
            GROUP BY board_type
            ORDER BY MIN(consecutive)
            """,
            params,
        )
        result["consecutive_breakdown"] = [dict(r) for r in cur.fetchall()]

        return result
    finally:
        conn.close()


def print_summary(stats: dict):
    """Pretty-print a summary dict to the logger."""
    logger.info("=" * 60)
    logger.info("  Limit-Up Backtest Summary")
    logger.info("=" * 60)
    logger.info(
        "  Date range: %s – %s",
        stats.get("date_range_start", "N/A"),
        stats.get("date_range_end", "N/A"),
    )
    logger.info(
        "  Events: %d | Unique stocks: %d | Industries: %d",
        stats.get("total_events", 0),
        stats.get("unique_stocks", 0),
        stats.get("unique_industries", 0),
    )
    logger.info(
        "  Events with indicators: %d / %d",
        stats.get("events_with_indicators", 0),
        stats.get("total_events", 0),
    )

    logger.info("  --- Consecutive Breakdown ---")
    for row in stats.get("consecutive_breakdown", []):
        logger.info("    %s: %d", row["board_type"], row["cnt"])

    logger.info("  --- Pattern Distribution ---")
    for row in stats.get("pattern_counts", []):
        logger.info("    %s: %d", row["pattern_type"], row["cnt"])

    logger.info("  --- Top Industries ---")
    for row in stats.get("top_industries", []):
        logger.info("    %s: %d", row["industry"], row["cnt"])
    logger.info("=" * 60)
