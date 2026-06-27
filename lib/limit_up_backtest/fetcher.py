"""
Data fetching layer for the limit-up backtest system.

Responsibilities:
- Fetch daily limit-up stock pools via akshare (ak.stock_zt_pool_em)
- Fetch K-line data via Tencent API (reusing cloud_stock_reporter.fetch_kline)
- In-memory K-line cache to avoid redundant API calls
- Rate limiting and retry logic
"""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# Reuse from the main reporter
sys.path.insert(0, "")
from cloud_stock_reporter import (
    code_prefix,
    fetch_kline,
    SESSION,
)

logger = logging.getLogger(__name__)

# ── K-line cache ───────────────────────────────────────────────────────

# Cache keyed by code (6-digit), stores a DataFrame covering a wide date range.
# Subsequent events for the same stock reuse the cached data.
_kline_cache: dict[str, pd.DataFrame] = {}


def clear_kline_cache():
    """Clear the in-memory K-line cache."""
    global _kline_cache
    _kline_cache = {}


def _get_kline_cached(
    code: str,
    lookback_days: int = 120,
    rate_delay: float = 0.5,
) -> pd.DataFrame | None:
    """Fetch K-line for a stock, using cache if available.

    Always fetches `lookback_days` bars and stores in cache.  If a cache
    entry already exists and is long enough, returns it directly.
    """
    if code in _kline_cache and len(_kline_cache[code]) >= lookback_days * 0.8:
        return _kline_cache[code]

    symbol = f"{code_prefix(code)}{code}"
    time.sleep(rate_delay)  # rate limit
    df = fetch_kline(symbol, scale="day", datalen=lookback_days)
    if df is not None:
        _kline_cache[code] = df
    return df


def _extract_window(
    df: pd.DataFrame,
    end_date: str,
    window_days: int,
) -> pd.DataFrame | None:
    """Extract a window of `window_days` bars ending on `end_date` (inclusive).

    Returns a DataFrame sorted by date ascending, or None if insufficient data.
    """
    mask = df["date"] <= end_date
    sliced = df[mask].tail(window_days)
    if len(sliced) < 26:
        return None
    return sliced.reset_index(drop=True)


# ── Limit-up pool fetching ──────────────────────────────────────────────

def _date_to_ak_format(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' to 'YYYYMMDD' for akshare."""
    return date_str.replace("-", "")


def fetch_limit_up_pool(date_str: str) -> pd.DataFrame | None:
    """Fetch the daily limit-up stock pool via akshare.

    Args:
        date_str: Trading date as 'YYYY-MM-DD'.

    Returns:
        DataFrame with akshare columns (代码, 名称, 涨跌幅, 最新价, 成交额,
        流通市值, 总市值, 换手率, 封板资金, 首次封板时间, 最后封板时间,
        炸板次数, 涨停统计, 连板数, 所属行业), or None on failure/empty.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed; cannot fetch limit-up pool")
        return None

    ak_date = _date_to_ak_format(date_str)
    try:
        df = ak.stock_zt_pool_em(date=ak_date)
    except Exception as e:
        logger.warning("Failed to fetch limit-up pool for %s: %s", date_str, e)
        return None

    if df is None or df.empty:
        return None

    return df


# ── Main fetch pipeline ─────────────────────────────────────────────────

def fetch_all_data_for_dates(
    dates: list[str],
    lookback_days: int = 90,
    max_workers: int = 4,
    rate_delay: float = 0.5,
    progress_callback=None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Fetch limit-up pools and K-line data for a list of dates.

    Returns:
        (events_by_date, errors)
        events_by_date: {date_str: [event_dict, ...]}
        errors: list of {date, code, error} dicts
    """
    events_by_date: dict[str, list[dict]] = {}
    errors: list[dict] = []

    # ── Step 1: Collect limit-up pools per date ─────────────────────
    date_pools: dict[str, pd.DataFrame] = {}
    for d in dates:
        if progress_callback:
            progress_callback(f"Fetching limit-up pool: {d}")
        pool = fetch_limit_up_pool(d)
        if pool is not None and not pool.empty:
            date_pools[d] = pool
            logger.info("  %s: %d limit-up stocks", d, len(pool))

    if not date_pools:
        logger.warning("No limit-up data found for any date in range")
        return events_by_date, errors

    # ── Step 2: Collect all unique codes and needed end dates ───────
    # Map code → latest end_date we need K-line for
    code_need_dates: dict[str, str] = {}
    for d, pool in date_pools.items():
        for _, row in pool.iterrows():
            code = str(row.get("代码", "")).strip()
            if not code or len(code) != 6:
                continue
            if code not in code_need_dates or d > code_need_dates[code]:
                code_need_dates[code] = d

    unique_codes = sorted(code_need_dates.keys())
    logger.info(
        "%d unique stocks to fetch K-line for across %d dates",
        len(unique_codes),
        len(date_pools),
    )

    # ── Step 3: Fetch K-line in parallel ────────────────────────────
    fetch_padding = lookback_days + 30  # extra padding for safety

    code_kline_map: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, str] = {}

    def _fetch_one(code: str):
        try:
            df = _get_kline_cached(code, lookback_days=fetch_padding, rate_delay=rate_delay)
            if df is not None:
                code_kline_map[code] = df
            else:
                fetch_errors[code] = "Insufficient K-line data"
        except Exception as e:
            fetch_errors[code] = str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in unique_codes}
        for i, fut in enumerate(as_completed(futures)):
            code = futures[fut]
            try:
                fut.result()
            except Exception as e:
                fetch_errors[code] = str(e)
            if progress_callback and (i + 1) % 20 == 0:
                progress_callback(
                    f"K-line fetch: {i + 1}/{len(unique_codes)}"
                )

    logger.info(
        "K-line fetch complete: %d success, %d failed",
        len(code_kline_map),
        len(fetch_errors),
    )
    for code, err in fetch_errors.items():
        errors.append({"date": code_need_dates.get(code, ""), "code": code, "error": err})

    # ── Step 4: Build event dicts per date ─────────────────────────
    col_map = {
        "代码": "code",
        "名称": "name",
        "涨跌幅": "change_pct",
        "最新价": "close_price",
        "成交额": "turnover",
        "流通市值": "float_market_cap",
        "总市值": "total_market_cap",
        "换手率": "turnover_rate",
        "封板资金": "board_lock_fund",
        "首次封板时间": "first_lock_time",
        "最后封板时间": "last_lock_time",
        "炸板次数": "blow_count",
        "涨停统计": "limit_up_count",
        "连板数": "consecutive",
        "所属行业": "industry",
    }

    for d, pool in date_pools.items():
        day_events = []
        for _, row in pool.iterrows():
            code = str(row.get("代码", "")).strip()
            if not code or len(code) != 6:
                continue

            event = {"trade_date": d}
            for src_col, dst_col in col_map.items():
                val = row.get(src_col)
                # Convert numpy/pandas types
                if pd.isna(val):
                    event[dst_col] = None
                elif isinstance(val, (float, int)):
                    event[dst_col] = val
                else:
                    event[dst_col] = str(val)

            # Convert code to string
            event["code"] = str(event.get("code", ""))

            # Parse numeric fields
            for f in [
                "change_pct", "close_price", "turnover",
                "float_market_cap", "total_market_cap", "turnover_rate",
                "board_lock_fund", "consecutive",
            ]:
                v = event.get(f)
                if v is not None and not isinstance(v, (int, float)):
                    try:
                        event[f] = float(v)
                    except (ValueError, TypeError):
                        event[f] = None

            for f in ["blow_count"]:
                v = event.get(f)
                if v is not None and not isinstance(v, int):
                    try:
                        event[f] = int(float(v))
                    except (ValueError, TypeError):
                        event[f] = None

            # Attach K-line DataFrame reference (will be processed later)
            event["_kline_df"] = None
            if code in code_kline_map:
                df = _extract_window(code_kline_map[code], d, lookback_days)
                if df is not None:
                    event["_kline_df"] = df
                else:
                    errors.append({
                        "date": d,
                        "code": code,
                        "error": "Insufficient K-line after windowing",
                    })
            else:
                errors.append({
                    "date": d,
                    "code": code,
                    "error": fetch_errors.get(code, "K-line not fetched"),
                })

            day_events.append(event)

        events_by_date[d] = day_events

    return events_by_date, errors


# ── Date utilities ──────────────────────────────────────────────────────

def generate_date_range(start: str, end: str) -> list[str]:
    """Generate list of date strings in [start, end], skipping weekends."""
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def get_incremental_dates(conn) -> list[str]:
    """Get dates from last_processed_date+1 through yesterday."""
    from .schema import get_last_processed_date

    last_date = get_last_processed_date(conn)
    if last_date:
        start = (
            datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")
    else:
        start = "2024-01-01"  # sensible default

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    if start > yesterday:
        return []
    return generate_date_range(start, yesterday)
