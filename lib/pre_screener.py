#!/usr/bin/env python3
"""Pre-screening module: filters A-share stocks before strategy screening.

Pipeline:
  1. Fetch all 主板+创业板 stocks from East Money API
  2. Exclude low-turnover stocks (configurable threshold)
  3. Technical analysis filter: exclude (score < 0 AND weekly trend == "下跌")
  4. Return candidate stock codes

Turnover source:
  - Morning (before 10:30): uses yesterday's full-day turnover snapshot
  - Afternoon (after 10:30): uses today's real-time turnover from Tencent API
  - Snapshot is auto-saved after each successful pool fetch

Caches results per day to avoid redundant heavy computation.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests

# ── Ensure cloud_stock_reporter is importable ──────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in __import__("sys").path:
    __import__("sys").path.insert(0, _PROJECT_DIR)

logger = logging.getLogger("stock_reporter.pre_screener")

# ── Config ─────────────────────────────────────────────────────────────


def _load_min_turnover():
    """Read MIN_TURNOVER from environment or .env, default 1亿 (100,000,000)."""
    val = os.environ.get("MIN_TURNOVER", "")
    if not val:
        # Try loading from .env
        env_path = os.path.join(_PROJECT_DIR, ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MIN_TURNOVER="):
                            val = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass
    if val:
        try:
            return int(val)
        except ValueError:
            try:
                return int(float(val))
            except ValueError:
                pass
    return 100_000_000  # default: 1亿


MIN_TURNOVER = _load_min_turnover()

# ── Data directory ─────────────────────────────────────────────────────
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_TURNOVER_SNAPSHOT = os.path.join(_DATA_DIR, "turnover_snapshot_{date}.json")

# ── Cutoff: before this time, use yesterday's turnover ─────────────────
_MORNING_CUTOFF_HOUR = 10
_MORNING_CUTOFF_MINUTE = 30


def _ensure_data_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _is_morning() -> bool:
    """Return True if current time is before the morning cutoff (10:30).

    Before cutoff, real-time turnover data is sparse (market just opened),
    so we should use yesterday's full-day turnover instead.
    """
    now = datetime.now()
    return (now.hour < _MORNING_CUTOFF_HOUR
            or (now.hour == _MORNING_CUTOFF_HOUR and now.minute < _MORNING_CUTOFF_MINUTE))


def _save_turnover_snapshot(pool: list[dict]) -> None:
    """Save {code: turnover} for all stocks in the pool to today's snapshot.

    This snapshot is used tomorrow morning when real-time turnover is not yet available.
    """
    _ensure_data_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    path = _TURNOVER_SNAPSHOT.format(date=today)
    snapshot = {s["code"]: s["turnover"] for s in pool if s.get("turnover", 0) > 0}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
        logger.info("Turnover snapshot saved: %d stocks → %s", len(snapshot), path)
    except Exception as e:
        logger.warning("Failed to save turnover snapshot: %s", e)


def _load_yesterday_turnover() -> dict[str, float]:
    """Load yesterday's turnover snapshot, keyed by stock code.

    Returns empty dict if snapshot doesn't exist or can't be read.
    """
    from datetime import timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = _TURNOVER_SNAPSHOT.format(date=yesterday)
    if not os.path.exists(path):
        # Try 2 days ago (for Monday → Friday)
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        path = _TURNOVER_SNAPSHOT.format(date=two_days_ago)
    if not os.path.exists(path):
        # Try 3 days ago (for post-holiday)
        three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        path = _TURNOVER_SNAPSHOT.format(date=three_days_ago)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d turnover records from %s", len(data), os.path.basename(path))
            return data
        except Exception as e:
            logger.warning("Failed to load turnover snapshot %s: %s", path, e)
    return {}


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Fetch stock pool
# ═══════════════════════════════════════════════════════════════════════


def fetch_stock_pool(turnover_map: Optional[dict[str, float]] = None) -> list[dict]:
    """Fetch stocks from akshare, prices from Tencent.

    Returns a list of dicts with keys: code, name, price, change_pct, turnover.
    Board filter and ST exclusion are driven by Config (STOCK_BOARDS / EXCLUDE_ST).

    If turnover_map is provided, real-time turnover values are replaced with
    the map's values (used in the morning to substitute yesterday's turnover).
    """
    from lib.config import get_config

    cfg = get_config()
    prefixes = cfg.allowed_prefixes
    exclude_st = cfg.exclude_st

    import akshare as ak

    logger.info("Fetching stock codes from akshare...")
    try:
        df = ak.stock_info_a_code_name()
    except Exception as e:
        logger.error("akshare stock list failed: %s", e)
        return []

    all_stocks = []
    codes_to_query = []

    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()

        if not code or len(code) != 6:
            continue
        # Board whitelist (config-driven)
        if not code.startswith(prefixes):
            continue
        # ST exclusion (config-driven)
        if exclude_st and "ST" in name.upper():
            continue

        codes_to_query.append(code)

    boards_label = cfg.stock_boards.replace(",", "+")
    logger.info("  Got %d codes (boards: %s)", len(codes_to_query), boards_label)

    # Batch-fetch real-time prices from Tencent
    logger.info("  Fetching real-time quotes from Tencent...")
    quotes = _batch_tencent_quotes(codes_to_query)

    for code in codes_to_query:
        q = quotes.get(code, {})
        realtime_turnover = q.get("turnover", 0.0)
        # In the morning, use yesterday's full-day turnover instead of sparse real-time data
        turnover = (turnover_map.get(code, realtime_turnover)
                    if turnover_map else realtime_turnover)
        all_stocks.append({
            "code": code,
            "name": q.get("name", ""),
            "price": q.get("price", 0.0),
            "change_pct": q.get("change_pct", 0.0),
            "turnover": turnover,
        })

    logger.info("  Fetched %d stocks with real-time data", len(all_stocks))
    return all_stocks


def _batch_tencent_quotes(codes: list[str]) -> dict[str, dict]:
    """Batch-fetch real-time quotes from Tencent API.

    Returns dict mapping code -> {name, price, change_pct, turnover}.
    """
    results = {}
    BATCH = 50

    for i in range(0, len(codes), BATCH):
        batch = codes[i:i + BATCH]
        symbols = []
        for code in batch:
            prefix = "sh" if code.startswith(("6",)) else "sz"
            symbols.append(f"{prefix}{code}")

        try:
            r = requests.get(
                f"https://qt.gtimg.cn/q={','.join(symbols)}",
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
            if r.status_code != 200:
                continue
            for line in r.text.split("\n"):
                if '="' not in line:
                    continue
                try:
                    raw = line.split('="', 1)[1].rstrip('";\n')
                    fields = raw.split("~")
                    if len(fields) < 4:
                        continue
                    code = fields[2]
                    name = fields[1]
                    price = float(fields[3]) if fields[3] else 0.0
                    prev_close = float(fields[4]) if fields[4] else price
                    change_pct = 0.0
                    if prev_close > 0:
                        change_pct = (price - prev_close) / prev_close * 100.0
                    # Tencent turnover is in 万元, convert to 元
                    turnover = float(fields[37]) if len(fields) > 37 and fields[37] else 0.0
                    turnover = turnover * 10000  # 万 → 元
                    results[code] = {
                        "name": name,
                        "price": price,
                        "change_pct": change_pct,
                        "turnover": turnover,
                    }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.debug("Tencent batch %d failed: %s", i // BATCH, e)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Turnover filter
# ═══════════════════════════════════════════════════════════════════════


def volume_filter(stocks: list[dict], min_turnover: Optional[int] = None) -> list[dict]:
    """Exclude stocks with 成交额 below threshold.

    Args:
        stocks: List of stock dicts from fetch_stock_pool()
        min_turnover: Minimum turnover in CNY (default: MIN_TURNOVER env/1亿)

    Returns:
        Filtered list
    """
    threshold = min_turnover if min_turnover is not None else MIN_TURNOVER
    threshold_str = _fmt_amount(threshold)

    passed = [s for s in stocks if s["turnover"] >= threshold]
    excluded = len(stocks) - len(passed)

    logger.info(
        "Turnover filter (≥ %s): %d passed, %d excluded",
        threshold_str, len(passed), excluded,
    )
    return passed


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Technical filter
# ═══════════════════════════════════════════════════════════════════════


def _analyze_one(stock: dict) -> Optional[dict]:
    """Analyze a single stock: fetch K-line, compute indicators, score.

    Returns dict with code, score, weekly_trend or None on failure.
    """
    from cloud_stock_reporter import (
        code_prefix,
        fetch_kline,
        compute_indicators,
        fetch_weekly_trend,
        score_stock,
    )

    code = stock["code"]
    prefix = code_prefix(code)
    symbol = f"{prefix}{code}"

    try:
        # Fetch K-line data
        df = fetch_kline(symbol)
        if df is None:
            return None

        # Compute indicators
        indicators = compute_indicators(df)
        if indicators is None:
            return None

        # Weekly trend
        weekly = fetch_weekly_trend(symbol)

        # Score
        score = score_stock(
            price=stock["price"],
            change_pct=stock["change_pct"],
            indicators=indicators,
            weekly=weekly,
        )

        return {
            "code": code,
            "name": stock["name"],
            "price": stock["price"],
            "change_pct": stock["change_pct"],
            "turnover": stock["turnover"],
            "score": score,
            "weekly_trend": weekly.get("weekly_trend", ""),
        }
    except Exception:
        return None


def technical_filter(
    stocks: list[dict],
    max_workers: int = 8,
) -> list[dict]:
    """Run technical analysis on each stock in parallel, exclude bad ones.

    Exclusion rule: score < 0 AND weekly_trend == "下跌"

    Returns list of dicts for stocks that passed (includes score + trend).
    """
    logger.info(
        "Technical filter: analyzing %d stocks (parallel, workers=%d)...",
        len(stocks), max_workers,
    )
    start = time.time()
    passed = []
    failed = 0
    excluded = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_analyze_one, s): s for s in stocks}
        for future in as_completed(future_map):
            completed += 1
            if completed % 200 == 0 or completed == len(stocks):
                elapsed = time.time() - start
                logger.info(
                    "  Progress: %d/%d (%.0fs, %.1fs/stock)",
                    completed, len(stocks), elapsed,
                    elapsed / completed if completed else 0,
                )

            try:
                result = future.result()
                if result is None:
                    failed += 1
                    continue

                # Exclusion rule: drop if score < 0 AND weekly trend is 下跌
                if result["score"] < 0 and result["weekly_trend"] == "下跌":
                    excluded += 1
                    continue

                passed.append(result)
            except Exception:
                failed += 1

    elapsed = time.time() - start
    logger.info(
        "Technical filter done in %.0fs: %d passed, %d excluded (score<0 + 周线下跌), %d data-failures",
        elapsed, len(passed), excluded, failed,
    )
    return passed


# ═══════════════════════════════════════════════════════════════════════
# Pipeline + Cache
# ═══════════════════════════════════════════════════════════════════════

_CACHE_PATH = os.path.join(_DATA_DIR, "pre_screened_{date}_{boards}.json")


def _cache_path(date_str: Optional[str] = None, boards_slug: str = "") -> str:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if not boards_slug:
        from lib.config import get_config
        boards_slug = get_config().boards_slug
    return _CACHE_PATH.format(date=date_str, boards=boards_slug)


def run_pre_screening(
    min_turnover: Optional[int] = None,
    max_workers: int = 8,
    force: bool = False,
    save_snapshot: bool = True,
) -> list[dict]:
    """Run full pre-screening pipeline (with daily caching).

    Args:
        min_turnover: Minimum 成交额 in CNY (default: MIN_TURNOVER env or 1亿)
        max_workers: Number of parallel workers for technical analysis
        force: If True, re-run even if today's cache exists
        save_snapshot: If True (default), save turnover snapshot for next morning

    Returns:
        List of dicts with code, name, price, change_pct, turnover, score, weekly_trend
    """
    _ensure_data_dir()
    threshold = min_turnover if min_turnover is not None else MIN_TURNOVER
    today = datetime.now().strftime("%Y-%m-%d")
    cache_path = _cache_path(today)

    # ── Check cache ──
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            logger.info(
                "Pre-screening: loaded %d candidates from cache (%s)",
                len(cached), today,
            )
            return cached
        except Exception:
            logger.warning("Failed to read cache, re-running")

    # ── Run pipeline ──
    logger.info("=" * 50)
    logger.info("Pre-screening started — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Min turnover: %s | Max workers: %d", _fmt_amount(threshold), max_workers)

    # ── Determine turnover source ──
    turnover_map = None
    if _is_morning():
        turnover_map = _load_yesterday_turnover()
        if turnover_map:
            logger.info("Morning mode: using yesterday's turnover for %d stocks", len(turnover_map))
        else:
            logger.warning("Morning mode but no yesterday snapshot found, using real-time turnover")

    # Stage 1
    pool = fetch_stock_pool(turnover_map=turnover_map)
    logger.info("Stage 1 done: %d stocks in pool", len(pool))

    # Save turnover snapshot for tomorrow morning (only in afternoon mode,
    # when we have reliable real-time turnover data).
    # Can be suppressed (e.g. PM composite run) so the closing scan owns this.
    if pool and not _is_morning() and save_snapshot:
        _save_turnover_snapshot(pool)

    # Stage 2
    liquid = volume_filter(pool, min_turnover=threshold)
    logger.info("Stage 2 done: %d stocks after turnover filter", len(liquid))

    # ── Safety: if ALL stocks failed volume filter, market likely not open ──
    if len(pool) > 0 and len(liquid) == 0:
        logger.warning(
            "⚠️  ALL %d stocks failed turnover filter (≥ %s) — "
            "market likely not open yet, skipping cache write",
            len(pool), _fmt_amount(threshold),
        )
        return []

    # Stage 3
    candidates = technical_filter(liquid, max_workers=max_workers)
    logger.info("Stage 3 done: %d candidates passed technical filter", len(candidates))

    # ── Save cache ──
    if candidates:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(candidates, f, ensure_ascii=False)
            logger.info("Cached %d candidates to %s", len(candidates), cache_path)
        except Exception as e:
            logger.warning("Failed to write cache: %s", e)
    else:
        logger.warning(
            "⚠️  0 candidates after all filters — cache NOT written, will retry next run"
        )

    logger.info("Pre-screening complete: %d candidates", len(candidates))
    return candidates


def get_pre_screened_codes(**kwargs) -> set[str]:
    """Convenience: run pre-screening and return just the set of stock codes."""
    return {s["code"] for s in run_pre_screening(**kwargs)}


def save_turnover_snapshot_for_today() -> bool:
    """Fetch full stock pool and save turnover snapshot for next morning.

    Intended to be called by the closing scan (15:00), when real-time
    turnover data is complete for the full trading day.

    Returns True if snapshot was saved successfully.
    """
    pool = fetch_stock_pool()
    if not pool:
        logger.warning("save_turnover_snapshot_for_today: empty pool, skipped")
        return False
    _save_turnover_snapshot(pool)
    return True


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _fmt_amount(amount: int) -> str:
    """Format a CNY amount for display."""
    yi = 100_000_000
    wan = 10_000
    if amount >= yi:
        return f"{amount / yi:.1f}亿"
    elif amount >= wan:
        return f"{amount / wan:.0f}万"
    return str(amount)


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    force = "--force" in sys.argv
    candidates = run_pre_screening(force=force)
    print(f"\nPre-screened candidates: {len(candidates)}")

    # Print top 20 by score
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    print(f"\nTop 20 by technical score:")
    print(f"{'Code':<10} {'Name':<10} {'Price':>8} {'Chg%':>8} {'Score':>6} {'Weekly':>8}")
    print("-" * 58)
    for s in sorted_candidates[:20]:
        print(
            f"{s['code']:<10} {s['name']:<10} {s['price']:>8.2f} "
            f"{s['change_pct']:>+7.2f}% {s['score']:>5d}  {s['weekly_trend']:>8}"
        )
