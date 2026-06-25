#!/usr/bin/env python3
"""Recommendation tracker — records picks and tracks performance over time.

Stores daily picks as JSON files under data/recommendations/.
Provides functions to save, load, and evaluate recommendation performance.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger("stock_reporter.tracker")

_PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_REC_DIR = os.path.join(_PROJECT_DIR, "data", "recommendations")

# HTTP session for price lookups
_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
})


def _ensure_dir():
    os.makedirs(_REC_DIR, exist_ok=True)


def _rec_path(date_str: str, suffix: str = "") -> str:
    base = f"{date_str}{suffix}"
    return os.path.join(_REC_DIR, f"{base}.json")


# ── Save ───────────────────────────────────────────────────────────────


def save_recommendation(
    date_str: str,
    picks: list[dict],
    pre_screened_count: int = 0,
    suffix: str = "",
) -> bool:
    """Save a day's recommendation picks to JSON.

    Args:
        date_str: Date string YYYY-MM-DD
        picks: List of dicts with code, name, price, score, strategies, industry
        pre_screened_count: Number of stocks that passed pre-screening
        suffix: Optional suffix for filename, e.g. '_pm' for afternoon run

    Returns:
        True on success
    """
    _ensure_dir()
    record = {
        "date": date_str,
        "pre_screened_count": pre_screened_count,
        "picks": picks,
    }
    try:
        with open(_rec_path(date_str, suffix), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        label = f"{date_str}{suffix}"
        logger.info("Recommendation saved: %d picks to %s", len(picks), label)
        return True
    except Exception as e:
        logger.warning("Failed to save recommendation: %s", e)
        return False


# ── Load ───────────────────────────────────────────────────────────────


def load_history(days: int = 30) -> list[dict]:
    """Load recommendation records from the last N days.

    Returns list of records (most recent first), each with date, picks, etc.
    """
    _ensure_dir()
    records = []
    for i in range(days):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        path = _rec_path(date_str)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    records.append(json.load(f))
            except Exception:
                logger.debug("Failed to read %s", path)
    return records


# ── Performance ────────────────────────────────────────────────────────


def _fetch_current_price(code: str) -> Optional[float]:
    """Fetch current price for a single stock via Tencent API."""
    prefix = "sh" if code.startswith(("6",)) else "sz"
    symbol = f"{prefix}{code}"
    try:
        url = f"https://qt.gtimg.cn/q={symbol}"
        r = _SESSION.get(url, timeout=5)
        if r.status_code == 200 and r.text and '="' in r.text:
            raw = r.text.split('="', 1)[1].rstrip('";\n')
            fields = raw.split("~")
            if len(fields) > 3 and fields[3]:
                return float(fields[3])
    except Exception:
        pass
    return None


def calc_performance(suffix: str = "") -> Optional[str]:
    """Calculate performance of yesterday's picks against current prices.

    Args:
        suffix: Same suffix used when saving (e.g. '_am', '_pm')

    Returns a WeCom-formatted string or None if no yesterday record.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    path = _rec_path(yesterday, suffix)
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
    except Exception:
        return None

    picks = record.get("picks", [])
    if not picks:
        return None

    lines = [
        f"📊 昨日推荐回顾 ({record['date']})",
        "━" * 24,
    ]

    up_count = 0
    for pick in picks[:5]:
        code = pick.get("code", "")
        name = pick.get("name", "")
        old_price = pick.get("price", 0)
        current_price = _fetch_current_price(code)

        if current_price and old_price > 0:
            change = (current_price - old_price) / old_price * 100
            emoji = "✅" if change >= 0 else "❌"
            lines.append(f"  {code} {name}: {change:+.1f}% {emoji}")
            if change >= 0:
                up_count += 1
        else:
            lines.append(f"  {code} {name}: 数据获取失败")

    total = len(picks)
    win_rate = up_count / total * 100 if total > 0 else 0
    lines.append(f"\n  胜率: {up_count}/{total} ({win_rate:.0f}%)")

    return "\n".join(lines)


def strategy_win_rate(days: int = 30) -> dict:
    """Calculate win rate per strategy over the last N days.

    Returns dict mapping strategy name to (wins, total, rate).
    """
    records = load_history(days=days)
    from collections import defaultdict

    strategy_stats: dict = defaultdict(lambda: {"wins": 0, "total": 0})

    for record in records:
        for pick in record.get("picks", []):
            price_at_rec = pick.get("price", 0)
            code = pick.get("code", "")
            current = _fetch_current_price(code)
            is_win = current is not None and price_at_rec > 0 and current >= price_at_rec

            strategies = pick.get("strategies", [])
            for s in strategies:
                # Extract strategy name (e.g., "💪主力资金#1" → "💪主力资金")
                sname = s.rsplit("#", 1)[0] if "#" in s else s
                strategy_stats[sname]["total"] += 1
                if is_win:
                    strategy_stats[sname]["wins"] += 1

    result = {}
    for sname, stats in strategy_stats.items():
        total = stats["total"]
        wins = stats["wins"]
        rate = wins / total * 100 if total > 0 else 0
        # Adjust weight up for higher win rates (3% floor)
        weight = max(3, round(rate / 10))
        result[sname] = {"wins": wins, "total": total, "rate": rate, "weight": weight}

    return result
