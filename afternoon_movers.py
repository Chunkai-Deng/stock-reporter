#!/usr/bin/env python3
"""Afternoon closing scan: top gainers by board + morning pick performance.

Fetches top 20 gainers for 主板 and 创业板 from East Money API,
computes technical indicators, cross-references morning recommendations,
and pushes a consolidated report to WeCom.

Usage:
    python afternoon_movers.py           # run once, print + push to WeCom
    python afternoon_movers.py --print   # print only, don't push
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import requests

# Ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("afternoon_movers")

# ── Config ───────────────────────────────────────────────────────────

TOP_N = 20
MAX_WORKERS = 8

# ── WeCom webhook ────────────────────────────────────────────────────


def _load_webhook_url() -> str:
    """Read webhook URL from env or .env file."""
    val = os.environ.get("WECOM_WEBHOOK_URL", "")
    if val:
        return val
    env_path = os.path.join(_SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("WECOM_WEBHOOK_URL="):
                        return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return ""


WEBHOOK_URL = _load_webhook_url()


def send_to_wecom(msg: str) -> bool:
    if not WEBHOOK_URL:
        logger.error("WECOM_WEBHOOK_URL not configured!")
        return False
    payload = {"msgtype": "text", "text": {"content": msg}}
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if r.json().get("errcode") == 0:
            logger.info("Sent to WeCom successfully.")
            return True
        logger.error("WeCom error: %s", r.text)
        return False
    except Exception as e:
        logger.error("WeCom request failed: %s", e)
        return False


# ── Data fetching ────────────────────────────────────────────────────


def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def fetch_all_real_time_quotes(codes: list[str]) -> dict[str, dict]:
    """Batch-fetch real-time quotes from Tencent API.

    Returns dict mapping code -> {name, price, change_pct, turnover, industry}.
    Tencent provides true real-time data (3-second refresh).
    """
    results = {}
    BATCH = 50

    for i in range(0, len(codes), BATCH):
        batch = codes[i:i + BATCH]
        symbols = []
        for code in batch:
            prefix = "sh" if code.startswith(("6",)) else "sz"
            symbols.append(f"{prefix}{code}")

        url = f"https://qt.gtimg.cn/q={','.join(symbols)}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            for line in r.text.split("\n"):
                line = line.strip()
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
                    turnover = float(fields[37]) if len(fields) > 37 and fields[37] else 0.0
                    turnover = turnover * 10000
                    results[code] = {
                        "code": code,
                        "name": name,
                        "price": price,
                        "change_pct": change_pct,
                        "turnover": turnover,
                        "industry": "",
                    }
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            logger.debug("Tencent batch failed: %s", e)

    return results


def fetch_top_gainers_realtime(
    stock_pool_codes: list[str],
    top_n: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Fetch all real-time quotes, then sort and split into main/chiNext top N.

    Uses Tencent real-time API (3s refresh), no stale data.

    Returns (main_board_top, chinext_top).
    """
    logger.info("Fetching real-time quotes for %d stocks via Tencent...", len(stock_pool_codes))
    start = time.time()

    all_quotes = fetch_all_real_time_quotes(stock_pool_codes)

    elapsed = time.time() - start
    logger.info("  Got %d quotes in %.1fs", len(all_quotes), elapsed)

    # Exclude ST and sort by change_pct descending
    valid = [
        q for q in all_quotes.values()
        if q["code"] and "ST" not in q["name"].upper()
    ]
    valid.sort(key=lambda x: x["change_pct"], reverse=True)

    # Split by board
    main_board = [s for s in valid if s["code"].startswith(("60", "00"))][:top_n]
    chinext = [s for s in valid if s["code"].startswith("30")][:top_n]

    return main_board, chinext


# ── Technical analysis (reuses cloud_stock_reporter) ─────────────────


def _analyze_gainer(stock: dict) -> Optional[dict]:
    """Compute technical indicators and score for a single stock."""
    from cloud_stock_reporter import (
        code_prefix, fetch_kline, compute_indicators,
        fetch_weekly_trend, score_stock,
    )

    code = stock["code"]
    prefix = code_prefix(code)

    try:
        df = fetch_kline(f"{prefix}{code}")
        if df is None:
            return None

        indicators = compute_indicators(df)
        if indicators is None:
            return None

        weekly = fetch_weekly_trend(f"{prefix}{code}")
        score = score_stock(stock["price"], stock["change_pct"], indicators, weekly)

        # Extract a few key signals for summary display
        divergence = indicators.get("divergence", "")
        macd_cross = indicators.get("macd_cross", "")
        vol_trend = indicators.get("vol_trend", "")
        rsi = indicators.get("rsi")
        j_val = indicators.get("j")

        signals = []
        if macd_cross:
            signals.append(f"MACD:{macd_cross}")
        if divergence:
            signals.append(divergence)
        if vol_trend and vol_trend != "正常":
            signals.append(vol_trend)
        if rsi is not None and (rsi > 70 or rsi < 30):
            signals.append(f"RSI:{rsi:.0f}")
        if j_val is not None and (j_val > 100 or j_val < 0):
            signals.append("KDJ极端")

        return {
            **stock,
            "score": score,
            "weekly_trend": weekly.get("weekly_trend", ""),
            "signals": signals[:3],  # top 3 signals
            "vol_ratio": indicators.get("vol_ratio"),
        }
    except Exception:
        return None


def enrich_with_indicators(stocks: list[dict]) -> list[dict]:
    """Run technical analysis on a list of stocks in parallel.

    Returns list enriched with score, signals, vol_ratio, weekly_trend.
    """
    logger.info("  Computing technical indicators for %d stocks...", len(stocks))
    enriched = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(_analyze_gainer, s): s for s in stocks}
        for future in as_completed(future_map):
            try:
                result = future.result()
                if result:
                    enriched.append(result)
                else:
                    # Keep the stock without indicators
                    stock = future_map[future]
                    stock["score"] = None
                    stock["signals"] = []
                    stock["vol_ratio"] = None
                    stock["weekly_trend"] = ""
                    enriched.append(stock)
            except Exception:
                stock = future_map[future]
                stock["score"] = None
                stock["signals"] = []
                stock["vol_ratio"] = None
                stock["weekly_trend"] = ""
                enriched.append(stock)

    # Sort back by change_pct descending
    enriched.sort(key=lambda x: x.get("change_pct", 0), reverse=True)

    elapsed = time.time() - start
    logger.info("  Technical analysis done in %.1fs", elapsed)
    return enriched


# ── Morning linkage ──────────────────────────────────────────────────


def _load_picks(date_str: str, suffix: str = "") -> Optional[list[dict]]:
    """Load recommendation picks from tracker by date and optional suffix."""
    rec_path = os.path.join(_SCRIPT_DIR, "data", "recommendations", f"{date_str}{suffix}.json")
    if not os.path.exists(rec_path):
        return None
    try:
        with open(rec_path, encoding="utf-8") as f:
            record = json.load(f)
        return record.get("picks", [])
    except Exception:
        return None


def _fetch_current_prices(codes: list[str]) -> dict[str, float]:
    """Batch-fetch current prices via Tencent API."""
    if not codes:
        return {}

    symbols = []
    for code in codes:
        prefix = "sh" if code.startswith(("6",)) else "sz"
        symbols.append(f"{prefix}{code}")

    prices = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i + 50]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for line in r.text.split("\n"):
                    if '="' in line:
                        raw = line.split('="', 1)[1].rstrip('";\n')
                        fields = raw.split("~")
                        if len(fields) > 3 and fields[3]:
                            prices[fields[2]] = float(fields[3])
        except Exception:
            pass

    return prices


def build_morning_linkage(today: str, top_gainer_codes: set[str]) -> str:
    """Build the morning linkage section of the report.

    Returns empty string if no morning picks found.
    """
    picks = _load_picks(today)
    if not picks:
        return ""

    pick_codes = [p.get("code", "") for p in picks if p.get("code")]
    current_prices = _fetch_current_prices(pick_codes)

    lines = [
        "\n🔗 晨间联动",
        "━" * 20,
        "  今日早间推荐表现:",
        "",
    ]

    up_count = 0
    highlighted = set()

    for i, pick in enumerate(picks, 1):
        code = pick.get("code", "")
        name = pick.get("name", "")
        morning_price = pick.get("price", 0)
        current_price = current_prices.get(code)

        # Check if this stock made today's top gainers (regardless of price)
        in_top = code in top_gainer_codes
        if in_top:
            highlighted.add(code)
        highlight = " 🔥 同时上榜!" if in_top else ""

        if current_price and morning_price > 0:
            change_pct = (current_price - morning_price) / morning_price * 100
            emoji = "✅" if change_pct >= 0 else "❌"
            lines.append(
                f"  {i}. {code} {name}: "
                f"早¥{morning_price:.2f}→现¥{current_price:.2f} ({change_pct:+.1f}%) {emoji}{highlight}"
            )
            if change_pct >= 0:
                up_count += 1
        else:
            lines.append(f"  {i}. {code} {name}: 数据获取失败")

    total = len(picks)
    win_rate = up_count / total * 100 if total > 0 else 0
    summary = f"\n  胜率: {up_count}/{total} ({win_rate:.0f}%)"

    if highlighted:
        summary += f" | 上榜: {len(highlighted)}只"

    lines.append(summary)

    return "\n".join(lines)


def build_yesterday_linkage(today: str, top_gainer_codes: set[str]) -> str:
    """Build the yesterday-PM linkage section.

    Loads previous trading day's 14:00 PM screening data and compares
    against today's top gainers. Returns empty string if no PM data found.
    """
    from datetime import datetime as dt, timedelta

    yesterday = (dt.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    picks = _load_picks(yesterday, suffix="_pm")
    if not picks:
        return ""

    pick_codes = [p.get("code", "") for p in picks if p.get("code")]
    current_prices = _fetch_current_prices(pick_codes)

    lines = [
        "\n🔗 昨日联动 (14:00选股)",
        "━" * 20,
        f"  昨日下午推荐表现:",
        "",
    ]

    up_count = 0
    highlighted = set()

    for i, pick in enumerate(picks, 1):
        code = pick.get("code", "")
        name = pick.get("name", "")
        pm_price = pick.get("price", 0)
        current_price = current_prices.get(code)

        in_top = code in top_gainer_codes
        if in_top:
            highlighted.add(code)
        highlight = " 🔥 同时上榜!" if in_top else ""

        if current_price and pm_price > 0:
            change_pct = (current_price - pm_price) / pm_price * 100
            emoji = "✅" if change_pct >= 0 else "❌"
            lines.append(
                f"  {i}. {code} {name}: "
                f"昨¥{pm_price:.2f}→现¥{current_price:.2f} ({change_pct:+.1f}%) {emoji}{highlight}"
            )
            if change_pct >= 0:
                up_count += 1
        else:
            lines.append(f"  {i}. {code} {name}: 数据获取失败")

    total = len(picks)
    win_rate = up_count / total * 100 if total > 0 else 0
    summary = f"\n  胜率: {up_count}/{total} ({win_rate:.0f}%)"

    if highlighted:
        summary += f" | 上榜: {len(highlighted)}只"

    lines.append(summary)

    return "\n".join(lines)


# ── Message builder ──────────────────────────────────────────────────


def _fmt_stock_line(idx: int, s: dict) -> str:
    """Format one stock as a single line."""
    sign = "+" if s.get("change_pct", 0) >= 0 else ""
    score_str = f"评分:{s['score']}" if s.get("score") is not None else "评分:--"
    industry = f" [{s.get('industry','')}]" if s.get("industry") else ""

    line = (
        f"  {idx:>2}. {s['code']} {s['name']}{industry} "
        f"¥{s['price']:.2f} ({sign}{s['change_pct']:.2f}%)  {score_str}"
    )

    signals = s.get("signals", [])
    if signals:
        line += f"  {'|'.join(signals)}"

    return line


def build_board_message(
    board_label: str,
    gainers: list[dict],
    part: int,
    total_parts: int,
) -> str:
    """Build message for a single board, kept under WeChat 2048 char limit."""
    t = datetime.now().strftime("%H:%M")
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"📈 收盘涨幅扫描 ({part}/{total_parts})  {today} {t}",
        "━" * 26,
        f"{board_label} Top {len(gainers)} 涨幅",
        "─" * 26,
    ]

    if not gainers:
        lines.append("  (暂无数据)")
    else:
        for idx, s in enumerate(gainers, 1):
            lines.append(_fmt_stock_line(idx, s))

    lines.append("")
    lines.append("⚠️ 仅供参考，不构成投资建议。")
    return "\n".join(lines)


def build_linkage_message(morning_section: str) -> str:
    """Build a standalone morning linkage message."""
    t = datetime.now().strftime("%H:%M")
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        f"📈 晨间联动  {today} {t}",
        "━" * 20,
        morning_section,
        "",
        "⚠️ 仅供参考，不构成投资建议。",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    push = "--print" not in sys.argv

    if push and not WEBHOOK_URL:
        logger.error("WECOM_WEBHOOK_URL not configured — cannot push to WeCom")
        sys.exit(1)

    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Afternoon scan — %s", today)

    # Step 1: Load stock pool from pre-screened cache
    stock_pool_codes = []
    try:
        from lib.pre_screener import _cache_path
        cache_path = _cache_path(today)
    except ImportError:
        cache_path = os.path.join(_SCRIPT_DIR, "data", f"pre_screened_{today}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            stock_pool_codes = [c["code"] for c in cached]
            logger.info("Loaded %d codes from pre-screen cache (%s)", len(stock_pool_codes), os.path.basename(cache_path))
        except Exception:
            pass

    if not stock_pool_codes:
        # Fallback: try to fetch stock pool from East Money
        try:
            from lib.pre_screener import fetch_stock_pool
            pool = fetch_stock_pool()
            stock_pool_codes = [s["code"] for s in pool]
        except Exception:
            pass

    if not stock_pool_codes:
        logger.error("No stock pool available, aborting")
        sys.exit(1)

    logger.info("Stock pool: %d codes", len(stock_pool_codes))

    # Step 2: Fetch real-time quotes + split into top gainers by board
    main_gainers, chinext_gainers = fetch_top_gainers_realtime(stock_pool_codes, top_n=TOP_N)
    logger.info("Main board top %d: fetched", len(main_gainers))
    logger.info("ChiNext top %d: fetched", len(chinext_gainers))

    # Step 3: Enrich with technical indicators
    all_gainer_codes: set[str] = set()
    board_results: dict[str, list[dict]] = {}

    for display_label, gainers in [("🔵 主板", main_gainers), ("🟢 创业板", chinext_gainers)]:
        gainers = enrich_with_indicators(gainers)
        board_results[display_label] = gainers
        for s in gainers:
            all_gainer_codes.add(s["code"])

    # Step 3: Linkage sections
    morning_section = build_morning_linkage(today, all_gainer_codes)
    yesterday_section = build_yesterday_linkage(today, all_gainer_codes)

    # Step 4: Build and send (split into multiple messages for WeChat limit)
    main_gainers = board_results.get("🔵 主板", [])
    chinext_gainers = board_results.get("🟢 创业板", [])

    total = 2
    if morning_section:
        total += 1
    if yesterday_section:
        total += 1

    # Message 1: 主板
    msg1 = build_board_message("🔵 主板", main_gainers, 1, total)
    print(msg1)
    # Message 2: 创业板
    msg2 = build_board_message("🟢 创业板", chinext_gainers, 2, total)
    print(msg2)
    # Message 3: 晨间联动
    msg3 = ""
    if morning_section:
        msg3 = build_linkage_message(morning_section)
        print(msg3)
    # Message 4: 昨日联动
    msg4 = ""
    if yesterday_section:
        msg4 = build_linkage_message(yesterday_section)
        print(msg4)

    if push:
        ok = True
        if not send_to_wecom(msg1):
            ok = False
        time.sleep(0.5)
        if not send_to_wecom(msg2):
            ok = False
        if msg3:
            time.sleep(0.5)
            if not send_to_wecom(msg3):
                ok = False
        if msg4:
            time.sleep(0.5)
            if not send_to_wecom(msg4):
                ok = False

        total_msgs = 2 + (1 if msg3 else 0) + (1 if msg4 else 0)
        if ok:
            logger.info(f"Report pushed to WeCom ({total_msgs} messages).")
        else:
            logger.error("Failed to push some messages to WeCom.")

    # ── Save turnover snapshot for next morning ──────────────────────
    try:
        from lib.pre_screener import save_turnover_snapshot_for_today
        logger.info("Saving turnover snapshot for next morning...")
        save_turnover_snapshot_for_today()
    except Exception as e:
        logger.warning("Failed to save turnover snapshot: %s", e)


if __name__ == "__main__":
    main()
