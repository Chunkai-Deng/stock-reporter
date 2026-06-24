#!/usr/bin/env python3
"""Stock screening CLI tool.

Run pywencai-based screening strategies to discover new A-share stocks.

Usage:
    python stock_screener.py --strategy low_price_bull --top 5
    python stock_screener.py --strategy value --top 10 --send
    python stock_screener.py --strategy value --codes-only
    python stock_screener.py --all
    python stock_screener.py --composite --send   # 综合选股
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.selectors import (
    screen_low_price_bull,
    screen_value_stocks,
    screen_profit_growth,
    screen_small_cap,
    screen_main_force,
    format_screening_results,
    run_all_screenings,
    _clean_codes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stock_screener")

# ── WeCom push (reuse from main script if available, else stand-alone) ─
try:
    from lib.config import get_config
    import requests

    _cfg = get_config()
    WEBHOOK_URL = _cfg.webhook_url
except ImportError:
    WEBHOOK_URL = os.environ.get("WECOM_WEBHOOK_URL", "")


def send_to_wecom(msg: str) -> bool:
    """Send a message to WeCom webhook."""
    if not WEBHOOK_URL:
        logger.error("WECOM_WEBHOOK_URL not configured!")
        return False
    payload = {"msgtype": "text", "text": {"content": msg}}
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        result = r.json()
        if result.get("errcode") == 0:
            logger.info("Sent to WeCom successfully.")
            return True
        else:
            logger.error("WeCom error: %s", r.text)
            return False
    except Exception as e:
        logger.error("WeCom request failed: %s", e)
        return False


# ── Strategy registry ────────────────────────────────────────────────

STRATEGIES = {
    "low_price_bull": ("低价擒牛", screen_low_price_bull),
    "value": ("价值投资", screen_value_stocks),
    "profit_growth": ("净利增长", screen_profit_growth),
    "small_cap": ("小市值", screen_small_cap),
    "main_force": ("主力资金", lambda: screen_main_force(days_ago=20)),
}


def main():
    parser = argparse.ArgumentParser(
        description="A-share stock screening tool (pywencai/问财)"
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        help="Screening strategy to run",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top stocks to show (default: 10)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all 5 strategies independently"
    )
    parser.add_argument(
        "--composite", action="store_true", help="综合选股: pre-screening + 5 strategies + cross-scoring + AI portfolio"
    )
    parser.add_argument(
        "--pm", action="store_true", help="午后模式: 单独保存数据到 _pm 文件（等同于 --suffix _pm）"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="推荐文件后缀，如 _am / _pm，不传则不额外保存"
    )
    parser.add_argument(
        "--send", action="store_true", help="Send results to WeCom webhook"
    )
    parser.add_argument(
        "--codes-only",
        action="store_true",
        help="Output stock codes only (one per line, for .env)",
    )
    args = parser.parse_args()

    if not args.strategy and not args.all and not args.composite:
        parser.print_help()
        sys.exit(1)

    # ── 综合选股: composite mode ─────────────────────────────────────
    if args.composite:
        suffix = args.suffix if args.suffix else ("_pm" if args.pm else "")
        msg = run_all_screenings(top_n=args.top, suffix=suffix)
        print(msg)
        if args.send:
            send_to_wecom(msg[:4000])
        sys.exit(0)

    # Collect results
    outputs: list[str] = []

    if args.all:
        strategies_to_run = list(STRATEGIES.items())
    else:
        strategies_to_run = [
            (args.strategy, STRATEGIES[args.strategy])
        ]

    for key, (name, func) in strategies_to_run:
        try:
            df, msg = func()
            print(f"\n{'='*50}")
            print(f"  {name}")
            print(f"{'='*50}")
            print(f"  {msg}")

            if df is not None and not df.empty:
                if args.codes_only:
                    codes = _clean_codes(df.head(args.top))
                    for c in codes:
                        print(c)
                else:
                    formatted = format_screening_results(df, name, args.top)
                    print(formatted)
                    outputs.append(formatted)
            else:
                print(f"  (无结果)")
                outputs.append(f"📋 {name}: 无符合条件的股票")
        except Exception as e:
            logger.error("%s failed: %s", name, e)
            outputs.append(f"📋 {name}: 失败 — {e}")

    # Send to WeCom if requested
    if args.send and outputs:
        combined = "\n\n".join(outputs)
        send_to_wecom(combined)


if __name__ == "__main__":
    main()
