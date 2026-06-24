#!/usr/bin/env python3
"""Macro economic analysis reporter.

Fetches China macro data (GDP/CPI/PMI/M2/etc.) and uses DeepSeek AI
to assess economic conditions and policy direction.

Usage:
    python macro_reporter.py           # print to stdout
    python macro_reporter.py --send    # push to WeCom
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.macro_data import MacroDataFetcher
from lib.macro_agents import MacroAgents
from lib.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("macro_reporter")

# ── WeCom ────────────────────────────────────────────────────────────

try:
    import requests
    _cfg = get_config()
    WEBHOOK_URL = _cfg.webhook_url
except ImportError:
    WEBHOOK_URL = os.environ.get("WECOM_WEBHOOK_URL", "")


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


# ── Build Message ────────────────────────────────────────────────────

def build_message(data: dict, analysis: dict) -> str:
    """Build a WeCom-formatted macro report message."""
    ctx = data.get("prompt_context", "")
    macro = analysis.get("macro_analysis", "")
    policy = analysis.get("policy_analysis", "")

    lines = [
        "\U0001f30f 宏观分析报告",
        "━" * 26,
        f"更新时间: {data.get('fetch_time', 'N/A')}",
        "",
    ]

    # Key indicators snapshot
    indicators = data.get("indicators", {})
    if indicators:
        lines.append("【核心指标速览】")
        for key, ind in indicators.items():
            latest = ind.get("latest")
            unit = ind.get("unit", "")
            val = f"{latest}{unit}" if latest is not None else "N/A"
            lines.append(f"  {ind.get('label', key)}: {val}")
        lines.append("")

    # AI analysis
    if macro:
        lines.append(f"【宏观判断】{macro}")
        lines.append("")
    if policy:
        lines.append(f"【政策分析】{policy}")
        lines.append("")

    lines.append("⚠️ 以上为AI分析，仅供参考，不构成投资建议。")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="China macro economic analysis reporter"
    )
    parser.add_argument("--send", action="store_true", help="Push report to WeCom")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    cfg = get_config()
    if not cfg.deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY is required for macro analysis")
        sys.exit(1)

    # Fetch data
    logger.info("Fetching macro data...")
    fetcher = MacroDataFetcher()
    data = fetcher.fetch_all_data()
    context = fetcher.build_prompt_context(data)
    data["prompt_context"] = context

    # AI analysis
    logger.info("Running AI analysis...")
    agents = MacroAgents()
    analysis = agents.run_macro_analysis(context)
    data["fetch_time"] = data.get("fetch_time", "")

    # Output
    msg = build_message(data, analysis)

    if not args.quiet:
        print(msg)

    if args.send:
        send_to_wecom(msg)


if __name__ == "__main__":
    main()
