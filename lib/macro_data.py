"""Macro economic data fetcher.

Fetches key Chinese macro indicators via akshare and public APIs.
Zero API key required — all data from free public sources.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

import akshare as ak
import pandas as pd
import requests

logger = logging.getLogger("stock_reporter.macro")

# Suppress SSL warnings for NBS API
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class MacroDataFetcher:
    """Fetch Chinese macro economic indicators and market indices."""

    NBS_URL = "https://data.stats.gov.cn/easyquery.htm"

    # Key NBS indicator codes
    NBS_INDICATORS = {
        "gdp_yoy": {
            "dbcode": "hgjd", "group_code": "A0103",
            "series_code": "A010301", "label": "GDP当季同比", "unit": "%",
            "period": "LAST8",
        },
        "cpi_yoy": {
            "dbcode": "hgyd", "group_code": "A01010J",
            "series_code": "A01010J01", "label": "CPI同比", "unit": "%",
            "period": "LAST8",
        },
        "ppi_yoy": {
            "dbcode": "hgyd", "group_code": "A010801",
            "series_code": "A01080101", "label": "PPI同比", "unit": "%",
            "period": "LAST8",
        },
        "manufacturing_pmi": {
            "dbcode": "hgyd", "group_code": "A0B01",
            "series_code": "A0B0101", "label": "制造业PMI", "unit": "",
            "period": "LAST8",
        },
        "m2_yoy": {
            "dbcode": "hgyd", "group_code": "A0D01",
            "series_code": "A0D0103", "label": "M2同比", "unit": "%",
            "period": "LAST8",
        },
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        })

    # ── NBS API ──────────────────────────────────────────────────────

    def _fetch_nbs_indicator(self, config: dict) -> dict | None:
        """Fetch a single NBS indicator series."""
        params = {
            "m": "QueryData",
            "dbcode": config["dbcode"],
            "rowcode": "zb",
            "colcode": "sj",
            "wds": "[]",
            "dfwds": json.dumps([
                {
                    "wdcode": "zb",
                    "valuecode": config["series_code"],
                }
            ]),
        }
        try:
            r = self.session.get(self.NBS_URL, params=params, timeout=15, verify=False)
            r.raise_for_status()
            data = r.json()
            return self._parse_nbs_response(data, config)
        except Exception as e:
            logger.debug("NBS fetch failed for %s: %s", config["label"], e)
            return None

    @staticmethod
    def _parse_nbs_response(data: dict, config: dict) -> dict | None:
        """Parse NBS JSON response into {label, values, latest, previous}."""
        try:
            returndata = data.get("returndata", {})
            datanodes = returndata.get("datanodes", [])
            if not datanodes:
                return None

            values = []
            for node in datanodes:
                code = node.get("code", "")
                # Extract date from code (format: zb.A010301_sj.2025Q1)
                parts = code.split(".")
                period = parts[-1] if len(parts) >= 3 else ""
                val = node.get("data", {}).get("data")
                if val is not None:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                values.append({"period": period, "value": val})

            if not values:
                return None

            latest = values[-1]["value"] if values else None
            previous = values[-2]["value"] if len(values) >= 2 else None

            return {
                "label": config["label"],
                "unit": config.get("unit", ""),
                "latest": latest,
                "previous": previous,
                "values": values,
            }
        except Exception as e:
            logger.debug("NBS parse failed: %s", e)
            return None

    # ── Market Indices ────────────────────────────────────────────────

    def _fetch_market_indices(self) -> dict:
        """Fetch recent A-share market index performance via akshare."""
        indices = {}
        index_map = {
            "上证指数": "sh000001",
            "深证成指": "sz399001",
            "创业板指": "sz399006",
            "沪深300": "sh000300",
        }
        try:
            df = ak.stock_zh_index_spot_em()
            for name, code in index_map.items():
                row = df[df["代码"] == code]
                if not row.empty:
                    r = row.iloc[0]
                    indices[name] = {
                        "latest": float(r["最新价"]) if r["最新价"] else None,
                        "change_pct": float(r["涨跌幅"]) if r["涨跌幅"] else None,
                        "volume": float(r["成交额"]) if r["成交额"] else None,
                    }
        except Exception as e:
            logger.debug("Market index fetch failed: %s", e)
        return indices

    # ── Main Fetch ────────────────────────────────────────────────────

    def fetch_all_data(self) -> dict:
        """Fetch all macro data and return structured results."""
        logger.info("Fetching macro economic data...")

        # NBS indicators
        nbs_results = {}
        for key, config in self.NBS_INDICATORS.items():
            result = self._fetch_nbs_indicator(config)
            if result:
                nbs_results[key] = result
            else:
                nbs_results[key] = {
                    "label": config["label"], "latest": None,
                    "previous": None, "error": "获取失败",
                }

        # Market indices
        market_indices = self._fetch_market_indices()

        # Macro news from akshare
        news = self._fetch_macro_news()

        return {
            "indicators": nbs_results,
            "market_indices": market_indices,
            "news": news,
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _fetch_macro_news(self, limit: int = 10) -> list:
        """Fetch macro-related financial news."""
        try:
            df = ak.stock_info_global_em()
            if df is None or df.empty:
                return []
            keywords = ["PMI", "CPI", "PPI", "GDP", "央行", "降准", "降息", "M2", "宏观"]
            filtered = df[df["标题"].str.contains("|".join(keywords), na=False)]
            return filtered.head(limit)["标题"].tolist()
        except Exception:
            return []

    # ── Prompt Building ───────────────────────────────────────────────

    def build_prompt_context(self, data: dict) -> str:
        """Format macro data into AI-readable text."""
        parts = [f"【中国宏观数据概览】  更新时间: {data.get('fetch_time', 'N/A')}\n"]

        # Indicators table
        indicators = data.get("indicators", {})
        if indicators:
            parts.append("── 核心指标 ──")
            for key, ind in indicators.items():
                latest = ind.get("latest")
                previous = ind.get("previous")
                latest_str = f"{latest}{ind.get('unit', '')}" if latest is not None else "N/A"
                prev_str = f"{previous}{ind.get('unit', '')}" if previous is not None else "N/A"

                # Compute direction
                direction = ""
                if latest is not None and previous is not None:
                    if latest > previous:
                        direction = "↑"
                    elif latest < previous:
                        direction = "↓"
                    else:
                        direction = "→"

                parts.append(
                    f"  {ind.get('label', key)}: {latest_str}  "
                    f"(前值: {prev_str}) {direction}"
                )

        # Market indices
        market = data.get("market_indices", {})
        if market:
            parts.append("\n── 大盘指数 ──")
            for name, info in market.items():
                chg = info.get("change_pct")
                sign = "+" if (chg and chg >= 0) else ""
                parts.append(
                    f"  {name}: {info.get('latest', 'N/A')}  "
                    f"({sign}{chg:.2f}%)" if chg is not None else f"  {name}: N/A"
                )

        # News
        news = data.get("news", [])
        if news:
            parts.append("\n── 宏观新闻 ──")
            for i, title in enumerate(news[:8], 1):
                parts.append(f"  {i}. {title}")

        return "\n".join(parts)
