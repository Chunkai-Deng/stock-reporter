"""Enhanced stock data fetchers: quarterly reports and fund flow.

Adapted from aiagents-stock/quarterly_report_data.py and fund_flow_akshare.py.
All akshare calls are wrapped in try/except for graceful degradation.
"""

import logging
from datetime import datetime
from typing import Any

import akshare as ak
import pandas as pd

logger = logging.getLogger("stock_reporter.enhanced")


# ═══════════════════════════════════════════════════════════════════════
# Quarterly Report Fetcher
# ═══════════════════════════════════════════════════════════════════════

class QuarterlyReportFetcher:
    """Fetch A-share quarterly financial reports via akshare (Sina Finance source)."""

    def __init__(self, periods: int = 8):
        self.periods = periods

    def get_quarterly_reports(self, symbol: str) -> dict:
        """Fetch income statement, balance sheet, cash flow, and financial indicators.

        Returns a dict with data_success flag and formatted text.
        """
        data: dict[str, Any] = {
            "symbol": symbol,
            "income_statement": None,
            "balance_sheet": None,
            "cash_flow": None,
            "financial_indicators": None,
            "data_success": False,
        }

        if not (symbol.isdigit() and len(symbol) == 6):
            return data

        try:
            data["income_statement"] = self._fetch_table(symbol, "利润表")
            data["balance_sheet"] = self._fetch_table(symbol, "资产负债表")
            data["cash_flow"] = self._fetch_table(symbol, "现金流量表")
            data["financial_indicators"] = self._fetch_indicators(symbol)

            if any([data["income_statement"], data["balance_sheet"],
                    data["cash_flow"], data["financial_indicators"]]):
                data["data_success"] = True
        except Exception as e:
            logger.warning("Quarterly report fetch failed for %s: %s", symbol, e)

        return data

    def _fetch_table(self, symbol: str, table_type: str) -> dict | None:
        try:
            df = ak.stock_financial_report_sina(stock=symbol, symbol=table_type)
            if df is None or df.empty:
                return None
            df = df.head(self.periods)
            data_list = []
            for _, row in df.iterrows():
                item = {}
                for col in df.columns:
                    v = row.get(col)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        continue
                    try:
                        item[col] = str(v)
                    except Exception:
                        item[col] = "N/A"
                if item:
                    data_list.append(item)
            return {"data": data_list, "periods": len(data_list)}
        except Exception as e:
            logger.debug("  %s fetch failed: %s", table_type, e)
            return None

    def _fetch_indicators(self, symbol: str) -> dict | None:
        try:
            df = ak.stock_financial_abstract(symbol=symbol)
            if df is None or df.empty:
                return None
            df = df.head(self.periods * 2)
            key_indicators = [
                '净资产收益率(ROE)', '总资产报酬率(ROA)', '销售净利率', '销售毛利率',
                '资产负债率', '流动比率', '速动比率',
                '基本每股收益', '每股净资产', '每股现金流',
            ]
            indicator_rows = df[df['指标'].isin(key_indicators)]
            if indicator_rows.empty:
                return None
            date_columns = [c for c in df.columns if c not in ('选项', '指标')]
            data_list = []
            for dc in date_columns[:self.periods]:
                item = {'报告期': dc}
                for _, row in indicator_rows.iterrows():
                    name = row['指标']
                    v = row.get(dc)
                    if v is not None and not (isinstance(v, float) and pd.isna(v)):
                        try:
                            item[name] = str(v)
                        except Exception:
                            item[name] = "N/A"
                    else:
                        item[name] = "N/A"
                data_list.append(item)
            return {"data": data_list, "periods": len(data_list)}
        except Exception as e:
            logger.debug("  Financial indicators fetch failed: %s", e)
            return None

    def format_for_ai(self, data: dict) -> str:
        """Format quarterly report data into AI-readable text."""
        if not data or not data.get("data_success"):
            return ""

        parts = [f"股票代码：{data.get('symbol', 'N/A')}  |  最近{self.periods}期季报\n"]

        for section, title, fields in [
            ("income_statement", "利润表", [
                '报告期', '营业总收入', '营业收入', '营业利润', '利润总额', '净利润',
                '归属于母公司所有者的净利润', '基本每股收益',
            ]),
            ("balance_sheet", "资产负债表", [
                '报告期', '资产总计', '负债合计', '所有者权益合计',
            ]),
            ("cash_flow", "现金流量表", [
                '报告期', '经营活动产生的现金流量净额',
                '投资活动产生的现金流量净额', '筹资活动产生的现金流量净额',
            ]),
        ]:
            sec = data.get(section)
            if sec:
                parts.append(f"\n── {title}（{sec.get('periods', 0)}期）──")
                for idx, item in enumerate(sec.get('data', []), 1):
                    vals = "  |  ".join(f"{f}: {item[f]}" for f in fields if f in item)
                    parts.append(f"  [{idx}] {vals}")

        indicators = data.get("financial_indicators")
        if indicators:
            key_fields = [
                '报告期', '净资产收益率(ROE)', '总资产报酬率(ROA)',
                '销售净利率', '销售毛利率', '资产负债率',
                '流动比率', '速动比率', '基本每股收益', '每股净资产',
            ]
            parts.append(f"\n── 关键财务指标（{indicators.get('periods', 0)}期）──")
            for idx, item in enumerate(indicators.get('data', []), 1):
                vals = "  |  ".join(f"{f}: {item.get(f, 'N/A')}" for f in key_fields if f in item)
                parts.append(f"  [{idx}] {vals}")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Fund Flow Fetcher
# ═══════════════════════════════════════════════════════════════════════

class FundFlowFetcher:
    """Fetch A-share individual stock fund flow data via akshare (East Money source)."""

    def __init__(self, days: int = 30):
        self.days = days

    def _get_market(self, symbol: str) -> str:
        if symbol.startswith(('60', '688')):
            return 'sh'
        elif symbol.startswith(('00', '30')):
            return 'sz'
        elif symbol.startswith(('8', '4')):
            return 'bj'
        return 'sz'

    def get_fund_flow_data(self, symbol: str) -> dict:
        """Fetch individual stock fund flow data.

        Returns a dict with data_success flag and raw records.
        """
        data: dict[str, Any] = {
            "symbol": symbol,
            "fund_flow_data": None,
            "data_success": False,
        }

        if not (symbol.isdigit() and len(symbol) == 6):
            return data

        try:
            market = self._get_market(symbol)
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if df is None or df.empty:
                return data

            df = df.head(self.days)
            records = []
            for _, row in df.iterrows():
                records.append({
                    '日期': str(row.get('日期', '')),
                    '收盘价': row.get('收盘价', 0),
                    '涨跌幅': row.get('涨跌幅', 0),
                    '主力净流入-净额': row.get('主力净流入-净额', 0),
                    '主力净流入-净占比': row.get('主力净流入-净占比', 0),
                    '超大单净流入-净额': row.get('超大单净流入-净额', 0),
                    '超大单净流入-净占比': row.get('超大单净流入-净占比', 0),
                    '大单净流入-净额': row.get('大单净流入-净额', 0),
                    '大单净流入-净占比': row.get('大单净流入-净占比', 0),
                    '中单净流入-净额': row.get('中单净流入-净额', 0),
                    '中单净流入-净占比': row.get('中单净流入-净占比', 0),
                    '小单净流入-净额': row.get('小单净流入-净额', 0),
                    '小单净流入-净占比': row.get('小单净流入-净占比', 0),
                })

            data["fund_flow_data"] = {
                "data": records,
                "days": len(records),
                "market": market,
                "query_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            data["data_success"] = True
        except Exception as e:
            logger.warning("Fund flow fetch failed for %s: %s", symbol, e)

        return data

    def format_for_ai(self, data: dict) -> str:
        """Format fund flow data into AI-readable text with statistical summary."""
        if not data or not data.get("data_success"):
            return ""

        fd = data.get("fund_flow_data")
        if not fd:
            return ""

        records = fd.get("data", [])
        if not records:
            return ""

        parts = [
            f"股票代码：{data.get('symbol', 'N/A')}  |  最近{fd.get('days', 0)}个交易日\n"
        ]

        # Show last 10 days in detail
        for idx, r in enumerate(records[-10:], 1):
            parts.append(
                f"[{idx}] {r['日期']}  收盘:{r['收盘价']}  涨跌:{r['涨跌幅']}%  "
                f"主力净额:{r['主力净流入-净额']}({r['主力净流入-净占比']}%)  "
                f"超大单:{r['超大单净流入-净额']}({r['超大单净流入-净占比']}%)  "
                f"大单:{r['大单净流入-净额']}({r['大单净流入-净占比']}%)"
            )

        # Statistical summary
        main_flows = [r['主力净流入-净额'] for r in records
                      if isinstance(r.get('主力净流入-净额'), (int, float))]
        if main_flows:
            total = sum(main_flows)
            avg = total / len(main_flows)
            pos_days = sum(1 for x in main_flows if x > 0)
            neg_days = sum(1 for x in main_flows if x < 0)
            parts.append(
                f"\n── 统计汇总（{len(main_flows)}日）──\n"
                f"累计净流入: {total:.0f}万  日均: {avg:.0f}万  "
                f"流入天数: {pos_days}  流出天数: {neg_days}  "
                f"流入占比: {pos_days/len(main_flows)*100:.1f}%"
            )

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Enhanced Data Manager (caching wrapper)
# ═══════════════════════════════════════════════════════════════════════

class EnhancedDataManager:
    """Manages quarterly report and fund flow data with daily in-memory cache."""

    def __init__(self):
        self._quarterly = QuarterlyReportFetcher()
        self._fund_flow = FundFlowFetcher()
        self._cache: dict[str, dict] = {}
        self._cache_date: str = ""

    def _check_cache(self):
        """Clear cache if the date has changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._cache_date:
            self._cache.clear()
            self._cache_date = today

    def fetch_for_stock(self, code: str) -> dict:
        """Fetch and cache enhanced data for a stock.

        Returns dict with keys: 'quarterly_text', 'fund_flow_text'
        """
        self._check_cache()

        if code in self._cache:
            return self._cache[code]

        result: dict[str, str] = {"quarterly_text": "", "fund_flow_text": ""}

        # Quarterly reports
        logger.info("  Fetching quarterly reports for %s...", code)
        q_data = self._quarterly.get_quarterly_reports(code)
        if q_data.get("data_success"):
            result["quarterly_text"] = self._quarterly.format_for_ai(q_data)

        # Fund flow
        logger.info("  Fetching fund flow for %s...", code)
        f_data = self._fund_flow.get_fund_flow_data(code)
        if f_data.get("data_success"):
            result["fund_flow_text"] = self._fund_flow.format_for_ai(f_data)

        self._cache[code] = result
        return result

    def get_quarterly_text(self, code: str) -> str:
        """Get pre-formatted quarterly report text for a stock."""
        return self.fetch_for_stock(code).get("quarterly_text", "")

    def get_fund_flow_text(self, code: str) -> str:
        """Get pre-formatted fund flow text for a stock."""
        return self.fetch_for_stock(code).get("fund_flow_text", "")
