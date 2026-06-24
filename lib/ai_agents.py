"""Multi-agent stock analysis engine.

Adapted from aiagents-stock/ai_agents.py.
Orchestrates 3 AI agents (technical, fundamental, fund flow) + team discussion + final decision.
"""

import logging
from typing import Any

from lib.deepseek_client import DeepSeekClient
from lib.config import get_config

logger = logging.getLogger("stock_reporter.agents")


class StockAnalysisAgents:
    """Multi-agent stock analysis orchestrator.

    Three core agents:
      - Technical analyst: price trends, indicators, chart patterns
      - Fundamental analyst: financials, quarterly reports, valuation
      - Fund flow analyst: capital flows, institutional behavior

    Then: team discussion -> final structured decision.
    """

    def __init__(self, model: str | None = None):
        cfg = get_config()
        self.model = model or cfg.deepseek_model
        self.client = DeepSeekClient(model=self.model)

    # ── Agent runners ────────────────────────────────────────────────

    def _run_technical(self, stock_info: dict, indicators: dict) -> dict:
        logger.info("  [技术面分析] running...")
        analysis = self.client.technical_analysis(stock_info, indicators)
        return {
            "agent_name": "技术分析师",
            "agent_role": "负责技术指标分析、图表形态识别、趋势判断",
            "analysis": analysis,
            "focus_areas": ["技术指标", "趋势分析", "支撑阻力", "交易信号"],
        }

    def _run_fundamental(
        self, stock_info: dict, quarterly_text: str = ""
    ) -> dict:
        logger.info("  [基本面分析] running...")
        analysis = self.client.fundamental_analysis(stock_info, quarterly_text)
        return {
            "agent_name": "基本面分析师",
            "agent_role": "负责公司财务分析、行业研究、估值分析",
            "analysis": analysis,
            "focus_areas": ["财务指标", "行业分析", "公司价值", "成长性", "季报趋势"],
        }

    def _run_fund_flow(
        self, stock_info: dict, indicators: dict, fund_flow_text: str = ""
    ) -> dict:
        logger.info("  [资金面分析] running...")
        analysis = self.client.fund_flow_analysis(
            stock_info, indicators, fund_flow_text
        )
        return {
            "agent_name": "资金面分析师",
            "agent_role": "负责资金流向分析和主力行为研判",
            "analysis": analysis,
            "focus_areas": ["主力资金", "机构动向", "资金博弈", "量价配合"],
        }

    # ── Orchestration ─────────────────────────────────────────────────

    def run_multi_agent_analysis(
        self,
        stock_info: dict,
        indicators: dict,
        quarterly_text: str = "",
        fund_flow_text: str = "",
    ) -> dict:
        """Run the three core agents and return their results.

        Returns dict keyed by agent key with analysis reports.
        """
        results: dict[str, Any] = {}

        # Technical (always runs — we always have indicators)
        try:
            results["technical"] = self._run_technical(stock_info, indicators)
        except Exception as e:
            logger.warning("Technical agent failed: %s", e)
            results["technical"] = {"agent_name": "技术分析师", "analysis": f"分析失败: {e}"}

        # Fundamental (runs even without quarterly data — model uses basic info)
        try:
            results["fundamental"] = self._run_fundamental(stock_info, quarterly_text)
        except Exception as e:
            logger.warning("Fundamental agent failed: %s", e)
            results["fundamental"] = {"agent_name": "基本面分析师", "analysis": f"分析失败: {e}"}

        # Fund flow (runs even without data — model uses volume context)
        try:
            results["fund_flow"] = self._run_fund_flow(
                stock_info, indicators, fund_flow_text
            )
        except Exception as e:
            logger.warning("Fund flow agent failed: %s", e)
            results["fund_flow"] = {"agent_name": "资金面分析师", "analysis": f"分析失败: {e}"}

        return results

    def conduct_team_discussion(
        self, agent_results: dict, stock_info: dict
    ) -> str:
        """Simulate a team meeting to synthesize all agent reports."""
        logger.info("  [团队讨论] synthesizing...")
        tech_rpt = agent_results.get("technical", {}).get("analysis", "")
        fund_rpt = agent_results.get("fundamental", {}).get("analysis", "")
        flow_rpt = agent_results.get("fund_flow", {}).get("analysis", "")
        return self.client.comprehensive_discussion(
            tech_rpt, fund_rpt, flow_rpt, stock_info
        )

    def make_final_decision(
        self, discussion: str, stock_info: dict, indicators: dict
    ) -> dict:
        """Generate a structured investment decision from the discussion."""
        logger.info("  [最终决策] generating...")
        return self.client.final_decision(discussion, stock_info, indicators)

    def suggest_portfolio(self, picks_summary: str) -> dict:
        """Generate portfolio allocation advice for top picks."""
        logger.info("  [组合建议] generating...")
        return self.client.portfolio_suggestion(picks_summary)
