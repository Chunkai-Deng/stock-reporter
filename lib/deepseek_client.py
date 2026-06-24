"""DeepSeek API client via OpenAI SDK.

Adapted from aiagents-stock/deepseek_client.py.
Provides specialized analysis prompts for multi-agent stock analysis.
"""

import json
import logging
import re
from typing import Any

from openai import OpenAI

from lib.config import get_config

logger = logging.getLogger("stock_reporter.deepseek")


class DeepSeekClient:
    """DeepSeek API client with specialized stock analysis prompts."""

    def __init__(self, model: str | None = None):
        cfg = get_config()
        self.model = model or cfg.deepseek_model
        self.client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
        )

    # ── Core API call ────────────────────────────────────────────────

    def call_api(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generic DeepSeek API call with reasoner model support."""
        model_to_use = model or self.model

        if "reasoner" in model_to_use.lower() and max_tokens <= 2000:
            max_tokens = 8000

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            message = response.choices[0].message

            # Return only the final content, skip reasoning process
            if message.content:
                return message.content

            return "API返回空响应"

        except Exception as e:
            logger.warning("DeepSeek API call failed: %s", e)
            return f"API调用失败: {e}"

    # ── Technical Analysis ───────────────────────────────────────────

    def technical_analysis(
        self,
        stock_info: dict,
        indicators: dict,
    ) -> str:
        """Technical analysis agent prompt."""
        prompt = f"""
你是一名资深的技术分析师。请基于以下股票数据进行专业的技术面分析：

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 涨跌幅：{stock_info.get('change_percent', 'N/A')}%

最新技术指标：
- 收盘价：{indicators.get('price', indicators.get('close', 'N/A'))}
- MA5：{indicators.get('ma5', 'N/A')}
- MA10：{indicators.get('ma10', 'N/A')}
- MA20：{indicators.get('ma20', 'N/A')}
- RSI：{indicators.get('rsi', 'N/A')}
- MACD DIF：{indicators.get('macd', 'N/A')}
- MACD DEA：{indicators.get('macd_signal', 'N/A')}
- MACD 柱：{indicators.get('macd_hist', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}
- 布林带中轨：{indicators.get('bb_middle', 'N/A')}
- KDJ-K：{indicators.get('k', 'N/A')}
- KDJ-D：{indicators.get('d', 'N/A')}
- KDJ-J：{indicators.get('j', 'N/A')}
- 量比：{indicators.get('vol_ratio', 'N/A')}
- 成交量趋势：{indicators.get('vol_trend', 'N/A')}
- ADX：{indicators.get('adx', 'N/A')}
- +DI：{indicators.get('plus_di', 'N/A')}
- -DI：{indicators.get('minus_di', 'N/A')}
- 支撑位：{indicators.get('support', 'N/A')}
- 压力位：{indicators.get('resistance', 'N/A')}
- 背离信号：{indicators.get('divergence', '无')}
- MACD交叉：{indicators.get('macd_cross', '无')}

请从以下角度进行分析：
1. 趋势分析（均线系统、价格走势）
2. 超买超卖分析（RSI、KDJ）
3. 动量分析（MACD、ADX）
4. 支撑阻力分析（布林带、60日高低点）
5. 成交量分析（量比、量价配合）
6. 背离与交叉信号
7. 短期、中期技术判断
8. 关键技术位分析

请给出专业、详细的技术分析报告，包含风险提示。控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一名经验丰富的股票技术分析师，具有深厚的技术分析功底。请简洁务实，避免模棱两可。"},
            {"role": "user", "content": prompt},
        ]
        return self.call_api(messages)

    # ── Fundamental Analysis ─────────────────────────────────────────

    def fundamental_analysis(
        self,
        stock_info: dict,
        quarterly_text: str = "",
    ) -> str:
        """Fundamental analysis agent prompt.

        Args:
            stock_info: dict with symbol, name, current_price, change_percent
            quarterly_text: pre-formatted quarterly report text (from EnhancedDataManager)
        """
        quarterly_section = ""
        if quarterly_text:
            quarterly_section = f"""

【最近8期季报详细数据】
{quarterly_text}

以上是通过akshare获取的最近8期季度财务报告，请重点基于这些数据进行趋势分析。
"""

        prompt = f"""
你是一名资深的基本面分析师，拥有CFA资格和10年以上的证券分析经验。请基于以下信息进行深入的基本面分析：

【基本信息】
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}
- 涨跌幅：{stock_info.get('change_percent', 'N/A')}%
{quarterly_section}

请从以下维度进行分析：

1. **盈利能力分析** — ROE/ROA水平、毛利率和净利率趋势
2. **财务健康度分析** — 资产负债结构、偿债能力、现金流状况
3. **成长性分析** — 收入和利润增长趋势、增长驱动因素
4. **季报趋势分析**（如有季报数据）— 营收/利润/现金流变化趋势、季度环比/同比
5. **估值分析** — 当前估值水平是否合理
6. **投资价值判断** — 综合评分、投资亮点、主要风险

请给出专业、详细的基本面分析报告。控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一名经验丰富的股票基本面分析师，擅长公司财务分析和行业研究。请简洁务实。"},
            {"role": "user", "content": prompt},
        ]
        return self.call_api(messages)

    # ── Fund Flow Analysis ───────────────────────────────────────────

    def fund_flow_analysis(
        self,
        stock_info: dict,
        indicators: dict,
        fund_flow_text: str = "",
    ) -> str:
        """Fund flow analysis agent prompt.

        Args:
            stock_info: dict with symbol, name, current_price, change_percent
            indicators: dict of technical indicators (for volume context)
            fund_flow_text: pre-formatted fund flow data text (from EnhancedDataManager)
        """
        fund_flow_section = ""
        if fund_flow_text:
            fund_flow_section = f"""

【近20个交易日资金流向详细数据】
{fund_flow_text}

以上是通过akshare从东方财富获取的实际资金流向数据，请重点基于这些数据进行趋势分析。
"""
        else:
            fund_flow_section = "\n注意：未能获取到资金流向数据，将基于成交量进行分析。\n"

        prompt = f"""
你是一名资深的资金面分析师，擅长从资金流向数据中洞察主力行为和市场趋势。

【基本信息】
股票代码：{stock_info.get('symbol', 'N/A')}
股票名称：{stock_info.get('name', 'N/A')}
当前价格：{stock_info.get('current_price', 'N/A')}
量比：{indicators.get('vol_ratio', 'N/A')}
成交量趋势：{indicators.get('vol_trend', 'N/A')}
{fund_flow_section}

请从以下角度分析：

1. **资金流向趋势** — 主力资金累计净流入/流出、趋势方向
2. **主力行为研判** — 吸筹建仓/派发出货/洗盘整理/拉升推动
3. **散户资金行为** — 主力与散户的博弈关系
4. **量价配合分析** — 资金流向与股价涨跌的配合度
5. **关键信号识别** — 买入/卖出/观望信号
6. **操作建议** — 基于资金面的明确判断

分析原则：
- 主力持续流入 + 股价上涨 → 强势信号
- 主力流出 + 股价上涨 → 警惕信号
- 主力流入 + 股价下跌 → 可能低位吸筹
- 主力流出 + 股价下跌 → 弱势信号

请给出专业、详细的资金面分析报告。控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一名经验丰富的资金面分析师，擅长市场资金流向和主力行为分析。请简洁务实。"},
            {"role": "user", "content": prompt},
        ]
        return self.call_api(messages, max_tokens=3000)

    # ── Team Discussion ──────────────────────────────────────────────

    def comprehensive_discussion(
        self,
        technical_report: str,
        fundamental_report: str,
        fund_flow_report: str,
        stock_info: dict,
    ) -> str:
        """Simulate a team investment meeting synthesizing all agent reports."""
        prompt = f"""
现在需要进行一场投资决策会议，你作为首席分析师，需要综合各位分析师的报告进行讨论。

股票基本信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

技术面分析报告：
{technical_report}

基本面分析报告：
{fundamental_report}

资金面分析报告：
{fund_flow_report}

请作为首席分析师，综合以上三个维度的分析报告，进行深入讨论：
1. 各个分析维度的一致性和分歧点
2. 不同分析结论的权重考量
3. 当前市场环境下的投资逻辑
4. 潜在风险和机会识别

请模拟一场专业的投资讨论会议，体现不同观点的碰撞和融合。控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一名资深的首席投资分析师，擅长综合不同维度的分析形成投资判断。请简洁务实。"},
            {"role": "user", "content": prompt},
        ]
        return self.call_api(messages, max_tokens=4000)

    # ── Final Decision ───────────────────────────────────────────────

    def final_decision(
        self,
        comprehensive_discussion: str,
        stock_info: dict,
        indicators: dict,
    ) -> dict[str, Any]:
        """Generate a structured investment decision as JSON."""
        prompt = f"""
基于前期的综合分析讨论，现在需要做出最终的投资决策。

股票信息：
- 股票代码：{stock_info.get('symbol', 'N/A')}
- 股票名称：{stock_info.get('name', 'N/A')}
- 当前价格：{stock_info.get('current_price', 'N/A')}

综合分析讨论结果：
{comprehensive_discussion}

当前关键技术位：
- MA20：{indicators.get('ma20', 'N/A')}
- 布林带上轨：{indicators.get('bb_upper', 'N/A')}
- 布林带下轨：{indicators.get('bb_lower', 'N/A')}
- 支撑位：{indicators.get('support', 'N/A')}
- 压力位：{indicators.get('resistance', 'N/A')}

请给出最终投资决策，以JSON格式输出。**硬性约束（必须遵守，否则格式错误）**：
- target_price（目标价位）：该股如果上涨可能达到的价格，必须**高于**当前价格
- stop_loss（止损价位）：该股如果下跌需要止损的价格，必须**低于**当前价格
- 要确保 target_price > stop_loss（目标价严格大于止损价），这个约束与评级无关，永远成立
- 如果是卖出评级，目标价可以保守（接近现价），止损价可以设近一些

{{
    "rating": "买入/持有/卖出",
    "target_price": "目标价位（必须高于现价）",
    "stop_loss": "止损价位（必须低于现价）",
    "take_profit": "止盈价位（可选，≥ target_price）",
    "holding_period": "持有周期",
    "position_size": "仓位建议（轻仓/中等仓位/重仓/空仓）",
    "risk_warning": "风险提示",
    "confidence_level": "信心度(1-10分)"
}}
"""
        messages = [
            {"role": "system", "content": "你是一名专业的投资决策专家，需要给出明确、可执行的投资建议。请输出JSON格式。"},
            {"role": "user", "content": prompt},
        ]
        response = self.call_api(messages, temperature=0.3, max_tokens=2000)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"decision_text": response}
        except Exception:
            return {"decision_text": response}

    # ── Portfolio Suggestion ───────────────────────────────────────────

    def portfolio_suggestion(
        self,
        picks_summary: str,
        total_stocks: int = 5,
    ) -> dict[str, Any]:
        """Generate portfolio allocation advice for the top stock picks.

        Args:
            picks_summary: Pre-formatted string of top stocks (code/name/price/score/industry)
            total_stocks: Number of stocks in the portfolio

        Returns:
            dict with allocation, total_position, overall_advice, watch_list
        """
        prompt = f"""
你是一名资深投资组合经理。请基于以下精选股票，设计一个投资组合方案。

精选股票（共{total_stocks}只）：
{picks_summary}

请生成一个投资组合分配方案，以JSON格式输出：

{{
    "allocation": [
        {{"code": "股票代码", "name": "股票名称", "ratio": "仓位占比%", "style": "波段/中线/短线"}},
        ...
    ],
    "total_position": "建议总仓位（如：6成）",
    "overall_advice": "整体操作思路和风险提示（100字以内）",
    "watch_list": ["备选股票代码1", "备选股票代码2"]
}}

要求：
- 仓位分配要考虑分散风险，单一股票不超过30%
- style 根据评分和策略特征判断：高评分+主力资金→波段，低估值+高增长→中线，小市值+动量→短线
- total_position 根据整体市场环境和选股质量判断
- 总仓位各分配比例之和应为100%
"""
        messages = [
            {"role": "system", "content": "你是一名专业的投资组合经理，擅长资产配置和风险管理。请输出JSON格式。"},
            {"role": "user", "content": prompt},
        ]
        response = self.call_api(messages, temperature=0.4, max_tokens=2000)

        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"overall_advice": response}
        except Exception:
            return {"overall_advice": response}
