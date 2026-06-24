"""Macro analysis AI agents.

Uses DeepSeek to analyze macro economic data and assess policy direction.
"""

import logging

from lib.deepseek_client import DeepSeekClient
from lib.config import get_config

logger = logging.getLogger("stock_reporter.macro_agents")


class MacroAgents:
    """AI agents for macro economic analysis."""

    def __init__(self, model: str | None = None):
        cfg = get_config()
        self.model = model or cfg.deepseek_model
        self.client = DeepSeekClient(model=self.model)

    def macro_analyst(self, context_text: str) -> str:
        """Analyze macro economic conditions."""
        prompt = f"""
你是一位资深宏观经济学家。请基于以下中国宏观数据分析当前经济形势：

{context_text}

请从以下角度分析：
1. **经济周期判断** — 当前处于复苏/过热/滞胀/衰退的哪个阶段？依据是什么？
2. **增长分析** — GDP增速趋势，工业生产的强弱
3. **通胀分析** — CPI/PPI走势，是否存在通胀或通缩压力
4. **流动性分析** — M2增速与信贷环境
5. **PMI信号** — 制造业和非制造业PMI反映的经济景气度

请给出简洁清晰的判断，控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一位资深宏观经济学家，擅长基于数据判断经济周期。请简洁务实。"},
            {"role": "user", "content": prompt},
        ]
        return self.client.call_api(messages)

    def policy_analyst(self, context_text: str) -> str:
        """Analyze policy and monetary stance."""
        prompt = f"""
你是一位资深政策分析师，专注于中国货币政策和财政政策研判。请基于以下数据进行分析：

{context_text}

请从以下角度分析：
1. **货币政策方向** — 当前偏宽松/中性/偏紧？依据M2、利率、降准降息信号
2. **财政政策力度** — 积极的/稳健的/紧缩的？依据政府支出、基建投资数据
3. **对A股的影响** — 当前政策组合对股市的风格切换有何影响？大盘蓝筹vs中小成长哪个更受益？
4. **关键风险** — 政策层面的主要不确定因素

请给出简洁清晰的判断，控制在200字以内。
"""
        messages = [
            {"role": "system", "content": "你是一位资深政策分析师，擅长研判货币和财政政策方向。请简洁务实。"},
            {"role": "user", "content": prompt},
        ]
        return self.client.call_api(messages)

    def run_macro_analysis(self, context_text: str) -> dict:
        """Run both macro agents and return combined results."""
        logger.info("Running macro analysis agents...")

        macro_report = ""
        policy_report = ""

        try:
            macro_report = self.macro_analyst(context_text)
        except Exception as e:
            logger.warning("Macro analyst failed: %s", e)
            macro_report = f"宏观分析暂不可用: {e}"

        try:
            policy_report = self.policy_analyst(context_text)
        except Exception as e:
            logger.warning("Policy analyst failed: %s", e)
            policy_report = f"政策分析暂不可用: {e}"

        return {
            "macro_analysis": macro_report,
            "policy_analysis": policy_report,
        }
