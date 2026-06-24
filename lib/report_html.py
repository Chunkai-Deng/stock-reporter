"""HTML report generator for stock analysis.

Generates clean, mobile-friendly HTML reports for:
  - Merged multi-stock report (all stocks in one HTML)
  - Single-stock technical analysis
  - Composite screening report (综合选股)
"""

import os
from datetime import datetime
from typing import Optional

# ── CSS (shared across all report types) ──────────────────────────────

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  background: #f5f6fa; color: #2d3436; padding: 12px;
  max-width: 600px; margin: 0 auto;
}
.card {
  background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.header { text-align: center; padding: 8px 0 16px; }
.stock-name { font-size: 20px; font-weight: 700; color: #2d3436; }
.stock-code { font-size: 13px; color: #636e72; margin-left: 6px; }
.price-row { margin-top: 10px; }
.price { font-size: 28px; font-weight: 800; }
.price.up { color: #e74c3c; }
.price.down { color: #27ae60; }
.change { font-size: 15px; font-weight: 600; margin-left: 6px; }
.change.up { color: #e74c3c; }
.change.down { color: #27ae60; }
.time-tag { font-size: 12px; color: #b2bec3; margin-top: 4px; }
.section-title {
  font-size: 14px; font-weight: 700; color: #636e72;
  padding: 8px 0 6px; border-bottom: 2px solid #dfe6e9; margin-bottom: 8px;
}
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; }
.metric { display: flex; justify-content: space-between; font-size: 13px; }
.metric .label { color: #636e72; }
.metric .value { font-weight: 600; color: #2d3436; }
.detail-text { font-size: 13px; line-height: 1.7; color: #2d3436; padding: 4px 0; }
.signal-box {
  padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 14px;
  font-weight: 700; text-align: center;
}
.signal-box.bullish { background: #fff3f3; color: #e74c3c; }
.signal-box.bearish { background: #f0fff4; color: #27ae60; }
.signal-box.neutral { background: #f8f9fa; color: #636e72; }
.score-badge {
  display: inline-block; padding: 3px 10px; border-radius: 12px;
  font-size: 13px; font-weight: 700; color: #fff;
}
.score-badge.high { background: #e74c3c; }
.score-badge.mid { background: #fdcb6e; color: #2d3436; }
.score-badge.low { background: #27ae60; }
.ai-section { background: #fafbff; border-left: 3px solid #6c5ce7; padding: 10px 12px;
               border-radius: 4px; margin: 8px 0; font-size: 13px; line-height: 1.7; }
.agent-tag {
  display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600; margin-right: 4px;
}
.agent-tag.tech { background: #ffeaa7; color: #d63031; }
.agent-tag.fund { background: #dfe6e9; color: #2d3436; }
.agent-tag.flow { background: #a29bfe; color: #fff; }
.decision-box {
  border: 2px solid #6c5ce7; border-radius: 10px; padding: 12px;
  margin: 10px 0; background: #fafbff;
}
.decision-title { font-size: 14px; font-weight: 700; color: #6c5ce7; margin-bottom: 6px; }
.decision-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; font-size: 13px; }
.decision-metrics .item { display: flex; justify-content: space-between; }
.footer {
  text-align: center; font-size: 11px; color: #b2bec3; padding: 12px 0 4px;
}
.divider { height: 1px; background: #dfe6e9; margin: 12px 0; }

.composite-header {
  text-align: center; padding: 12px 0; background: linear-gradient(135deg, #6c5ce7, #a29bfe);
  color: #fff; border-radius: 12px; margin-bottom: 12px;
}
.composite-title { font-size: 18px; font-weight: 800; }
.composite-subtitle { font-size: 12px; opacity: 0.85; margin-top: 4px; }
.pick-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #f0f0f0; }
.pick-rank { font-size: 22px; font-weight: 800; color: #6c5ce7; width: 32px; text-align: center; }
.pick-info { flex: 1; margin-left: 8px; }
.pick-name { font-size: 15px; font-weight: 700; }
.pick-code { font-size: 12px; color: #636e72; }
.pick-score {
  text-align: right; min-width: 48px;
  font-size: 16px; font-weight: 800; color: #e74c3c;
}
.pick-strategies { font-size: 11px; color: #636e72; margin-top: 2px; }
.portfolio-bar {
  display: flex; height: 28px; border-radius: 6px; overflow: hidden; margin: 8px 0;
}
.portfolio-bar .seg { display: flex; align-items: center; justify-content: center;
                       font-size: 11px; font-weight: 700; color: #fff; }
.portfolio-legend { font-size: 12px; display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.portfolio-legend .leg-item { display: flex; align-items: center; }
.portfolio-legend .swatch { width: 10px; height: 10px; border-radius: 2px; margin-right: 3px; }
.warning { font-size: 11px; color: #636e72; padding: 4px 0; }
"""

BAR_COLORS = ["#6c5ce7", "#e74c3c", "#fdcb6e", "#00b894", "#0984e3",
              "#e17055", "#a29bfe", "#fd79a8"]

# ═══════════════════════════════════════════════════════════════════════
# Single-stock report
# ═══════════════════════════════════════════════════════════════════════

def build_stock_report(
    name: str,
    code: str,
    price: float,
    change_pct: float,
    indicators: dict,
    weekly: dict,
    ai_text: Optional[str] = None,
    enhanced_section: Optional[str] = None,
) -> str:
    """Generate an HTML report for a single stock."""

    direction = "up" if change_pct >= 0 else "down"
    sign = "+" if change_pct >= 0 else ""
    t = datetime.now().strftime("%H:%M")

    def fmt(v, precision=2):
        if v is None:
            return "--"
        return f"{v:.{precision}f}"

    # Indicator unpack
    ma5 = indicators.get("ma5")
    ma10 = indicators.get("ma10")
    ma20 = indicators.get("ma20")
    macd_val = indicators.get("macd")
    macd_signal = indicators.get("macd_signal")
    macd_hist = indicators.get("macd_hist")
    rsi = indicators.get("rsi")
    k = indicators.get("k")
    d = indicators.get("d")
    j = indicators.get("j")
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_width = indicators.get("bb_width_pct")
    support = indicators.get("support")
    resistance = indicators.get("resistance")
    vol_ratio = indicators.get("vol_ratio")
    vol_trend = indicators.get("vol_trend", "")
    adx = indicators.get("adx")
    plus_di = indicators.get("plus_di")
    minus_di = indicators.get("minus_di")
    divergence = indicators.get("divergence", "")
    macd_cross = indicators.get("macd_cross", "")
    weekly_trend = weekly.get("weekly_trend", "")

    # Score
    score = _score_stock(price, change_pct, indicators, weekly)
    score_class = "high" if score >= 3 else ("mid" if score >= 0 else "low")

    # Signal
    signal_text, signal_class = _signal_display(score, divergence)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} ({code})</title>
<style>{CSS}</style>
</head>
<body>

<div class="header">
  <div class="stock-name">{name}<span class="stock-code">{code}</span></div>
  <div class="price-row">
    <span class="price {direction}">¥{price:.2f}</span>
    <span class="change {direction}">{sign}{change_pct:.2f}%</span>
  </div>
  <div class="time-tag">{t}</div>
</div>

<div class="card">
  <div class="signal-box {signal_class}">{signal_text}</div>
  <div style="text-align:center;">
    <span class="score-badge {score_class}">综合评分: {score}/10</span>
  </div>
</div>

<div class="card">
  <div class="section-title">📈 均线系统</div>
  <div class="metrics">
    <div class="metric"><span class="label">MA5</span><span class="value">{fmt(ma5)}</span></div>
    <div class="metric"><span class="label">MA10</span><span class="value">{fmt(ma10)}</span></div>
    <div class="metric"><span class="label">MA20</span><span class="value">{fmt(ma20)}</span></div>
    <div class="metric"><span class="label">周线趋势</span><span class="value">{weekly_trend or '未知'}</span></div>
  </div>
</div>

<div class="card">
  <div class="section-title">📉 MACD</div>
  <div class="metrics">
    <div class="metric"><span class="label">DIF</span><span class="value">{fmt(macd_val)}</span></div>
    <div class="metric"><span class="label">DEA</span><span class="value">{fmt(macd_signal)}</span></div>
    <div class="metric"><span class="label">柱</span><span class="value">{fmt(macd_hist)}</span></div>
    <div class="metric"><span class="label">信号</span><span class="value">{macd_cross or '--'}</span></div>
  </div>
</div>

<div class="card">
  <div class="section-title">⚡ 超买超卖</div>
  <div class="metrics">
    <div class="metric"><span class="label">RSI(14)</span><span class="value">{fmt(rsi, 1)}</span></div>
    <div class="metric"><span class="label">K</span><span class="value">{fmt(k, 1)}</span></div>
    <div class="metric"><span class="label">D</span><span class="value">{fmt(d, 1)}</span></div>
    <div class="metric"><span class="label">J</span><span class="value">{fmt(j, 1)}</span></div>
  </div>
</div>

<div class="card">
  <div class="section-title">📐 布林带 (20,2)</div>
  <div class="metrics">
    <div class="metric"><span class="label">上轨</span><span class="value">{fmt(bb_upper)}</span></div>
    <div class="metric"><span class="label">下轨</span><span class="value">{fmt(bb_lower)}</span></div>
    <div class="metric"><span class="label">带宽</span><span class="value">{fmt(bb_width, 1)}%</span></div>
    <div class="metric"><span class="label">量比</span><span class="value">{fmt(vol_ratio)}</span></div>
  </div>
  <div style="font-size:12px;color:#636e72;margin-top:4px;">成交量: {vol_trend}</div>
</div>

<div class="card">
  <div class="section-title">🎯 支撑/压力 & 趋势强度</div>
  <div class="metrics">
    <div class="metric"><span class="label">支撑位</span><span class="value">{fmt(support)}</span></div>
    <div class="metric"><span class="label">压力位</span><span class="value">{fmt(resistance)}</span></div>
    <div class="metric"><span class="label">ADX</span><span class="value">{fmt(adx, 1)}</span></div>
    <div class="metric"><span class="label">+DI / -DI</span><span class="value">{fmt(plus_di,1)}/{fmt(minus_di,1)}</span></div>
  </div>
  <div style="font-size:12px;color:#636e72;margin-top:4px;">背离: {divergence or '无'}</div>
</div>
"""

    # AI section
    if ai_text:
        html += f"""
<div class="card">
  <div class="section-title">🤖 AI 视角</div>
  <div class="ai-section">{ai_text}</div>
</div>
"""

    # Enhanced multi-agent section
    if enhanced_section:
        html += enhanced_section

    html += f"""
<div class="footer">⚠️ 以上分析仅供参考，不构成投资建议 | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</body></html>"""
    return html


def build_enhanced_html(agent_results: dict, discussion: str, decision: dict) -> str:
    """Build the multi-agent analysis HTML block."""

    parts = ['<div class="card"><div class="section-title">🧠 多智能体分析</div>']

    # Technical
    tech = agent_results.get("technical", {}).get("analysis", "")
    if tech:
        parts.append(f'<div style="margin:6px 0;"><span class="agent-tag tech">技术面</span></div>')
        parts.append(f'<div class="detail-text">{tech[:300]}</div>')

    # Fundamental
    fund = agent_results.get("fundamental", {}).get("analysis", "")
    if fund:
        parts.append(f'<div style="margin:6px 0;"><span class="agent-tag fund">基本面</span></div>')
        parts.append(f'<div class="detail-text">{fund[:300]}</div>')

    # Fund flow
    flow = agent_results.get("fund_flow", {}).get("analysis", "")
    if flow:
        parts.append(f'<div style="margin:6px 0;"><span class="agent-tag flow">资金面</span></div>')
        parts.append(f'<div class="detail-text">{flow[:300]}</div>')

    # Discussion
    if discussion:
        parts.append(f'<div style="margin:8px 0;font-size:13px;line-height:1.7;"><strong>团队讨论:</strong> {discussion[:400]}</div>')

    # Decision
    if decision:
        rating = decision.get("rating", "N/A")
        target = decision.get("target_price", "N/A")
        stop_loss = decision.get("stop_loss", "N/A")
        take_profit = decision.get("take_profit", "")
        position = decision.get("position_size", "N/A")
        confidence = decision.get("confidence_level", "N/A")
        risk = decision.get("risk_warning", decision.get("decision_text", ""))

        parts.append('<div class="decision-box">')
        parts.append('<div class="decision-title">📋 最终决策</div>')
        parts.append('<div class="decision-metrics">')
        parts.append(f'<div class="item"><span>评级</span><span><strong>{rating}</strong></span></div>')
        parts.append(f'<div class="item"><span>目标价</span><span>{target}</span></div>')
        parts.append(f'<div class="item"><span>止损位</span><span>{stop_loss}</span></div>')
        if take_profit:
            parts.append(f'<div class="item"><span>止盈</span><span>{take_profit}</span></div>')
        parts.append(f'<div class="item"><span>仓位</span><span>{position}</span></div>')
        parts.append(f'<div class="item"><span>信心度</span><span>{confidence}/10</span></div>')
        parts.append('</div>')
        if isinstance(risk, str) and len(risk) < 200:
            parts.append(f'<div style="font-size:12px;color:#636e72;margin-top:6px;">风险: {risk}</div>')
        parts.append('</div>')

    parts.append('</div>')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Composite screening report
# ═══════════════════════════════════════════════════════════════════════

def build_composite_report(full_msg: str) -> str:
    """Wrap the composite screening text output in clean HTML.

    Parses the plain-text composite output and renders it as a styled HTML card.
    """
    t = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>综合选股</title>
<style>{CSS}</style>
</head>
<body>

<div class="composite-header">
  <div class="composite-title">📋 综合选股</div>
  <div class="composite-subtitle">预筛选 + 5策略交叉打分 + AI组合建议 | {t}</div>
</div>

<div class="card">
  <div class="detail-text" style="white-space:pre-wrap;">{_escape_html(full_msg)}</div>
</div>

<div class="footer">⚠️ 以上为量化筛选，仅供参考，不构成投资建议</div>
</body></html>"""
    return html


def build_portfolio_html(picks: list, portfolio: dict) -> str:
    """Build an HTML representation of the portfolio allocation."""
    parts = ['<div class="card"><div class="section-title">💼 AI 组合建议</div>']

    total_pos = portfolio.get("total_position", "N/A")
    advice = portfolio.get("overall_advice", "")
    allocation = portfolio.get("allocation", [])

    parts.append(f'<div style="font-size:13px;margin-bottom:8px;">总仓位: <strong>{total_pos}</strong></div>')

    if allocation:
        # Portfolio bar
        parts.append('<div class="portfolio-bar">')
        for i, alloc in enumerate(allocation):
            pct_str = alloc.get("ratio", "0%").replace("%", "")
            try:
                pct = float(pct_str)
            except ValueError:
                pct = 0
            color = BAR_COLORS[i % len(BAR_COLORS)]
            parts.append(
                f'<div class="seg" style="width:{pct}%;background:{color};">'
                f'{alloc.get("code", "")[:6]}</div>'
            )
        parts.append('</div>')

        # Legend
        parts.append('<div class="portfolio-legend">')
        for i, alloc in enumerate(allocation):
            color = BAR_COLORS[i % len(BAR_COLORS)]
            parts.append(
                f'<div class="leg-item">'
                f'<span class="swatch" style="background:{color};"></span>'
                f'{alloc.get("code", "")} {alloc.get("name", "")} '
                f'<strong>{alloc.get("ratio", "")}</strong> ({alloc.get("style", "")})'
                f'</div>'
            )
        parts.append('</div>')

    if advice:
        parts.append(f'<div class="ai-section" style="margin-top:8px;">{advice}</div>')

    parts.append('</div>')
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Merged multi-stock report
# ═══════════════════════════════════════════════════════════════════════

def build_merged_report(stocks: list, title: str = "股票监控报告",
                        screening_section: str = "") -> str:
    """Build a single HTML report containing multiple stock analyses.

    Each stock in the list should be a dict from process_stock():
      {code, name, price, change_pct, indicators, weekly, ai_text,
       agent_results, discussion, decision}
    screening_section: optional pre-formatted screening text to show at top.
    """
    t = datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(stocks)

    # Quick stats
    up_count = sum(1 for s in stocks if s.get("change_pct", 0) >= 0)
    down_count = n - up_count

    def fmt(v, precision=2):
        if v is None:
            return "--"
        return f"{v:.{precision}f}"

    rows = ""
    for s in stocks:
        direction = "up" if s.get("change_pct", 0) >= 0 else "down"
        sign = "+" if s.get("change_pct", 0) >= 0 else ""
        score = _score_stock(s["price"], s.get("change_pct", 0), s["indicators"], s["weekly"])
        score_class = "high" if score >= 3 else ("mid" if score >= 0 else "low")
        signal_text, signal_class = _signal_display(score, s["indicators"].get("divergence", ""))

        ind = s["indicators"]
        w = s["weekly"]

        # Build each stock card
        rows += f"""
<div class="card">
  <div class="stock-card-header">
    <div>
      <span class="stock-name">{s['name']}</span>
      <span class="stock-code">{s['code']}</span>
    </div>
    <div class="stock-summary">
      <span class="price {direction}">¥{s['price']:.2f}</span>
      <span class="change {direction}">{sign}{s['change_pct']:.2f}%</span>
      <span class="score-badge {score_class}">{score}/10</span>
    </div>
  </div>

  <div class="signal-box {signal_class}">{signal_text}</div>

  <div class="section-title">📈 技术速览</div>
  <div class="metrics">
    <div class="metric"><span class="label">MA5 / MA10 / MA20</span><span class="value">{fmt(ind.get('ma5'))} / {fmt(ind.get('ma10'))} / {fmt(ind.get('ma20'))}</span></div>
    <div class="metric"><span class="label">MACD 信号</span><span class="value">{ind.get('macd_cross') or '--'} 柱={fmt(ind.get('macd_hist'))}</span></div>
    <div class="metric"><span class="label">RSI / KDJ-J</span><span class="value">{fmt(ind.get('rsi'),1)} / {fmt(ind.get('j'),1)}</span></div>
    <div class="metric"><span class="label">布林带宽 / 量比</span><span class="value">{fmt(ind.get('bb_width_pct'),1)}% / {fmt(ind.get('vol_ratio'))} ({ind.get('vol_trend','--')})</span></div>
    <div class="metric"><span class="label">支撑 / 压力</span><span class="value">{fmt(ind.get('support'))} / {fmt(ind.get('resistance'))}</span></div>
    <div class="metric"><span class="label">ADX / +DI / -DI</span><span class="value">{fmt(ind.get('adx'),1)} / {fmt(ind.get('plus_di'),1)} / {fmt(ind.get('minus_di'),1)}</span></div>
    <div class="metric"><span class="label">周线 / 背离</span><span class="value">{w.get('weekly_trend','--')} / {ind.get('divergence') or '无'}</span></div>
  </div>
</div>
"""

        # AI text
        ai = s.get("ai_text")
        if ai:
            rows += f"""
<div class="card">
  <div class="section-title">🤖 {s['name']} AI 视角</div>
  <div class="ai-section">{ai}</div>
</div>
"""

        # Enhanced multi-agent
        if s.get("agent_results") and s.get("decision"):
            rows += build_enhanced_html(s["agent_results"], s.get("discussion", ""), s["decision"])

    # Screening section (first report of the day only)
    screening_html = ""
    if screening_section:
        screening_escaped = _escape_html(screening_section)
        screening_html = f"""
<div class="card" style="border-left: 3px solid #fdcb6e;">
  <div class="section-title">🔍 每日选股扫描</div>
  <div class="detail-text" style="white-space:pre-wrap;">{screening_escaped}</div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", sans-serif;
  background: #f5f6fa; color: #2d3436; padding: 12px;
  max-width: 600px; margin: 0 auto;
}}
.card {{
  background: #fff; border-radius: 12px; padding: 16px; margin-bottom: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}
.header {{ text-align: center; padding: 12px 0; }}
.header-title {{ font-size: 18px; font-weight: 800; color: #2d3436; }}
.header-time {{ font-size: 12px; color: #b2bec3; margin-top: 2px; }}
.header-stats {{ font-size: 13px; color: #636e72; margin-top: 6px; }}
.stock-card-header {{
  display: flex; justify-content: space-between; align-items: center;
  padding-bottom: 8px; border-bottom: 1px solid #f0f0f0; margin-bottom: 8px;
}}
.stock-summary {{ text-align: right; }}
.stock-name {{ font-size: 16px; font-weight: 700; }}
.stock-code {{ font-size: 12px; color: #636e72; margin-left: 4px; }}
.price {{ font-size: 20px; font-weight: 800; }}
.price.up {{ color: #e74c3c; }}
.price.down {{ color: #27ae60; }}
.change {{ font-size: 13px; font-weight: 600; margin: 0 6px; }}
.change.up {{ color: #e74c3c; }}
.change.down {{ color: #27ae60; }}
.score-badge {{
  display: inline-block; padding: 2px 8px; border-radius: 12px;
  font-size: 12px; font-weight: 700; color: #fff; vertical-align: middle;
}}
.score-badge.high {{ background: #e74c3c; }}
.score-badge.mid {{ background: #fdcb6e; color: #2d3436; }}
.score-badge.low {{ background: #27ae60; }}
.section-title {{
  font-size: 13px; font-weight: 700; color: #636e72;
  padding: 6px 0; border-bottom: 2px solid #dfe6e9; margin-bottom: 6px;
}}
.metrics {{ display: grid; grid-template-columns: 1fr; gap: 4px; }}
.metric {{ display: flex; justify-content: space-between; font-size: 12px; }}
.metric .label {{ color: #636e72; }}
.metric .value {{ font-weight: 600; color: #2d3436; }}
.signal-box {{
  padding: 10px; border-radius: 8px; margin: 6px 0; font-size: 13px;
  font-weight: 700; text-align: center;
}}
.signal-box.bullish {{ background: #fff3f3; color: #e74c3c; }}
.signal-box.bearish {{ background: #f0fff4; color: #27ae60; }}
.signal-box.neutral {{ background: #f8f9fa; color: #636e72; }}
.ai-section {{
  background: #fafbff; border-left: 3px solid #6c5ce7; padding: 10px 12px;
  border-radius: 4px; font-size: 13px; line-height: 1.7;
}}
.agent-tag {{
  display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600; margin-right: 4px;
}}
.agent-tag.tech {{ background: #ffeaa7; color: #d63031; }}
.agent-tag.fund {{ background: #dfe6e9; color: #2d3436; }}
.agent-tag.flow {{ background: #a29bfe; color: #fff; }}
.decision-box {{
  border: 2px solid #6c5ce7; border-radius: 10px; padding: 12px;
  margin: 10px 0; background: #fafbff;
}}
.decision-title {{ font-size: 14px; font-weight: 700; color: #6c5ce7; margin-bottom: 6px; }}
.decision-metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; font-size: 12px; }}
.decision-metrics .item {{ display: flex; justify-content: space-between; }}
.detail-text {{ font-size: 13px; line-height: 1.7; color: #2d3436; padding: 4px 0; }}
.footer {{
  text-align: center; font-size: 11px; color: #b2bec3; padding: 12px 0 8px;
}}
.divider {{ height: 2px; background: #6c5ce7; margin: 8px 0 12px; border-radius: 1px; }}
</style>
</head>
<body>

<div class="header">
  <div class="header-title">📊 {title}</div>
  <div class="header-time">{t} | {n} 只股票 | {up_count}涨{down_count}跌</div>
  <div class="divider"></div>
</div>
{screening_html}{rows}
<div class="footer">⚠️ 以上分析仅供参考，不构成投资建议</div>
</body></html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════
# Merged multi-stock report (end)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _score_stock(price, change_pct, indicators, weekly) -> int:
    """Recompute score inline (same logic as cloud_stock_reporter.score_stock)."""
    score = 0
    # MA
    if price and indicators.get("ma20"):
        score += 1 if price > indicators["ma20"] else -1
    ma5, ma10 = indicators.get("ma5"), indicators.get("ma10")
    if ma5 and ma10:
        score += 1 if ma5 > ma10 else -1
    # MACD cross
    cross = indicators.get("macd_cross", "")
    if cross == "金叉":
        score += 1
    elif cross == "死叉":
        score -= 1
    # RSI
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
    # Bollinger
    if price and indicators.get("bb_upper") and indicators.get("bb_lower"):
        if price >= indicators["bb_upper"] * 0.99:
            score -= 1
        elif price <= indicators["bb_lower"] * 1.01:
            score += 1
    # Volume
    vol_trend = indicators.get("vol_trend", "")
    if vol_trend == "放量":
        score += 1 if change_pct >= 0 else -1
    # KDJ
    j = indicators.get("j")
    if j is not None:
        if j > 100:
            score -= 1
        elif j < 0:
            score += 1
    # ADX
    adx = indicators.get("adx")
    if adx is not None and adx > 40:
        plus_di = indicators.get("plus_di")
        minus_di = indicators.get("minus_di")
        if plus_di and minus_di and plus_di > minus_di:
            score += 1
        else:
            score -= 1
    # Divergence
    div = indicators.get("divergence", "")
    if div == "顶背离":
        score -= 2
    elif div == "底背离":
        score += 2
    # Weekly
    wt = weekly.get("weekly_trend", "")
    if wt == "上涨":
        score += 1
    elif wt == "下跌":
        score -= 1
    return max(-10, min(10, score))


def _signal_display(score: int, divergence: str) -> tuple:
    """Return (text, css_class) for the signal display."""
    if divergence == "顶背离" and score <= -1:
        return "⚠️ 卖出信号 — 顶背离", "bearish"
    elif divergence == "底背离" and score >= 0:
        return "🔥 买入信号 — 底背离", "bullish"
    elif score >= 5:
        return "📈 强烈看涨，持仓待涨", "bullish"
    elif score >= 3:
        return "📈 偏多，逢低可加仓", "bullish"
    elif score >= 1:
        return "📊 震荡偏多", "neutral"
    elif score <= -5:
        return "📉 强烈看跌，建议清仓", "bearish"
    elif score <= -3:
        return "📉 偏空，减仓或止盈", "bearish"
    elif score <= -1:
        return "📊 震荡偏弱", "neutral"
    else:
        return "📊 观望 — 等信号明确", "neutral"


def _escape_html(text: str) -> str:
    """Escape HTML entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def save_report(html: str, prefix: str = "stock", subdir: str = "") -> str:
    """Save an HTML report to the reports directory. Returns the file path."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if subdir:
        report_dir = os.path.join(base, "reports", subdir)
    else:
        report_dir = os.path.join(base, "reports")
    os.makedirs(report_dir, exist_ok=True)

    ts = datetime.now().strftime("%H%M%S")
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d')}_{ts}.html"
    filepath = os.path.join(report_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    return filepath
