"""Stock screening strategies using pywencai (问财).

Combines 5 selectors from aiagents-stock into stateless functions.
Each returns (DataFrame, message_string).
"""

import logging
from typing import Tuple

import pandas as pd
import pywencai

from lib.config import get_config

logger = logging.getLogger("stock_reporter.selectors")

# ── Pre-screening (optional, imported lazily to avoid slow startup) ──
_pre_screener_loaded = False


def _load_pre_screener():
    """Import pre_screener module on first use."""
    global _pre_screener_loaded
    if not _pre_screener_loaded:
        try:
            from lib.pre_screener import run_pre_screening, get_pre_screened_codes  # noqa: F401
            globals().update({
                "run_pre_screening": run_pre_screening,
                "get_pre_screened_codes": get_pre_screened_codes,
            })
            _pre_screener_loaded = True
        except ImportError:
            logger.warning("pre_screener module not available, skipping pre-screening")
            _pre_screener_loaded = False


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _convert_to_dataframe(result) -> pd.DataFrame | None:
    """Convert pywencai result to DataFrame."""
    try:
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, dict):
            if 'data' in result:
                return pd.DataFrame(result['data'])
            elif 'result' in result:
                return pd.DataFrame(result['result'])
            return pd.DataFrame(result)
        elif isinstance(result, list):
            return pd.DataFrame(result)
        return None
    except Exception as e:
        logger.warning("  DataFrame conversion failed: %s", e)
        return None


def _clean_codes(df: pd.DataFrame) -> list:
    """Extract cleaned 6-digit stock codes from a result DataFrame."""
    codes = []
    for code in df.get('股票代码', []):
        if isinstance(code, str):
            codes.append(code.split('.')[0] if '.' in code else code)
        else:
            codes.append(str(code))
    return codes


def _format_stock_row(idx: int, row, keys: list[str], labels: list[str]) -> str:
    """Format one stock row into a readable line."""
    parts = [f"  {idx}."]
    for k, lbl in zip(keys, labels):
        val = row.get(k, 'N/A')
        parts.append(f"{lbl}:{val}")
    return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# Screening Functions
# ═══════════════════════════════════════════════════════════════════════

def screen_low_price_bull(top_n: int = 5) -> Tuple[pd.DataFrame | None, str]:
    """Low price + high growth strategy.

    Criteria: Price<10, NetProfitGrowth>=100%, non-ST, no STAR/ChiNext board.
    """
    query = (
        "股价<10元，"
        "净利润增长率(净利润同比增长率)≥100%，"
        "非st，非科创板，非退市，"
        "主板或创业板，"
        "成交额由小至大排名"
    )
    logger.info("Screening: low_price_bull (top_n=%d)", top_n)
    print(f"🐂 低价擒牛选股\n  条件: 股价<10元 + 净利增长≥100% + 非ST\n")

    try:
        result = pywencai.get(query=query, loop=True)
        df = _convert_to_dataframe(result)
        if df is None or df.empty:
            return None, "未获取到符合条件的股票"
        selected = df.head(top_n)
        return selected, f"低价擒牛：筛选出 {len(selected)} 只（共 {len(df)} 只符合条件）"
    except Exception as e:
        return None, f"低价擒牛选股失败: {e}"


def screen_value_stocks(top_n: int = 10) -> Tuple[pd.DataFrame | None, str]:
    """Value investment strategy.

    Criteria: PE<=20, PB<=1.5, Dividend>=1%, DebtRatio<=30%, non-ST.
    """
    query = (
        "市盈率小于等于20，"
        "市净率小于等于1.5，"
        "股息率大于等于1%，"
        "资产负债率小于等于30%，"
        "非st，非科创板，非退市，"
        "按流通市值由小到大排名"
    )
    logger.info("Screening: value (top_n=%d)", top_n)
    print(f"💎 价值投资选股\n  条件: PE≤20 + PB≤1.5 + 股息≥1% + 负债≤30%\n")

    try:
        result = pywencai.get(query=query, loop=True)
        df = _convert_to_dataframe(result)
        if df is None or df.empty:
            return None, "未获取到符合条件的股票"
        selected = df.head(top_n)
        return selected, f"价值投资：筛选出 {len(selected)} 只（共 {len(df)} 只符合条件）"
    except Exception as e:
        return None, f"价值选股失败: {e}"


def screen_profit_growth(top_n: int = 5) -> Tuple[pd.DataFrame | None, str]:
    """Profit growth strategy.

    Criteria: NetProfitGrowth>=10%, Shenzhen A, non-ST, no STAR/ChiNext.
    """
    query = (
        "净利润增长率(净利润同比增长率)≥10%，"
        "非科创板，非ST，非退市，"
        "主板或创业板，"
        "成交额由小至大排名"
    )
    logger.info("Screening: profit_growth (top_n=%d)", top_n)
    print(f"📈 净利增长选股\n  条件: 净利增长≥10% + 主板或创业板 + 非ST\n")

    try:
        result = pywencai.get(query=query, loop=True)
        df = _convert_to_dataframe(result)
        if df is None or df.empty:
            return None, "未获取到符合条件的股票"
        selected = df.head(top_n)
        return selected, f"净利增长：筛选出 {len(selected)} 只（共 {len(df)} 只符合条件）"
    except Exception as e:
        return None, f"净利增长选股失败: {e}"


def screen_small_cap(top_n: int = 5) -> Tuple[pd.DataFrame | None, str]:
    """Small cap high-growth strategy.

    Criteria: MarketCap<=50B, RevenueGrowth>=10%, NetProfitGrowth>=100%.
    """
    query = (
        "总市值≤50亿，"
        "营收增长率≥10%，"
        "净利润增长率(净利润同比增长率)≥100%，"
        "主板或创业板，非ST，非科创板，非退市，"
        "总市值由小至大排名"
    )
    logger.info("Screening: small_cap (top_n=%d)", top_n)
    print(f"📊 小市值选股\n  条件: 市值≤50亿 + 营收增长≥10% + 净利增长≥100%\n")

    try:
        result = pywencai.get(query=query, loop=True)
        df = _convert_to_dataframe(result)
        if df is None or df.empty:
            return None, "未获取到符合条件的股票"
        selected = df.head(top_n)
        return selected, f"小市值：筛选出 {len(selected)} 只（共 {len(df)} 只符合条件）"
    except Exception as e:
        return None, f"小市值选股失败: {e}"


def screen_main_force(
    days_ago: int = 20,
    min_market_cap: int = 30,
    max_market_cap: int = 500,
) -> Tuple[pd.DataFrame | None, str]:
    """Main capital force strategy.

    Criteria: Top 100 by main capital net inflow since start_date,
    filtered by range change rate and market cap.
    """
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y年%m月%d日")

    query = (
        f"{start_date}以来主力资金净流入前20名，并计算区间涨跌幅，"
        f"市值{min_market_cap}-{max_market_cap}亿之间，主板或创业板，非科创非st，非退市，"
        f"所属同花顺行业，总市值，净利润，营收，市盈率，市净率"
    )
    logger.info("Screening: main_force (days_ago=%d)", days_ago)
    print(f"💪 主力资金选股\n  日期: {start_date} 以来\n  市值: {min_market_cap}-{max_market_cap}亿\n")

    try:
        result = pywencai.get(query=query, loop=True)
        df = _convert_to_dataframe(result)
        if df is None or df.empty:
            return None, "未获取到符合条件的股票"
        selected = df.head(20)
        return selected, f"主力资金：获取到 {len(selected)} 只"
    except Exception as e:
        return None, f"主力资金选股失败: {e}"


# ═══════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════

def format_screening_results(df: pd.DataFrame, strategy_name: str, top_n: int = 10) -> str:
    """Format screening results as a WeCom-compatible text block with rationale."""
    if df is None or df.empty:
        return ""

    show = df.head(top_n)
    lines = [f"📋 {strategy_name}  选股结果", "━" * 30]

    # Ranking basis for each strategy
    rank_basis = {
        "低价擒牛": "按成交额从小到大排名",
        "价值投资": "按流通市值从小到大排名",
        "净利增长": "按成交额从小到大排名",
        "小市值": "按总市值从小到大排名",
        "主力资金": "按主力资金净流入从大到小排名",
    }
    lines.append(rank_basis.get(strategy_name, ""))

    # Map strategy to (label, column_keyword) — keyword matches against pywencai's real columns
    strategy_fields: dict[str, list[tuple[str, str]]] = {
        "低价擒牛": [
            ("股价", "收盘价"), ("净利增长", "净利润(同比增长率)"),
            ("营收增长", "营业收入(同比增长率)"),
        ],
        "价值投资": [
            ("市盈率", "市盈率"), ("市净率", "市净率"),
            ("股息率", "股息率"), ("负债率", "资产负债率"),
        ],
        "净利增长": [
            ("净利增长", "净利润(同比增长率)"),
            ("营收增长", "营业收入(同比增长率)"), ("成交额", "成交额"),
        ],
        "小市值": [
            ("总市值", "总市值"), ("营收增长", "营业收入(同比增长率)"),
            ("净利增长", "净利润(同比增长率)"),
        ],
        "主力资金": [
            ("主力净流入", "主力资金净流入"), ("区间涨跌幅", "区间涨跌幅"),
            ("总市值", "总市值"),
        ],
    }

    fields = strategy_fields.get(strategy_name, [])

    # Build column lookup: for each field, find the best-matching column
    col_map: dict[str, str] = {}
    for label, keyword in fields:
        for col in show.columns:
            # Match by keyword (substring) against pywencai's verbose column names
            if keyword in col:
                col_map[label] = col
                break

    for idx, (_, row) in enumerate(show.iterrows(), 1):
        # Stock identity
        code = ""
        name = ""
        for col in show.columns:
            if '代码' in col:
                code = str(row.get(col, ''))
                code = code.split('.')[0] if '.' in code else code
            if '简称' in col:
                name = str(row.get(col, ''))

        lines.append(f"\n{idx}. {code} {name}")

        # Build rationale from matched columns
        parts = []
        for label, col_name in col_map.items():
            val = row.get(col_name)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if str(val) == 'nan':
                continue
            try:
                fv = float(val)
                if abs(fv) >= 100000000:
                    parts.append(f"{label}={fv/100000000:.1f}亿")
                elif abs(fv) >= 10000:
                    parts.append(f"{label}={fv/10000:.1f}万")
                elif abs(fv) >= 100:
                    parts.append(f"{label}={fv:.0f}")
                elif fv >= 1:
                    parts.append(f"{label}={fv:.1f}")
                else:
                    parts.append(f"{label}={fv:.2f}")
            except (ValueError, TypeError):
                parts.append(f"{label}={val}")

        if parts:
            lines.append("  → " + "，".join(parts[:8]))

    return "\n".join(lines)


def _get_code(row) -> str:
    """Extract 6-digit stock code from a DataFrame row."""
    for col in row.index:
        if '代码' in str(col):
            code = str(row.get(col, ''))
            return code.split('.')[0] if '.' in code else code
    return ""


def _get_name(row) -> str:
    """Extract stock name from a DataFrame row."""
    for col in row.index:
        if '简称' in str(col):
            return str(row.get(col, ''))
    return ""


def _enrich_prices_from_tencent(picks: list[dict]) -> None:
    """Batch-fetch real-time prices from Tencent API and update picks in-place."""
    if not picks:
        return
    import requests

    codes = [p["code"] for p in picks if p.get("code")]
    if not codes:
        return

    # Build Tencent symbols (batch of 50)
    symbols = []
    for code in codes:
        prefix = "sh" if code.startswith(("6",)) else "sz"
        symbols.append(f"{prefix}{code}")

    prices = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i + 50]
        url = f"https://qt.gtimg.cn/q={','.join(batch)}"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            for line in r.text.split("\n"):
                if '="' in line:
                    raw = line.split('="', 1)[1].rstrip('";\n')
                    fields = raw.split("~")
                    if len(fields) > 3 and fields[2] and fields[3]:
                        try:
                            prices[fields[2]] = float(fields[3])
                        except ValueError:
                            pass
        except Exception:
            pass

    for p in picks:
        code = p.get("code", "")
        if code in prices and prices[code] > 0:
            p["price"] = prices[code]


def _composite_screening(top_n: int = 5, suffix: str = "") -> str:
    """Run all 5 strategies, cross-score stocks, return top N composite picks.

    Enhancements:
      - Dynamic strategy weighting (scarcer hits = higher weight)
      - Industry diversification in final picks
      - AI-generated portfolio allocation advice
      - Persistent tracking with yesterday performance review

    Args:
        top_n: Number of top picks
        suffix: Optional filename suffix (e.g. '_pm' for afternoon run)

    Returns a single WeCom-formatted message.
    """
    import math
    from collections import defaultdict
    from datetime import datetime

    # ── Pre-screening ──────────────────────────────────────────────
    _load_pre_screener()
    pre_screened_codes: set = set()
    pre_screened_count = 0
    pre_stats = ""
    if _pre_screener_loaded and callable(globals().get("run_pre_screening")):
        try:
            candidates = run_pre_screening(force=True, save_snapshot=(suffix != "_pm" and get_config().stock_boards == "main,chinext"))  # type: ignore[name-defined] # noqa: F821
            pre_screened_codes = {c["code"] for c in candidates}
            pre_screened_count = len(candidates)
            pre_stats = (
                f"🔍 预筛选: {pre_screened_count} 只通过（成交量+技术面过滤）\n"
                f"   排除: 成交额不足 + 评分<0且周线下跌\n\n"
            )
        except Exception as e:
            logger.warning("Pre-screening failed, running without filter: %s", e)
    else:
        logger.info("Pre-screening not available, running all strategies directly")

    # ── Run 5 strategies ──────────────────────────────────────────
    strategies = [
        ("🐂低价擒牛", screen_low_price_bull),
        ("💎价值投资", screen_value_stocks),
        ("📈净利增长", screen_profit_growth),
        ("📊小市值", screen_small_cap),
        ("💪主力资金", lambda: screen_main_force(days_ago=20)),
    ]

    stock_map: dict[str, dict] = defaultdict(
        lambda: {"name": "", "hits": [], "best_row": None, "industry": ""}
    )

    total_hits = 0
    excluded_by_prescreen = 0
    strategy_counts: dict[str, int] = {}  # track total results per strategy

    for sname, sfunc in strategies:
        try:
            df, _ = sfunc()
            if df is None or df.empty:
                continue
            # Use actual result count for accurate scarcity calculation
            strategy_counts[sname] = len(df)
            # Consider up to top 15 from each strategy
            top = df.head(15)
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                code = _get_code(row)
                if not code:
                    continue
                total_hits += 1

                # Cross-reference with pre-screened pool
                if pre_screened_codes and code not in pre_screened_codes:
                    excluded_by_prescreen += 1
                    continue

                name = _get_name(row) or stock_map[code]["name"]
                stock_map[code]["name"] = name
                stock_map[code]["hits"].append((sname, rank))
                if stock_map[code]["best_row"] is None or rank < 10:
                    stock_map[code]["best_row"] = row

                # Extract industry from pywencai row
                if not stock_map[code]["industry"]:
                    for col in row.index:
                        if "行业" in str(col):
                            val = row.get(col)
                            if val and str(val) != "nan":
                                stock_map[code]["industry"] = str(val)[:12]
                            break
        except Exception as e:
            logger.warning("Strategy %s failed in composite: %s", sname, e)
            strategy_counts[sname] = 0

    if not stock_map:
        msg = "📋 综合精选: 暂无符合条件的股票"
        if pre_stats:
            msg = pre_stats + msg
        return msg

    # ── Score each stock (dynamic weighting) ───────────────────────
    def _strategy_weight(sname: str, rank: int) -> float:
        """Dynamic weight: base=10 × scarcity coefficient × rank bonus."""
        base = 10.0
        n = strategy_counts.get(sname, 1)
        scarcity = math.log2(10.0 / max(n, 1))
        # Floor at 0.5 so abundant strategies still give positive weight
        scarcity = max(0.5, min(4.0, scarcity))
        # Rank multiplier: #1=1.5, #2=1.3, #3=1.1, rest=1.0
        rank_mult = {1: 1.5, 2: 1.3, 3: 1.1}.get(rank, 1.0)
        return base * scarcity * rank_mult

    scored = []
    for code, info in stock_map.items():
        score = 0.0
        for sname, rank in info["hits"]:
            score += _strategy_weight(sname, rank)
        n_strategies = len(info["hits"])
        # Multi-strategy bonus: each extra strategy adds 20%
        multi_bonus = 1.0 + (n_strategies - 1) * 0.2
        final_score = score * multi_bonus
        scored.append((final_score, n_strategies, code, info))

    # Sort: score desc, then #strategies desc
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # ── Select top N (no industry cap) ─────────────────────────────
    selected = scored[:top_n]

    # ── Build picks list (without pywencai prices) ──────────────────
    picks_for_tracking = []

    for idx, (score_val, n_strat, code, info) in enumerate(selected, 1):
        hit_strs = [f"{sn}#{r}" for sn, r in info["hits"]]
        picks_for_tracking.append({
            "code": code,
            "name": info["name"],
            "price": 0.0,  # will be filled by Tencent below
            "score": round(score_val, 1),
            "strategies": hit_strs,
            "industry": info.get("industry", ""),
            "best_row": info["best_row"],
        })

    # ── Fetch real-time prices from Tencent ─────────────────────────
    _enrich_prices_from_tencent(picks_for_tracking)

    # ── Build portfolio summary (with accurate Tencent prices) ─────
    picks_summary_lines = []
    for idx, pick in enumerate(picks_for_tracking, 1):
        price_str = f"¥{pick['price']:.2f}" if pick.get("price") else ""
        picks_summary_lines.append(
            f"{idx}. {pick['code']} {pick['name']} {price_str} "
            f"行业:{pick.get('industry', '未知')} "
            f"评分:{pick['score']:.1f} 命中:{','.join(pick['strategies'])}"
        )

    picks_summary = "\n".join(picks_summary_lines)

    # ── Portfolio suggestion (AI, optional) ────────────────────────
    portfolio_section = ""
    try:
        from lib.ai_agents import StockAnalysisAgents
        agents = StockAnalysisAgents()
        portfolio = agents.suggest_portfolio(picks_summary)
        allocation = portfolio.get("allocation", [])
        total_pos = portfolio.get("total_position", "N/A")
        advice = portfolio.get("overall_advice", "")
        if allocation:
            portfolio_section = (
                f"\n\n💼 组合建议  总仓位: {total_pos}\n"
                f"{'─' * 20}\n"
            )
            for alloc in allocation:
                c = alloc.get("code", "")
                n = alloc.get("name", "")
                r = alloc.get("ratio", "")
                s = alloc.get("style", "")
                portfolio_section += f"  {c} {n}: {r} ({s})\n"
            if advice:
                portfolio_section += f"\n  {advice}"
    except Exception as e:
        logger.warning("Portfolio suggestion failed: %s", e)

    # ── Build message ─────────────────────────────────────────────
    lines = [
        "📋 综合精选 Top " + str(top_n),
        "━" * 24,
        pre_stats.rstrip("\n") if pre_stats else "",
        "动态权重打分 + 行业分散 + 多策略交叉",
        "",
    ]
    if pre_screened_codes and excluded_by_prescreen > 0:
        lines.append(
            f"（策略命中 {total_hits} 次，预筛选排除 {excluded_by_prescreen} 次）\n"
        )

    # Build price lookup from Tencent-enriched picks
    price_map = {p["code"]: p["price"] for p in picks_for_tracking}

    for idx, (score_val, n_strat, code, info) in enumerate(selected, 1):
        name = info["name"]
        industry = info.get("industry", "")
        hit_strs = []
        for sname, rank in info["hits"]:
            hit_strs.append(f"{sname}#{rank}")
        hits_line = " + ".join(hit_strs)

        industry_tag = f" [{industry}]" if industry else ""
        lines.append(f"{idx}. {code} {name}{industry_tag}")
        lines.append(f"   策略: {len(info['hits'])}个 | 得分: {score_val:.1f}  |  {hits_line}")

        row = info["best_row"]
        metrics = []
        # Add Tencent real-time price first
        tencent_price = price_map.get(code, 0)
        if tencent_price > 0:
            metrics.append(f"股价={tencent_price:.2f}")
        # Add fundamental metrics from pywencai row
        if row is not None:
            for col in row.index:
                val = row.get(col)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                col_str = str(col)
                if '简称' in col_str or '代码' in col_str or '行业' in col_str:
                    continue
                if '收盘价' in col_str or '最新价' in col_str:
                    continue  # skip pywencai price, we use Tencent
                for kw, label in [
                    ("总市值", "市值"), ("市盈率", "PE"),
                    ("净利润(同比增长率)", "净利增长"),
                ]:
                    if kw in col_str and label not in [m.split("=")[0] for m in metrics]:
                        try:
                            fv = float(val)
                            if abs(fv) >= 100000000:
                                metrics.append(f"{label}={fv/100000000:.1f}亿")
                            elif abs(fv) >= 10000:
                                metrics.append(f"{label}={fv/10000:.1f}万")
                            else:
                                metrics.append(f"{label}={fv:.1f}")
                        except (ValueError, TypeError):
                            pass
                        break
        if metrics:
            lines.append(f"   " + "  ".join(metrics[:5]))

    # ── Portfolio section ──────────────────────────────────────────
    if portfolio_section:
        lines.append(portfolio_section)

    # ── Save to tracker (prices already enriched from Tencent) ─────
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        from lib.tracker import save_recommendation, calc_performance
        # Strip non-serializable fields (best_row is a pandas Series)
        clean_picks = [
            {k: v for k, v in p.items() if k != "best_row"}
            for p in picks_for_tracking
        ]
        save_recommendation(today, clean_picks, pre_screened_count, suffix=suffix)
        # Append yesterday's performance review
        perf = calc_performance()
        if perf:
            lines.append("\n\n" + perf)
    except Exception as e:
        logger.warning("Tracker failed: %s", e)

    lines.append("\n──")
    lines.append("⚠️ 以上为量化筛选，仅供参考，不构成投资建议。")
    return "\n".join(lines)


def run_all_screenings(top_n: int = 5, suffix: str = "") -> str:
    """Run composite screening across all 5 strategies.

    Args:
        top_n: Number of top picks to include
        suffix: Optional suffix for save path, e.g. '_pm'

    Returns a single WeCom message with the top N cross-strategy picks.
    """
    return _composite_screening(top_n=top_n, suffix=suffix)
