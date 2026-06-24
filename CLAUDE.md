# CLAUDE.md — Stock Reporter

A-share (A股) stock analysis and WeCom reporting system deployed on cloud server.

## Project Overview

Cloud-deployed A-share market monitoring system that:
- Fetches real-time quotes and K-line data from Tencent Finance (`qt.gtimg.cn`)
- Computes technical indicators (MA, MACD, RSI, Bollinger, KDJ, ADX, divergence)
- Generates trading signals with composite scoring
- Pushes reports to WeCom (企业微信) via webhook bot
- Optional DeepSeek AI analysis and multi-agent enhanced analysis
- Supports daemon mode for scheduled reporting during market hours (Mon–Fri, 9:30–15:00)

## Architecture

```
stock-reporter/
├── cloud_stock_reporter.py   # ** Main entry ** — single-stock/multi-stock reporter
├── afternoon_movers.py       # Afternoon closing scan: top gainers by board
├── stock_screener.py         # CLI for pywencai-based screening strategies
├── macro_reporter.py         # China macro economic analysis (GDP/CPI/PMI/M2)

├── .env                      # All configuration (webhook, API keys, flags)
├── lib/
│   ├── config.py             # Typed Config dataclass, reads from .env
│   ├── ai_agents.py          # Multi-agent analysis: technical/fundamental/fund-flow
│   ├── enhanced_data.py      # Quarterly reports + fund flow data (daily cached)
│   ├── selectors.py          # 5 screening strategies (via pywencai/问财)
│   ├── report_html.py        # HTML report generation + merged report builder
│   ├── pre_screener.py       # Pre-screening stock pool (turnover filter)
│   ├── tracker.py            # Recommendation tracking + persistence
│   ├── macro_data.py         # Macro data fetching (GDP/CPI/PMI/M2/interest rates)
│   ├── macro_agents.py       # Macro analysis AI agents
│   └── deepseek_client.py    # DeepSeek API client with retry/fallback
├── data/                     # Daily pre-screened pools (sharded by board) + turnover snapshots
│   └── recommendations/      # Saved stock picks by date
├── reports/                  # Generated HTML reports, organized by date
└── aiagents-stock/           # **Sub-project**: Streamlit UI for stock analysis
    ├── app.py                # Main Streamlit app (114KB)
    ├── smart_monitor_*.py    # Smart monitoring system (kline, QMT, deepseek, engine)
    ├── longhubang_*.py       # 龙虎榜 (Dragon-Tiger Board) analysis
    ├── news_flow_*.py        # News flow analysis system
    ├── sector_strategy_*.py  # Sector rotation strategy
    ├── portfolio_*.py        # Portfolio management
    └── macro_*.py            # Macro cycle analysis
```

## Key Files & Roles

| File | Role |
|------|------|
| `cloud_stock_reporter.py` | Main reporter — fetches quotes (Tencent), computes indicators, scores stocks, pushes WeCom. Supports `--daemon` mode and single-run mode. |
| `afternoon_movers.py` | PM closing scan — top 20 gainers for 主板 & 创业板, enriches with technical indicators, cross-references morning picks, pushes to WeCom. Uses ThreadPoolExecutor. |
| `stock_screener.py` | Screening CLI — 5 strategies: `low_price_bull`, `value`, `profit_growth`, `small_cap`, `main_force`. Composite mode runs all + AI portfolio. |
| `macro_reporter.py` | Macro economic analysis — fetches China macro data, runs DeepSeek analysis, pushes to WeCom. |


## Data Flow

1. **Stock list** from `.env` `STOCK_CODES` (or env var)
2. **Real-time quotes** via Tencent `qt.gtimg.cn/q=` API (3s refresh, works from cloud)
3. **K-line data** via Tencent `web.ifzq.gtimg.cn/appstock/app/fqkline/get` (daily/weekly, 90 bars)
4. **Technical indicators** computed locally (pure Python with numpy/pandas)
5. **Composite scoring** (10 dimensions: MA, MACD, RSI, Bollinger, Volume, KDJ, ADX, Divergence, Weekly trend, MACD crossover)
6. **AI analysis** — optional DeepSeek v4-pro single call + optional multi-agent enhanced analysis (technical/fundamental/fund-flow → team discussion → final decision)
7. **HTML report** generated via `lib/report_html.py` → uploaded to WeCom as file, falls back to text push
8. **Screening** — pywencai-based strategies, results saved to `data/recommendations/`

## Configuration (.env)

Key variables (all loaded by `lib/config.py`):
- `WECOM_WEBHOOK_URL` — WeCom bot webhook (required)
- `STOCK_CODES` — comma-separated A-share codes
- `DEEPSEEK_API_KEY` — DeepSeek API key (optional, skips AI if unset)
- `DEEPSEEK_MODEL` — default `deepseek-v4-pro`
- `REPORT_INTERVAL_MINUTES` — daemon cycle interval (default 60)
- `ENABLE_ENHANCED_ANALYSIS` — multi-agent analysis gate (true/false)
- `SCREENING_ENABLED` — run daily screening on daemon startup
- `STOCK_BOARDS` — boards to include in screening: `main`(主板), `chinext`(创业板), `star`(科创板), `bse`(北交所). Comma-separated, default `main,chinext`. Override via env var at runtime for per-run control.
- `EXCLUDE_ST` — exclude ST/*ST stocks from screening (default `true`)
- `MIN_TURNOVER` — pre-screening minimum daily turnover (default 1亿)

### Board Filter Architecture

```
STOCK_BOARDS env var (or .env default)
    ↓
Config.allowed_prefixes  →  whitelist filter in fetch_stock_pool()
Config.boards_slug       →  cache sharding: pre_screened_{date}_{boards}.json
```

Different board configs produce separate caches — safe to run multiple times/day with different settings.
Example: `STOCK_BOARDS=main python stock_screener.py --composite --send`

## Python Environment

- Virtual env: `/home/.venv/bin/python3`
- Package manager: `pip3` (points to `/home/.venv/bin/pip3`)
- Key deps: `requests`, `pandas`, `numpy`, `openai` (for DeepSeek), `akshare`, `pywencai`

## Commands

```bash
# Single report run (all stocks)
/home/.venv/bin/python3 cloud_stock_reporter.py

# Daemon mode (scheduled during market hours)
/home/.venv/bin/python3 cloud_stock_reporter.py --daemon

# Afternoon scan (print only)
/home/.venv/bin/python3 afternoon_movers.py --print

# Afternoon scan + push to WeCom
/home/.venv/bin/python3 afternoon_movers.py

# Stock screening (single strategy)
/home/.venv/bin/python3 stock_screener.py --strategy value --top 10

# Composite screening + push to WeCom
/home/.venv/bin/python3 stock_screener.py --composite --send

# Macro report
/home/.venv/bin/python3 macro_reporter.py --send

# Analyze specific stocks


```

## Cron Setup

Daily schedule (Mon–Fri only), staggered to avoid API/resource contention:

| Time | Task | Boards | Output |
|------|------|--------|--------|
| 09:30 | 早间主板选股 | `main` | `_am` suffix，纯文本 |
| 09:35 | 早间综合选股 | `main,chinext` | 纯文本 |
| 09:45 | Daemon 启动 | — | cycle1 ~09:46, cycle2 ~14:46，HTML 文件 |
| 14:25 | 午后主板选股 | `main` | `_pm_main` suffix，纯文本 |
| 14:30 | 午后综合选股 | `main,chinext` | `_pm` suffix，纯文本 |
| 15:00 | 收盘涨幅扫描 | pre-screened cache | text × 2~4 |
| 15:15 | 清理 daemon | — | — |

Timeline (no overlapping API-heavy tasks):
```
09:30 ═ composite _am ═════ (3-4min, pywencai+Tencent)
09:35 ═ composite ═════════ (3-4min, pywencai+Tencent)    ── 早间选股结束
09:45 ═ daemon start ══════ (7-14min, Tencent+DeepSeek)   ── 无 pywencai 竞争
14:25 ═ composite _pm_main ═ (3-4min, pywencai+Tencent)   ── daemon 已停
14:30 ═ composite _pm ═════ (3-4min, pywencai+Tencent)    ── 午后选股结束
14:46 ═ daemon cycle2 ═════ (7-14min, Tencent+DeepSeek)   ── 无 pywencai 竞争
15:00 ═ afternoon scan ════ (Tencent only, 无 pywencai/DeepSeek)
```

Key design decisions:
- All pywencai-heavy tasks (3 composites) never overlap with each other or with daemon
- `afternoon_movers` uses only Tencent + cached data — safe to co-run with daemon tail
- `--suffix _am` / `--pm` prevent recommendation files from being overwritten
- Different boards → different pre_screened caches → no file contention
- Scheduling via system crontab (`crontab -l` to view)

## Tencent API Details

- **Quote**: `https://qt.gtimg.cn/q=sh600519,sz000858` — returns semicolon-delimited lines with `~`-separated fields
- **K-line**: `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,,,90,qfq`
  - Keys: `qfqday`, `qfqweek`, `qfqmonth`
  - Returns 6 columns: `[date, open, close, high, low, volume]`
  - Some rows may have extra columns (dividend/rights-issue metadata)
- Stock code prefix: `6xxxxx` → `sh`, else → `sz`

## Technical Indicator Scoring

Score range: roughly -10 to +10. Each dimension contributes ±1 (divergence ±2):
- **Bullish**: price > MA20, MA5 > MA10, MACD golden cross, RSI < 30, near BB lower, volume breakout up, KDJ < 0, ADX > 40 with +DI > -DI, bottom divergence, weekly up
- **Bearish**: opposite conditions
- **Divergence has double weight** — 顶背离 -2, 底背离 +2
- Signal text thresholds: ≥5 strong bullish, ≥3 mild bullish, ≤-5 strong bearish, ≤-3 mild bearish

## Multi-Agent Enhanced Analysis

When `ENABLE_ENHANCED_ANALYSIS=true`:
1. Three analyst agents: technical, fundamental, fund-flow (each calls DeepSeek)
2. Team discussion phase (agents debate findings)
3. Final decision: rating, target price, stop loss, take profit, position size, confidence (1-10)

## aiagents-stock Sub-project (Legacy/Reference)

A comprehensive Streamlit-based web UI for stock analysis with:
- Smart monitoring (K-line + DeepSeek analysis)
- Dragon-Tiger Board (龙虎榜) analysis with scoring
- News flow analysis with sentiment
- Sector rotation strategy
- Portfolio management
- Macro cycle analysis
- Multiple SQLite databases (~30MB total)

This sub-project is **not actively used** in the current cloud reporter pipeline but contains valuable reference code for DeepSeek integration patterns and analysis strategies.

## Important Notes

- Sina Finance APIs are **blocked from datacenter IPs** — Tencent APIs used as fallback that works from cloud
- WeCom messages have ~4000 char limit — reports split into multiple messages or uploaded as HTML files
- Pre-screened stock pools are cached daily in `data/pre_screened_YYYY-MM-DD.json`
- Turnover snapshots saved daily for next-morning pre-screening reference
- Daemon automatically exits on weekends and after 15:00 market close
- All AI analysis includes disclaimer: "以上为AI分析，仅供参考，不构成投资建议"
