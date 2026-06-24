#!/usr/bin/env python3
"""Standalone stock reporter for cloud server deployment.

Fetches A-share stock data from Sina Finance, computes technical indicators,
generates trading signals, and sends detailed reports to WeCom (企业微信).

Usage:
    python cloud_stock_reporter.py

Requires:
    pip install requests pandas numpy

Configuration via environment variables:
    WECOM_WEBHOOK_URL  - WeCom bot webhook URL (required)
    STOCK_CODES        - Comma-separated stock codes (default: from config.json or hardcoded)
"""

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

# Ensure lib/ is importable even when run from cron (cwd may differ)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stock_reporter")

# ── Multi-agent enhancement (optional) ────────────────────────────────
try:
    from lib.config import get_config
    from lib.ai_agents import StockAnalysisAgents
    from lib.enhanced_data import EnhancedDataManager

    _cfg = get_config()
    _ENHANCED_ENABLED = bool(_cfg.enable_enhanced_analysis and _cfg.deepseek_api_key)
    if _ENHANCED_ENABLED:
        _enhanced_data_mgr = EnhancedDataManager()
        logger.info("Enhanced multi-agent analysis: ENABLED")
    else:
        logger.info("Enhanced multi-agent analysis: disabled")
except ImportError:
    _ENHANCED_ENABLED = False
    logger.info("Enhanced multi-agent analysis: lib modules not available")

# ── Load .env file ───────────────────────────────────────────────────

def _load_env() -> dict:
    """Load key=value pairs from the .env file next to this script."""
    env = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip()
        except Exception:
            pass
    return env


_ENV = _load_env()


def _env_or(key: str, default: str = "") -> str:
    """Read config from environment variable, falling back to .env, then default."""
    return os.environ.get(key) or _ENV.get(key) or default


# ── Config ───────────────────────────────────────────────────────────
WEBHOOK_URL = _env_or("WECOM_WEBHOOK_URL", "")

STOCK_CODES_RAW = _env_or("STOCK_CODES", "600519,000858")
STOCK_CODES = [c.strip() for c in STOCK_CODES_RAW.split(",") if c.strip()]

DEEPSEEK_API_KEY = _env_or("DEEPSEEK_API_KEY", "")

REPORT_INTERVAL_MINUTES = int(_env_or("REPORT_INTERVAL_MINUTES", "60"))

# Tencent API settings (works from cloud servers; Sina blocks datacenter IPs)
TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q={symbol}"
TENCENT_KLINE_URL = (
    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
)

# HTTP session
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
})

# ── Helpers ──────────────────────────────────────────────────────────

_SH_PREFIXES = ("6",)
_SZ_PREFIXES = ("0", "3")


def _http_get_with_retry(url, *, max_retries=3, **kwargs):
    """HTTP GET with retry on transient failures."""
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, timeout=10, **kwargs)
            if r.status_code == 200:
                return r
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            logger.debug("Request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            time.sleep(1 + attempt)
    return None


def code_prefix(code: str) -> str:
    """Return 'sh' or 'sz' for a 6-digit A-share code."""
    c = code.strip()
    return "sh" if c.startswith(_SH_PREFIXES) else "sz"


def _last(series: pd.Series) -> Optional[float]:
    """Return last non-NaN value of a series."""
    v = series.dropna()
    if v.empty:
        return None
    return float(v.iloc[-1])


# ── Data Fetching ────────────────────────────────────────────────────

def fetch_quote(symbol: str):
    """Fetch real-time quote from Tencent Finance.
    Returns (name, price, change_pct, change_amt).
    """
    try:
        r = _http_get_with_retry(
            TENCENT_QUOTE_URL.format(symbol=symbol),
        )
        if r and r.text:
            text = r.text
            # Tencent format: v_sh600519="1~name~code~price~prev_close~..."
            if '="' in text:
                raw = text.split('="', 1)[1].rstrip('";\n')
                fields = raw.split("~")
                if len(fields) > 4:
                    name = fields[1]
                    price = float(fields[3]) if fields[3] else 0.0
                    prev_close = float(fields[4]) if fields[4] else price
                    change_pct = 0.0
                    change_amt = 0.0
                    if prev_close > 0:
                        change_pct = (price - prev_close) / prev_close * 100.0
                        change_amt = price - prev_close
                    return name, price, change_pct, change_amt
    except Exception as e:
        logger.warning("Quote fetch failed for %s: %s", symbol, e)
    return "", 0.0, 0.0, 0.0


def fetch_kline(symbol: str, scale: str = "day", datalen: int = 90) -> Optional[pd.DataFrame]:
    """Fetch K-line data from Tencent and return as DataFrame."""
    try:
        r = _http_get_with_retry(
            TENCENT_KLINE_URL,
            params={"param": f"{symbol},{scale},,,{datalen},qfq"},
        )
        if not r or not r.text:
            return None
        raw = r.json()
        data = raw.get("data", {}).get(symbol, {})
        # Tencent uses "qfqday" / "qfqweek" / "qfqmonth" keys
        key = f"qfq{scale}"
        klines = data.get(key)
        if not klines or len(klines) < 30:
            return None
        # Tencent format: [date, open, close, high, low, volume, (optional ex-rights info)]
        # Some rows have a 7th column with dividend/rights-issue metadata
        klines = [row[:6] for row in klines]
        df = pd.DataFrame(klines, columns=["date", "open", "close", "high", "low", "volume"])
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.warning("K-line fetch failed for %s: %s", symbol, e)
        return None


# ── Technical Indicators ─────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame):
    """Compute all technical indicators from daily K-line data.
    Returns a dict of indicator values.
    """
    close = df["close"].dropna()
    high = df["high"].dropna()
    low = df["low"].dropna()
    volume = df["volume"].dropna()

    if len(close) < 26:
        return None

    # -- MA --
    ma5_s = close.rolling(5).mean()
    ma10_s = close.rolling(10).mean()
    ma20_s = close.rolling(20).mean()

    # -- MACD --
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_s = ema12 - ema26
    sig_s = macd_s.ewm(span=9, adjust=False).mean()

    # -- RSI (14) --
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_s = 100.0 - (100.0 / (1.0 + rs))

    # -- Bollinger Bands (20, 2) --
    bb_sigma = close.rolling(20).std()
    bb_upper_s = ma20_s + 2 * bb_sigma
    bb_lower_s = ma20_s - 2 * bb_sigma

    # -- KDJ (9, 3, 3) --
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100.0
    k_vals = [50.0] * 8
    d_vals = [50.0] * 8
    for i in range(8, len(rsv)):
        r = rsv.iloc[i]
        if pd.isna(r):
            k_vals.append(k_vals[-1])
            d_vals.append(d_vals[-1])
        else:
            k_vals.append(2/3 * k_vals[-1] + 1/3 * r)
            d_vals.append(2/3 * d_vals[-1] + 1/3 * k_vals[-1])
    j_vals = [3*k - 2*d for k, d in zip(k_vals, d_vals)]

    # -- Volume ratio --
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio_s = volume / vol_avg20.replace(0, np.nan)

    # -- Support / Resistance (60-day) --
    support_val = float(low.tail(60).min()) if len(low) >= 20 else float(low.min())
    resistance_val = float(high.tail(60).max()) if len(high) >= 20 else float(high.max())

    # -- ADX (14) --
    adx_val, plus_di, minus_di = compute_adx(high, low, close)

    # -- Divergence --
    div_text = detect_divergence(close, rsi_s, macd_s)

    # -- MACD crossover (detect actual DIF/DEA cross within last 3 bars) --
    macd_hist_series = macd_s - sig_s
    macd_cross = ""
    if len(macd_hist_series) >= 5:
        h = macd_hist_series.iloc[-5:].values
        for i in range(1, min(4, len(h))):
            if h[i - 1] <= 0 < h[i]:
                macd_cross = "金叉"
                break
            elif h[i - 1] >= 0 > h[i]:
                macd_cross = "死叉"
                break

    # -- Volume trend --
    vr = _last(vol_ratio_s) or 1.0
    vol_trend = "放量" if vr > 1.2 else ("缩量" if vr < 0.8 else "正常")

    # -- Bollinger width --
    bb_m = _last(ma20_s)
    bb_u = _last(bb_upper_s)
    bb_l = _last(bb_lower_s)
    bb_w = ((bb_u - bb_l) / bb_m * 100.0) if bb_m and bb_u and bb_l else None

    return {
        "ma5": _last(ma5_s),
        "ma10": _last(ma10_s),
        "ma20": _last(ma20_s),
        "macd": _last(macd_s),
        "macd_signal": _last(sig_s),
        "macd_hist": _last(macd_s - sig_s),
        "rsi": _last(rsi_s),
        "bb_upper": bb_u,
        "bb_middle": bb_m,
        "bb_lower": bb_l,
        "bb_width_pct": bb_w,
        "k": k_vals[-1] if k_vals else None,
        "d": d_vals[-1] if d_vals else None,
        "j": j_vals[-1] if j_vals else None,
        "support": support_val,
        "resistance": resistance_val,
        "vol_ratio": vr,
        "vol_trend": vol_trend,
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "divergence": div_text,
        "macd_cross": macd_cross,
    }


# ── Weekly Trend ─────────────────────────────────────────────────────

def fetch_weekly_trend(symbol: str) -> dict:
    """Fetch weekly K-line from Tencent to determine medium-term trend."""
    try:
        r = _http_get_with_retry(
            TENCENT_KLINE_URL,
            params={"param": f"{symbol},week,,,30,qfq"},
        )
        if r and r.text:
            raw = r.json()
            data = raw.get("data", {}).get(symbol, {})
            klines = data.get("qfqweek")
            if klines and len(klines) >= 4:
                # Tencent format: [date, open, close, high, low, volume]
                closes = [float(row[2]) for row in klines if row[2]]
                if len(closes) >= 4:
                    prev = closes[-2]
                    curr = closes[-1]
                    change = (curr - prev) / prev * 100.0 if prev > 0 else 0.0
                    trend = "上涨" if change > 0 else "下跌"
                    return {"weekly_change_pct": change, "weekly_trend": trend}
    except Exception:
        logger.debug("Weekly K-line fetch failed", exc_info=True)
    return {"weekly_change_pct": None, "weekly_trend": ""}


# ── ADX ──────────────────────────────────────────────────────────────

def compute_adx(high, low, close, period: int = 14):
    """Compute ADX, +DI, -DI using Wilder's smoothing."""
    try:
        h = high.values if hasattr(high, "values") else high
        l = low.values if hasattr(low, "values") else low
        c = close.values if hasattr(close, "values") else close
    except Exception:
        return None, None, None

    if len(h) < period + 2:
        return None, None, None

    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    atr = np.zeros_like(tr); atr[0] = tr[0]
    pdi_raw = np.zeros_like(tr); pdi_raw[0] = plus_dm[0]
    mdi_raw = np.zeros_like(tr); mdi_raw[0] = minus_dm[0]

    for i in range(1, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
        pdi_raw[i] = pdi_raw[i-1] * (1 - alpha) + plus_dm[i] * alpha
        mdi_raw[i] = mdi_raw[i-1] * (1 - alpha) + minus_dm[i] * alpha

    plus_di = np.where(atr > 0, 100 * pdi_raw / atr, 0.0)
    minus_di = np.where(atr > 0, 100 * mdi_raw / atr, 0.0)
    denom = plus_di + minus_di
    dx = np.zeros_like(plus_di)
    mask = denom > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / denom[mask]

    adx = np.zeros_like(dx); adx[0] = dx[0]
    for i in range(1, len(dx)):
        adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha

    return (
        float(adx[-1]) if not np.isnan(adx[-1]) else None,
        float(plus_di[-1]) if not np.isnan(plus_di[-1]) else None,
        float(minus_di[-1]) if not np.isnan(minus_di[-1]) else None,
    )


# ── Divergence Detection ─────────────────────────────────────────────

def _find_swings(data, window: int = 5):
    """Find local peaks and troughs in a 1-d array."""
    peaks, troughs = [], []
    for i in range(window, len(data) - window):
        left = data[i - window:i]
        right = data[i + 1:i + window + 1]
        if data[i] >= max(left) and data[i] > max(right):
            peaks.append(i)
        if data[i] <= min(left) and data[i] < min(right):
            troughs.append(i)
    return peaks, troughs


def detect_divergence(close, rsi_s, macd_s, window: int = 5) -> str:
    """Detect RSI/MACD divergence against price."""
    try:
        c = close.values if hasattr(close, "values") else close
        r = rsi_s.values if hasattr(rsi_s, "values") else rsi_s
        m = macd_s.values if hasattr(macd_s, "values") else macd_s
    except Exception:
        return ""

    valid = np.isfinite(c) & np.isfinite(r) & np.isfinite(m)
    c, r, m = c[valid], r[valid], m[valid]
    if len(c) < 20:
        return ""

    tail = min(30, len(c))
    c_tail = c[-tail:]
    r_tail = r[-tail:]
    m_tail = m[-tail:]

    peaks, troughs = _find_swings(c_tail, window)
    _, r_peaks = _find_swings(r_tail, window)
    r_inv = -r_tail
    _, r_troughs = _find_swings(r_inv, window)
    _, m_peaks = _find_swings(m_tail, window)
    m_inv = -m_tail
    _, m_troughs = _find_swings(m_inv, window)

    # Top divergence
    if len(peaks) >= 2:
        i1, i2 = peaks[-2], peaks[-1]
        if c_tail[i2] > c_tail[i1]:
            rsi_div = (
                any(abs(p - i2) <= 2 for p in r_peaks)
                and any(abs(p - i1) <= 2 for p in r_peaks)
                and r_tail[i2] < r_tail[i1]
            )
            macd_div = (
                any(abs(p - i2) <= 2 for p in m_peaks)
                and any(abs(p - i1) <= 2 for p in m_peaks)
                and m_tail[i2] < m_tail[i1]
            )
            if rsi_div or macd_div:
                return "顶背离"

    # Bottom divergence
    if len(troughs) >= 2:
        i1, i2 = troughs[-2], troughs[-1]
        if c_tail[i2] < c_tail[i1]:
            rsi_div = (
                any(abs(t - i2) <= 2 for t in r_troughs)
                and any(abs(t - i1) <= 2 for t in r_troughs)
                and r_tail[i2] > r_tail[i1]
            )
            macd_div = (
                any(abs(t - i2) <= 2 for t in m_troughs)
                and any(abs(t - i1) <= 2 for t in m_troughs)
                and m_tail[i2] > m_tail[i1]
            )
            if rsi_div or macd_div:
                return "底背离"

    return ""


# ── Trading Signal Analysis ──────────────────────────────────────────

def score_stock(price: float, change_pct: float,
                indicators: dict, weekly: dict) -> int:
    """Score a stock based on technical indicators (0 = neutral, + bullish, - bearish).

    Uses MA, MACD, RSI, Bollinger, KDJ, Volume, ADX, Divergence, Weekly trend
    to produce a composite score.  Returns a single integer.
    """
    # Unpack indicators
    ma5 = indicators["ma5"]
    ma10 = indicators["ma10"]
    ma20 = indicators["ma20"]
    macd_val = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    rsi = indicators["rsi"]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_width = indicators["bb_width_pct"]
    j_val = indicators["j"]
    adx = indicators["adx"]
    plus_di = indicators["plus_di"]
    minus_di = indicators["minus_di"]
    divergence = indicators["divergence"]
    weekly_trend = weekly.get("weekly_trend", "")
    macd_cross = indicators["macd_cross"]
    vol_trend = indicators["vol_trend"]

    score = 0

    # -- MA --
    if price and ma20:
        if price > ma20:
            score += 1
        else:
            score -= 1
    if ma5 and ma10:
        if ma5 > ma10:
            score += 1
        else:
            score -= 1

    # -- MACD --
    if macd_cross == "金叉":
        score += 1
    elif macd_cross == "死叉":
        score -= 1

    # -- RSI --
    if rsi is not None:
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

    # -- Bollinger --
    if price and bb_upper and bb_lower:
        if price >= bb_upper * 0.99:
            score -= 1
        elif price <= bb_lower * 1.01:
            score += 1

    # -- Volume --
    if vol_trend == "放量":
        if change_pct >= 0:
            score += 1
        else:
            score -= 1

    # -- KDJ --
    if j_val is not None:
        if j_val > 100:
            score -= 1
        elif j_val < 0:
            score += 1

    # -- ADX --
    if adx is not None:
        if adx > 40:
            if plus_di and minus_di and plus_di > minus_di:
                score += 1
            else:
                score -= 1

    # -- Divergence --
    if divergence == "顶背离":
        score -= 2
    elif divergence == "底背离":
        score += 2

    # -- Weekly trend --
    if weekly_trend == "上涨":
        score += 1
    elif weekly_trend == "下跌":
        score -= 1

    return score


def analyze_stock(name: str, code: str, price: float, change_pct: float,
                  indicators: dict, weekly: dict) -> str:
    """Score all indicators and generate a detailed trading signal message."""
    # Unpack indicators
    ma5 = indicators["ma5"]
    ma10 = indicators["ma10"]
    ma20 = indicators["ma20"]
    macd_val = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    rsi = indicators["rsi"]
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_width = indicators["bb_width_pct"]
    k_val = indicators["k"]
    d_val = indicators["d"]
    j_val = indicators["j"]
    support = indicators["support"]
    resistance = indicators["resistance"]
    vol_ratio = indicators["vol_ratio"]
    vol_trend = indicators["vol_trend"]
    adx = indicators["adx"]
    plus_di = indicators["plus_di"]
    minus_di = indicators["minus_di"]
    divergence = indicators["divergence"]
    weekly_trend = weekly.get("weekly_trend", "")
    macd_cross = indicators["macd_cross"]

    notes: list[str] = []

    # -- MA --
    if price and ma20:
        if price > ma20:
            notes.append("股价站在均线上方，中线趋势向好")
        else:
            notes.append("股价跌到均线下方，中线走势偏弱")
    if ma5 and ma10:
        if ma5 > ma10:
            notes.append("短期均线向上，短线有支撑")
        else:
            notes.append("短期均线向下，短线承压")

    # -- MACD --
    if macd_cross == "金叉":
        notes.append("MACD近期金叉，多头信号")
    elif macd_cross == "死叉":
        notes.append("MACD近期死叉，空头信号")
    elif macd_val is not None and macd_signal is not None:
        if macd_val > macd_signal:
            notes.append("MACD在零轴上方运行，趋势偏多")
        else:
            notes.append("MACD在零轴下方运行，趋势偏弱")

    # -- RSI --
    if rsi is not None:
        if rsi < 30:
            notes.append("跌过头了，随时可能反弹")
        elif rsi > 70:
            notes.append("涨太快了，小心短期回调")
        else:
            notes.append("市场情绪正常，没有极端信号")

    # -- Bollinger --
    if price and bb_upper and bb_lower:
        if price >= bb_upper * 0.99:
            notes.append("价格顶着布林上轨，上方空间有限")
        elif price <= bb_lower * 1.01:
            notes.append("价格贴着布林下轨，有反弹空间")
        elif bb_width is not None and bb_width < 5:
            notes.append("布林带收窄，快要变盘了")

    # -- Volume --
    if vol_trend == "放量":
        notes.append("成交量放大，资金在动")
        if change_pct >= 0:
            notes.append("放量上涨，量价配合不错")
        else:
            notes.append("放量下跌，出货迹象要留意")
    elif vol_trend == "缩量":
        notes.append("成交缩量，市场参与度不高")

    # -- KDJ --
    if j_val is not None:
        if j_val > 100:
            notes.append("KDJ高位钝化，追高要谨慎")
        elif j_val < 0:
            notes.append("KDJ在低位区，超跌反弹可期")

    # -- ADX --
    if adx is not None:
        if adx > 40:
            if plus_di and minus_di and plus_di > minus_di:
                notes.append("ADX显示强势上涨，顺势而为")
            else:
                notes.append("ADX显示强势下跌，不要逆势")
        elif adx > 20:
            notes.append("趋势正在形成中，方向开始明朗")
        else:
            notes.append("ADX偏低，行情偏震荡，指标信号参考价值有限")

    # -- Divergence --
    if divergence == "顶背离":
        notes.append("顶背离！股价新高但MACD/RSI没跟上，大概率要回调")
    elif divergence == "底背离":
        notes.append("底背离！股价新低但指标拒绝跟随，反转机会来了")

    # -- Weekly trend --
    if weekly_trend == "上涨":
        notes.append("周线也在涨，中线趋势没坏")
    elif weekly_trend == "下跌":
        notes.append("周线是跌的，日线反弹要打折扣")

    # -- Support / Resistance --
    if price and support and resistance:
        dist_to_support = (price - support) / support * 100
        dist_to_resist = (resistance - price) / price * 100
        if dist_to_support < 3:
            notes.append(f"离支撑位很近（{dist_to_support:.1f}%），下跌空间不大")
        if dist_to_resist < 3:
            notes.append(f"离压力位不远（{dist_to_resist:.1f}%），突破才能看高")

    # Compute score using shared function
    score = score_stock(price, change_pct, indicators, weekly)

    # -- Build trading signal text --
    detail = "；".join(notes) if notes else "数据不够，再等一等"
    signal, action = _trading_signal_text(
        score=score, divergence=divergence, price=price,
        support=support, resistance=resistance,
        rsi=rsi, j=j_val, adx=adx, vol_trend=vol_trend,
    )

    # -- Build WeCom message --
    t = datetime.now().strftime("%H:%M")
    sign = "+" if change_pct >= 0 else ""
    trend_emoji = "\U0001f534" if change_pct >= 0 else "\U0001f7e2"

    msg = f"\U0001f4ca {name} ({code})  {t}\n"
    msg += chr(0x2501) * 26 + "\n"
    msg += f"\U0001f4b0 现价: ¥{price:.2f}  |  {trend_emoji} {sign}{change_pct:.2f}%\n\n"

    msg += f"\U0001f4c8 均线系统\n"
    msg += f"   MA5:  {ma5:.2f}" if ma5 else "   MA5:  --"
    msg += f"  |  MA10: {ma10:.2f}" if ma10 else "  |  MA10: --"
    msg += f"  |  MA20: {ma20:.2f}\n\n" if ma20 else "  |  MA20: --\n\n"

    msg += f"\U0001f4c9 MACD\n"
    msg += f"   DIF: {macd_val:.2f}" if macd_val is not None else "   DIF: --"
    msg += f"  |  DEA: {macd_signal:.2f}" if macd_signal is not None else "  |  DEA: --"
    hist = (macd_val - macd_signal) if macd_val is not None and macd_signal is not None else None
    msg += f"  |  柱: {hist:.2f}\n\n" if hist is not None else "  |  柱: --\n\n"

    rsi_str = f"{rsi:.1f}" if rsi is not None else "--"
    k_str = f"{k_val:.1f}" if k_val is not None else "--"
    d_str = f"{d_val:.1f}" if d_val is not None else "--"
    j_str = f"{j_val:.1f}" if j_val is not None else "--"
    msg += f"⚡ RSI(14): {rsi_str}  |  KDJ: K={k_str} D={d_str} J={j_str}\n\n"

    bb_u_str = f"{bb_upper:.2f}" if bb_upper is not None else "--"
    bb_l_str = f"{bb_lower:.2f}" if bb_lower is not None else "--"
    msg += f"\U0001f4d0 布林带 (20,2)\n"
    msg += f"   上轨: {bb_u_str}  |  下轨: {bb_l_str}\n"
    if bb_width is not None and price:
        msg += f"   带宽: {(bb_upper - bb_lower):.2f} ({bb_width:.1f}%)\n"
    msg += "\n"

    sup_str = f"¥{support:.2f}" if support is not None else "--"
    res_str = f"¥{resistance:.2f}" if resistance is not None else "--"
    msg += f"\U0001f3af 支撑/压力\n"
    msg += f"   支撑: {sup_str}  |  压力: {res_str}\n"
    if price and support and resistance:
        msg += f"   距支撑: {((price-support)/price*100):.1f}%  |  距压力: {((resistance-price)/price*100):.1f}%\n"
    msg += "\n"

    vr_str = f"{vol_ratio:.2f}" if vol_ratio is not None else "--"
    msg += f"\U0001f4ca 量比: {vr_str} ({vol_trend})  |  周线: {weekly_trend or '未知'}\n\n"

    msg += f"\U0001f9e0 综合评分: {score}/10\n"
    msg += f"\U0001f4a1 交易信号:\n{signal}\n{action}\n"
    msg += f"\n---\n{detail}"

    return msg


def _trading_signal_text(*, score, divergence, price, support, resistance,
                         rsi, j, adx, vol_trend) -> tuple:
    """Generate trading signal text. Returns (signal_line, action_lines)."""
    targets = []
    if support and resistance and price:
        downside = (price - support) / price * 100
        upside = (resistance - price) / price * 100
        targets.append(f"支撑位 ¥{support:.2f}（距现价 {downside:.1f}%）")
        targets.append(f"压力位 ¥{resistance:.2f}（距现价 {upside:.1f}%）")

    target_line = " | ".join(targets)

    if divergence == "顶背离" and score <= -1:
        signal = "⚠️ 卖出信号 — 顶背离"
        action = "顶背离是强烈的见顶信号，建议减仓或清仓止盈"
    elif divergence == "底背离" and score >= 0:
        signal = "\U0001f525 买入信号 — 底背离"
        action = "底背离通常预示反转，可考虑分批建仓"
    elif score >= 5:
        signal = "\U0001f4c8 强烈看涨，持仓待涨"
        action = "指标全面偏多，可继续持有"
    elif score >= 3:
        signal = "\U0001f4c8 偏多，逢低可加仓"
        action = "趋势向好但不宜追高"
    elif score >= 1:
        signal = "\U0001f4ca 震荡偏多，持有观望"
        action = "短期方向不明显，已经有了可以先不动"
    elif score <= -5:
        signal = "\U0001f4c9 强烈看跌，建议清仓"
        action = "指标全面偏空，持币观望更安全"
    elif score <= -3:
        signal = "\U0001f4c9 偏空，减仓或止盈"
        action = "走势偏弱，仓位重的建议减一部分"
    elif score <= -1:
        signal = "\U0001f4ca 震荡偏弱，多看少动"
        action = "方向偏弱但不算极端，有持仓可减，没持仓先等等"
    else:
        signal = "\U0001f4ca 观望 — 等信号明确"
        action = "目前多空力量均衡，没有明确方向"

    return signal, f"{action}\n{target_line}"


# ── AI Analysis (DeepSeek) ───────────────────────────────────────────

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def _build_ai_prompt(name: str, code: str, price: float, change_pct: float,
                     indicators: dict, weekly: dict) -> str:
    """Format all technical indicators into a structured prompt for the AI."""
    sign = "+" if change_pct >= 0 else ""

    def fmt(v, precision=2):
        if v is None:
            return "--"
        return f"{v:.{precision}f}"

    prompt = f"""请分析以下A股的技术指标数据，给出简洁的综合判断和操作建议（200字以内）：

股票：{name}（{code}）
现价：¥{price:.2f}（{sign}{change_pct:.2f}%）

【均线系统】
MA5={fmt(indicators['ma5'])}  MA10={fmt(indicators['ma10'])}  MA20={fmt(indicators['ma20'])}

【MACD】
DIF={fmt(indicators['macd'])}  DEA={fmt(indicators['macd_signal'])}  柱={fmt(indicators['macd_hist'])}

【RSI】{fmt(indicators['rsi'], 1)}
【KDJ】K={fmt(indicators['k'], 1)} D={fmt(indicators['d'], 1)} J={fmt(indicators['j'], 1)}

【布林带(20,2)】
上轨={fmt(indicators['bb_upper'])} 中轨={fmt(indicators['bb_middle'])} 下轨={fmt(indicators['bb_lower'])} 带宽={fmt(indicators['bb_width_pct'], 1)}%

【成交量】
量比={fmt(indicators['vol_ratio'])}（{indicators['vol_trend']}）

【ADX】{fmt(indicators['adx'], 1)}  +DI={fmt(indicators['plus_di'], 1)}  -DI={fmt(indicators['minus_di'], 1)}

【背离信号】{indicators['divergence'] or '无'}

【支撑/压力】
支撑位=¥{fmt(indicators['support'])}  压力位=¥{fmt(indicators['resistance'])}

【周线趋势】{weekly.get('weekly_trend', '未知')}

请从以下角度简要分析：
1. 当前趋势判断（多头/空头/震荡）
2. 关键技术信号（注意背离、金叉死叉、超买超卖）
3. 短期操作建议
4. 主要风险提示"""
    return prompt


def ask_deepseek(name: str, code: str, price: float, change_pct: float,
                 indicators: dict, weekly: dict) -> Optional[str]:
    """Ask DeepSeek v4-pro for AI-powered stock analysis. Returns text or None."""
    if not DEEPSEEK_API_KEY:
        logger.debug("DeepSeek API key not configured, skipping AI analysis")
        return None

    prompt = _build_ai_prompt(name, code, price, change_pct, indicators, weekly)

    payload = {
        "model": "deepseek-v4-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一位资深A股技术分析师，擅长综合多种技术指标进行行情研判。"
                    "请基于提供的数据给出客观、专业的分析。"
                    "回答应简洁务实，避免模棱两可的表述。"
                    "结尾请加注：以上为AI分析，仅供参考，不构成投资建议。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.6,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        logger.info("  Asking DeepSeek for analysis...")
        r = SESSION.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            # Fall back to reasoning_content if content is empty
            text = (content or message.get("reasoning_content", "") or "").strip()
            if text:
                logger.info("  AI analysis received (%d chars)", len(text))
                return text
            logger.warning("  DeepSeek returned empty content")
        else:
            logger.warning("  DeepSeek API status %d: %s", r.status_code, r.text[:200])
    except requests.Timeout:
        logger.warning("  DeepSeek API timed out")
    except Exception as e:
        logger.warning("  DeepSeek API call failed: %s", e)

    return "AI 分析暂时不可用，请稍后重试"


def _format_multi_agent_section(agent_results: dict, discussion: str, decision: dict) -> str:
    """Format multi-agent analysis results into a WeCom message block."""
    lines = ["\n\U0001f916 AI 多维度分析", "━" * 26]

    # Technical
    tech = agent_results.get("technical", {}).get("analysis", "")
    if tech:
        lines.append(f"【技术面】{tech[:200]}")

    # Fundamental
    fund = agent_results.get("fundamental", {}).get("analysis", "")
    if fund:
        lines.append(f"【基本面】{fund[:200]}")

    # Fund flow
    flow = agent_results.get("fund_flow", {}).get("analysis", "")
    if flow:
        lines.append(f"【资金面】{flow[:200]}")

    # Team discussion
    if discussion:
        lines.append(f"【团队讨论】{discussion[:300]}")

    # Final decision
    if decision:
        rating = decision.get("rating", "N/A")
        target = decision.get("target_price", "N/A")
        stop_loss = decision.get("stop_loss", "N/A")
        take_profit = decision.get("take_profit", "")
        position = decision.get("position_size", "N/A")
        confidence = decision.get("confidence_level", "N/A")
        risk = decision.get("risk_warning", decision.get("decision_text", ""))

        # Sanity check: target should be > stop_loss for buy/hold ratings
        warning = ""
        try:
            t_val = float(target)
            s_val = float(stop_loss)
            is_bearish = "卖出" in str(rating) or "看空" in str(rating)
            if not is_bearish and s_val >= t_val:
                warning = " ⚠️目标价≤止损位，数据可能有误"
        except (ValueError, TypeError):
            pass

        lines.append(
            f"【最终决策】{warning}\n"
            f"  评级: {rating}  |  目标价: {target}  |  止损位: {stop_loss}"
        )
        if take_profit:
            lines[-1] += f"  |  止盈: {take_profit}"
        lines.append(f"  建议仓位: {position}  |  信心度: {confidence}/10")
        if isinstance(risk, str) and len(risk) < 200:
            lines.append(f"  风险: {risk}")

    lines.append("⚠️ 以上为AI分析，仅供参考，不构成投资建议。")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def process_stock(code: str) -> Optional[dict]:
    """Process a single stock: fetch data, compute indicators, return analysis dict.

    Returns dict with keys: code, name, price, change_pct, indicators, weekly,
    text_msg, ai_text, agent_results, discussion, decision.
    Returns None if data fetch fails.
    """
    prefix = code_prefix(code)
    symbol = f"{prefix}{code}"
    logger.info("Processing %s (%s)...", code, symbol)

    # Step 1: Real-time quote
    name, price, change_pct, _change_amt = fetch_quote(symbol)
    if price == 0.0 and not name:
        logger.warning("Stock %s not found, skipping", code)
        return None
    logger.info("  %s: ¥%.2f (%+.2f%%)", name, price, change_pct)

    # Step 2: Daily K-line + indicators
    df = fetch_kline(symbol)
    if df is None:
        logger.warning("  No K-line data for %s", code)
        return None
    indicators = compute_indicators(df)
    if indicators is None:
        logger.warning("  Insufficient data for indicators on %s", code)
        return None

    # Step 3: Weekly trend
    weekly = fetch_weekly_trend(symbol)

    # Step 4: Analyze and format
    text_msg = analyze_stock(name, code, price, change_pct, indicators, weekly)

    # Step 5: AI analysis (optional — skipped if no API key configured)
    ai_text = ask_deepseek(name, code, price, change_pct, indicators, weekly)
    if ai_text:
        text_msg += f"\n\n\U0001f916 AI 视角\n{'─' * 20}\n{ai_text}"

    # Step 6: Enhanced multi-agent analysis (optional — gated by ENABLE_ENHANCED_ANALYSIS)
    agent_results = {}
    discussion = ""
    decision = {}
    if _ENHANCED_ENABLED:
        try:
            logger.info("  Enhanced multi-agent analysis...")
            stock_info = {
                "symbol": code,
                "name": name,
                "current_price": price,
                "change_percent": change_pct,
            }

            # Fetch enhanced data (quarterly reports + fund flow, cached daily)
            enhanced = _enhanced_data_mgr.fetch_for_stock(code)

            # Run multi-agent analysis
            agents = StockAnalysisAgents()
            agent_results = agents.run_multi_agent_analysis(
                stock_info=stock_info,
                indicators=indicators,
                quarterly_text=enhanced.get("quarterly_text", ""),
                fund_flow_text=enhanced.get("fund_flow_text", ""),
            )

            # Team discussion + final decision
            discussion = agents.conduct_team_discussion(agent_results, stock_info)
            decision = agents.make_final_decision(discussion, stock_info, indicators)

            # Append to text message
            section = _format_multi_agent_section(agent_results, discussion, decision)
            text_msg += "\n" + section
            logger.info("  Enhanced analysis appended to message.")
        except Exception as e:
            logger.warning("  Enhanced analysis failed for %s: %s", code, e)

    return {
        "code": code,
        "name": name,
        "price": price,
        "change_pct": change_pct,
        "indicators": indicators,
        "weekly": weekly,
        "text_msg": text_msg,
        "ai_text": ai_text,
        "agent_results": agent_results,
        "discussion": discussion,
        "decision": decision,
    }


def _extract_webhook_key() -> Optional[str]:
    """Extract the 'key' parameter from WEBHOOK_URL."""
    import re
    m = re.search(r'key=([a-f0-9-]+)', WEBHOOK_URL)
    return m.group(1) if m else None


def _upload_to_wecom(filepath: str) -> Optional[str]:
    """Upload a file to WeCom via webhook upload API. Returns media_id or None."""
    key = _extract_webhook_key()
    if not key:
        return None
    upload_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file"
    try:
        with open(filepath, "rb") as f:
            r = SESSION.post(upload_url, files={"file": f}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data.get("errcode") == 0:
                return data.get("media_id")
            logger.warning("  Upload failed: %s", r.text)
        return None
    except Exception as e:
        logger.warning("  Upload error: %s", e)
        return None


def send_html_report(html: str, filename: str) -> bool:
    """Generate HTML file, upload to WeCom, and send as file. Falls back to text."""
    if not WEBHOOK_URL:
        logger.error("WECOM_WEBHOOK_URL not configured!")
        return False

    temp_path = f"/tmp/{filename}"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        logger.error("  Failed to write HTML: %s", e)
        return False

    media_id = _upload_to_wecom(temp_path)
    if media_id:
        payload = {"msgtype": "file", "file": {"media_id": media_id}}
        try:
            r = SESSION.post(WEBHOOK_URL, json=payload, timeout=10)
            if r.json().get("errcode") == 0:
                logger.info("  HTML report sent to WeCom.")
                return True
            logger.warning("  File send failed: %s", r.text)
        except Exception as e:
            logger.warning("  File send error: %s", e)

    # Fallback: send as text (truncated)
    logger.info("  Falling back to text push...")
    return send_to_wecom(html[:4000])


def send_to_wecom(msg: str) -> bool:
    """Send a plain-text message to WeCom webhook. Returns True on success."""
    if not WEBHOOK_URL:
        logger.error("WECOM_WEBHOOK_URL not configured!")
        return False

    payload = {"msgtype": "text", "text": {"content": msg}}
    try:
        r = SESSION.post(WEBHOOK_URL, json=payload, timeout=10)
        result = r.json()
        if result.get("errcode") == 0:
            logger.info("  Sent to WeCom successfully.")
            return True
        else:
            logger.error("  WeCom error: %s", r.text)
            return False
    except Exception as e:
        logger.error("  WeCom request failed: %s", e)
        return False


def run_report(screening_text: str = ""):
    """Execute one full report cycle for all stocks — produce single merged HTML."""
    logger.info("=" * 50)
    logger.info("Report cycle — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    all_data = []
    for code in STOCK_CODES:
        try:
            data = process_stock(code)
            if data:
                all_data.append(data)
        except Exception as e:
            logger.error("Failed to process %s: %s", code, e)

    if not all_data:
        logger.warning("No stock data collected, skipping report.")
        return

    # Generate merged HTML
    try:
        from lib.report_html import build_merged_report, save_report

        html = build_merged_report(all_data, title="股票监控报告", screening_section=screening_text)
        ts = datetime.now().strftime("%Y-%m-%d")
        filepath = save_report(html, prefix="merged", subdir=ts)

        # Save path hint for cc-connect pickup
        hint_path = f"/tmp/last_stock_report.txt"
        with open(hint_path, "w") as f:
            f.write(filepath)

        logger.info("Merged HTML saved: %s", filepath)

        # Push HTML file to WeCom, fall back to per-stock text if upload fails
        filename = f"stock_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        html_sent = send_html_report(html, filename)
        if not html_sent:
            for d in all_data:
                send_to_wecom(d["text_msg"])
    except ImportError:
        logger.warning("report_html not available, falling back to per-stock text")
        for d in all_data:
            send_to_wecom(d["text_msg"])
    except Exception as e:
        logger.warning("HTML build failed: %s, falling back to text", e)
        for d in all_data:
            send_to_wecom(d["text_msg"])

    logger.info("Cycle complete.")


def daemon_loop():
    """Run reports on a configurable interval during A-share market hours.

    Market hours: Mon–Fri, 9:30–15:00.  The daemon will push one report
    immediately on start, then sleep REPORT_INTERVAL_MINUTES between pushes.
    It exits automatically after 14:35 (last push before market close).
    """
    interval_sec = REPORT_INTERVAL_MINUTES * 60
    logger.info("Daemon started | interval=%d min | stocks=%s",
                REPORT_INTERVAL_MINUTES, STOCK_CODES)

    # ── Daily screening (once per daemon session) ────────────────────
    _screening_done_date = ""
    if _ENHANCED_ENABLED:
        # SCREENING_ENABLED is read from lib.config when enhanced mode is on
        try:
            _screening_cfg = get_config()
            _screening_on = _screening_cfg.screening_enabled
        except Exception:
            _screening_on = False
    else:
        _screening_on = False

    _screening_text = ""  # stored for merging into HTML report

    def _run_daily_screening():
        """Run all screening strategies once, push text + store for HTML merge."""
        nonlocal _screening_done_date, _screening_text
        today = datetime.now().strftime("%Y-%m-%d")
        if _screening_done_date == today:
            return
        try:
            from lib.selectors import run_all_screenings
            logger.info("Running daily stock screening...")
            msg = run_all_screenings(top_n=5)
            if msg:
                header = "\U0001f50d 每日选股扫描\n" + "━" * 20 + "\n"
                _screening_text = header + msg
                send_to_wecom(header + msg)
                logger.info("Daily screening sent to WeCom (text) and staged for HTML merge.")
            _screening_done_date = today
        except Exception as e:
            logger.warning("Daily screening failed: %s", e)

    # Run screening once at startup if enabled
    if _screening_on:
        # Wait until market is open (9:30) if started early
        _now = datetime.now()
        _market_open = _now.replace(hour=9, minute=30, second=30, microsecond=0)
        if _now < _market_open:
            _wait = (_market_open - _now).total_seconds()
            logger.info("Market not open yet, waiting %.0fs until 9:30:30...", _wait)
            time.sleep(_wait)
        _run_daily_screening()

    # Graceful shutdown
    shutdown_flag = False

    def _handle_signal(sig, frame):
        nonlocal shutdown_flag
        logger.info("Received signal %s, shutting down gracefully...", sig)
        shutdown_flag = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not shutdown_flag:
        now = datetime.now()

        # Exit on weekends
        if now.weekday() >= 5:
            logger.info("Weekend — exiting daemon")
            break

        # Exit past 15:00
        if now.hour >= 15:
            logger.info("Past 15:00 — exiting daemon")
            break

        # Run one cycle (include screening only on first cycle)
        cycle_start = datetime.now()
        run_report(screening_text=_screening_text)
        _screening_text = ""  # only first cycle carries screening

        if shutdown_flag:
            break

        # Calculate next run from cycle START time, not end time,
        # so run duration doesn't cause drift.
        next_time = cycle_start + timedelta(minutes=REPORT_INTERVAL_MINUTES)

        # Check if next run would be past market close
        if next_time.weekday() >= 5:
            logger.info("Next run falls on weekend — exiting daemon")
            break
        if next_time.hour >= 15:
            logger.info("Next run (%s) past 15:00 — exiting daemon",
                        next_time.strftime("%H:%M"))
            break

        # Sleep the remaining time, accounting for the run duration
        elapsed = (datetime.now() - cycle_start).total_seconds()
        remaining = interval_sec - elapsed
        if remaining < 0:
            remaining = 0
        logger.info("Next report in %d min (around %s)",
                    int(remaining / 60), next_time.strftime("%H:%M"))
        while remaining > 0 and not shutdown_flag:
            chunk = min(remaining, 30)
            time.sleep(chunk)
            remaining -= chunk

    logger.info("Daemon stopped.")


def main():
    """Main entry point.

    Usage:
        python cloud_stock_reporter.py            # single run
        python cloud_stock_reporter.py --daemon   # run on interval during market hours
    """
    if not WEBHOOK_URL:
        logger.error(
            "WECOM_WEBHOOK_URL is required. "
            "Set it via environment variable or .env file."
        )
        sys.exit(1)

    if "--daemon" in sys.argv:
        daemon_loop()
    else:
        # Single-run mode (original behaviour)
        logger.info("Stock codes: %s", STOCK_CODES)
        logger.info("Webhook URL: %s...", WEBHOOK_URL[:50])
        if DEEPSEEK_API_KEY:
            logger.info("DeepSeek AI: enabled")
        else:
            logger.info("DeepSeek AI: disabled (no API key)")

        run_report()


if __name__ == "__main__":
    main()
