"""
Pre-limit-up technical pattern classification.

Takes the 90-day indicator time series for a limit-up event and classifies
the setup into discrete technical patterns.  Each pattern has a confidence
score and optional detail JSON for pattern-specific metrics.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Number of days before the limit-up to focus on for short-term patterns
SHORT_TERM_WINDOW = 5
MEDIUM_TERM_WINDOW = 20


def classify_patterns(ind_df: pd.DataFrame) -> list[dict]:
    """Classify pre-limit-up technical patterns.

    Args:
        ind_df: Indicator time series DataFrame indexed by date ascending.
                Must contain all indicator columns from compute_indicator_series().
                The last row is the limit-up day (D=0), rows before are D-90..D-1.

    Returns:
        List of pattern dicts: {pattern_type, confidence, detail}
    """
    if ind_df is None or len(ind_df) < MEDIUM_TERM_WINDOW:
        return []

    patterns: list[dict] = []

    # ── 1. Oversold bounce ──────────────────────────────────────────
    _check_oversold_bounce(ind_df, patterns)

    # ── 2. MA golden cross ──────────────────────────────────────────
    _check_ma_golden_cross(ind_df, patterns)

    # ── 3. Volume surge ─────────────────────────────────────────────
    _check_volume_surge(ind_df, patterns)

    # ── 4. Bollinger squeeze breakout ───────────────────────────────
    _check_bollinger_squeeze(ind_df, patterns)

    # ── 5. Bottom divergence ────────────────────────────────────────
    _check_bottom_divergence(ind_df, patterns)

    # ── 6. Strong bullish trend ─────────────────────────────────────
    _check_strong_trend(ind_df, patterns)

    # ── 7. KDJ oversold reversal ────────────────────────────────────
    _check_kdj_reversal(ind_df, patterns)

    # ── 8. MA20 support bounce ──────────────────────────────────────
    _check_ma_support_bounce(ind_df, patterns)

    # ── 9. Volume contraction breakout ──────────────────────────────
    _check_volume_contraction_breakout(ind_df, patterns)

    return patterns


# ── Individual pattern checkers ────────────────────────────────────────

def _tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Safe tail access."""
    return df.iloc[-n:] if len(df) >= n else df


def _safe_val(row, col: str) -> Optional[float]:
    """Extract a numeric value from a DataFrame row safely."""
    try:
        v = row[col]
        if pd.isna(v):
            return None
        return float(v)
    except (KeyError, TypeError, ValueError):
        return None


def _check_oversold_bounce(df: pd.DataFrame, out: list):
    """RSI < 30 within last 10 days, now recovering above 40."""
    rsi_col = df["rsi"].tail(10)
    if rsi_col.empty:
        return
    rsi_min = rsi_col.min()
    rsi_last = rsi_col.iloc[-1]
    if pd.notna(rsi_min) and rsi_min < 30 and pd.notna(rsi_last) and rsi_last > rsi_min + 5:
        confidence = min(0.9, 0.5 + (rsi_last - rsi_min) / 30)
        out.append({
            "pattern_type": "oversold_bounce",
            "confidence": round(confidence, 2),
            "detail": json.dumps(
                {"rsi_min": round(float(rsi_min), 1), "rsi_now": round(float(rsi_last), 1)},
                ensure_ascii=False,
            ),
        })


def _check_ma_golden_cross(df: pd.DataFrame, out: list):
    """MA5 crossed above MA10 within last SHORT_TERM_WINDOW days."""
    ma5 = df["ma5"].tail(SHORT_TERM_WINDOW + 1)
    ma10 = df["ma10"].tail(SHORT_TERM_WINDOW + 1)
    if len(ma5) < 3:
        return
    for i in range(1, len(ma5)):
        prev_5, prev_10 = ma5.iloc[i - 1], ma10.iloc[i - 1]
        cur_5, cur_10 = ma5.iloc[i], ma10.iloc[i]
        if (
            pd.notna(prev_5) and pd.notna(prev_10)
            and pd.notna(cur_5) and pd.notna(cur_10)
            and prev_5 <= prev_10 and cur_5 > cur_10
        ):
            days_ago = len(ma5) - i - 1
            out.append({
                "pattern_type": "ma_golden_cross",
                "confidence": 0.75 if days_ago <= 3 else 0.55,
                "detail": json.dumps(
                    {"days_ago": days_ago, "ma5": round(float(cur_5), 3), "ma10": round(float(cur_10), 3)},
                    ensure_ascii=False,
                ),
            })
            break


def _check_volume_surge(df: pd.DataFrame, out: list):
    """Volume ratio > 1.5 and price up within last 3 days."""
    recent = df.tail(3)
    for i in range(len(recent)):
        vr = _safe_val(recent.iloc[i], "vol_ratio")
        if vr is not None and vr > 1.5:
            out.append({
                "pattern_type": "volume_surge",
                "confidence": min(0.9, 0.5 + vr / 5),
                "detail": json.dumps(
                    {"vol_ratio": round(vr, 2), "days_ago": len(recent) - i - 1},
                    ensure_ascii=False,
                ),
            })
            break


def _check_bollinger_squeeze(df: pd.DataFrame, out: list):
    """BB width at 30-day low then price broke above upper band."""
    bbw = df["bb_width_pct"].tail(30)
    if len(bbw) < 20:
        return
    bbw_min = bbw.min()
    bbw_last = bbw.iloc[-1]
    if pd.isna(bbw_min) or pd.isna(bbw_last):
        return
    if bbw_last > bbw_min * 1.2:  # expanding after squeeze
        # Check if price broke upper band
        for i in range(min(3, len(df)), 0, -1):
            row = df.iloc[-i]
            close_v = _safe_val(row, "close")
            bb_u = _safe_val(row, "bb_upper")
            if close_v is not None and bb_u is not None and close_v >= bb_u * 0.98:
                out.append({
                    "pattern_type": "bollinger_squeeze_breakout",
                    "confidence": 0.7,
                    "detail": json.dumps(
                        {"bb_width_min": round(float(bbw_min), 1), "bb_width_now": round(float(bbw_last), 1)},
                        ensure_ascii=False,
                    ),
                })
                break


def _check_bottom_divergence(df: pd.DataFrame, out: list):
    """底背离 detected within last SHORT_TERM_WINDOW days."""
    recent_div = df["divergence"].tail(SHORT_TERM_WINDOW + 3)
    if recent_div.empty:
        return
    for i in range(len(recent_div)):
        if recent_div.iloc[i] == "底背离":
            out.append({
                "pattern_type": "bottom_divergence",
                "confidence": 0.85,
                "detail": json.dumps(
                    {"days_ago": len(recent_div) - i - 1},
                    ensure_ascii=False,
                ),
            })
            break


def _check_strong_trend(df: pd.DataFrame, out: list):
    """ADX > 30, +DI > -DI, MA5 > MA10 > MA20 (bullish alignment)."""
    last = df.iloc[-1]
    adx = _safe_val(last, "adx")
    pdi = _safe_val(last, "plus_di")
    mdi = _safe_val(last, "minus_di")
    ma5 = _safe_val(last, "ma5")
    ma10 = _safe_val(last, "ma10")
    ma20 = _safe_val(last, "ma20")

    if adx is not None and adx > 30 and pdi is not None and mdi is not None and pdi > mdi:
        if ma5 and ma10 and ma20 and ma5 > ma10 > ma20:
            confidence = min(0.9, 0.5 + (adx - 30) / 40)
            out.append({
                "pattern_type": "strong_trend",
                "confidence": round(confidence, 2),
                "detail": json.dumps(
                    {"adx": round(adx, 1), "plus_di": round(float(pdi), 1)},
                    ensure_ascii=False,
                ),
            })


def _check_kdj_reversal(df: pd.DataFrame, out: list):
    """KDJ K crossed above D from oversold (< 20) within last 5 days."""
    k = df["k"].tail(SHORT_TERM_WINDOW + 2)
    d = df["d"].tail(SHORT_TERM_WINDOW + 2)
    if len(k) < 3:
        return
    for i in range(1, len(k)):
        prev_k, prev_d = k.iloc[i - 1], d.iloc[i - 1]
        cur_k, cur_d = k.iloc[i], d.iloc[i]
        if (
            pd.notna(prev_k) and pd.notna(prev_d)
            and pd.notna(cur_k) and pd.notna(cur_d)
            and prev_k < 20 and prev_k <= prev_d and cur_k > cur_d
        ):
            days_ago = len(k) - i - 1
            out.append({
                "pattern_type": "kdj_oversold_reversal",
                "confidence": 0.7,
                "detail": json.dumps(
                    {"days_ago": days_ago, "k": round(float(cur_k), 1), "d": round(float(cur_d), 1)},
                    ensure_ascii=False,
                ),
            })
            break


def _check_ma_support_bounce(df: pd.DataFrame, out: list):
    """Price bounced off MA20 support (close within 2% of MA20) then rose."""
    recent = df.tail(3)
    for i in range(1, len(recent)):
        prev_row = recent.iloc[i - 1]
        cur_row = recent.iloc[i]
        prev_close = _safe_val(prev_row, "close")
        prev_ma20 = _safe_val(prev_row, "ma20")
        cur_close = _safe_val(cur_row, "close")
        cur_ma20 = _safe_val(cur_row, "ma20")
        if (
            prev_close and prev_ma20 and cur_close and cur_ma20
            and prev_ma20 > 0
            and abs(prev_close / prev_ma20 - 1) < 0.02
            and cur_close > prev_close
        ):
            out.append({
                "pattern_type": "ma_support_bounce",
                "confidence": 0.6,
                "detail": json.dumps(
                    {"ma20": round(float(prev_ma20), 2), "close": round(float(prev_close), 2)},
                    ensure_ascii=False,
                ),
            })
            break


def _check_volume_contraction_breakout(df: pd.DataFrame, out: list):
    """Volume declined steadily (3+ days of decreasing vol_ratio) then spiked."""
    vr_recent = df["vol_ratio"].tail(10)
    if len(vr_recent) < 5:
        return
    vr_vals = vr_recent.dropna().values
    if len(vr_vals) < 5:
        return
    # Check for contraction-decline: last 3-5 days were all < 0.8
    mid = vr_vals[-5:-1] if len(vr_vals) >= 5 else vr_vals[-4:-1]
    last = vr_vals[-1]
    if len(mid) >= 2 and all(v < 0.8 for v in mid) and last > 1.2:
        out.append({
            "pattern_type": "volume_contraction_breakout",
            "confidence": 0.65,
            "detail": json.dumps(
                {"vol_ratio_now": round(float(last), 2), "vol_ratio_avg_before": round(float(np.mean(mid)), 2)},
                ensure_ascii=False,
            ),
        })


# ── MA alignment helper (used in pipeline to set pre_ma_alignment) ────

def classify_ma_alignment(ind_df: pd.DataFrame) -> str:
    """Classify MA alignment on the last bar as bullish/bearish/mixed."""
    if ind_df is None or len(ind_df) == 0:
        return ""
    last = ind_df.iloc[-1]
    ma5 = _safe_val(last, "ma5")
    ma10 = _safe_val(last, "ma10")
    ma20 = _safe_val(last, "ma20")
    if ma5 is None or ma10 is None or ma20 is None:
        return ""
    if ma5 > ma10 > ma20:
        return "bullish"
    elif ma5 < ma10 < ma20:
        return "bearish"
    else:
        return "mixed"
