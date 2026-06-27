"""
Full time-series technical indicator computation.

Adapted from cloud_stock_reporter.py's compute_indicators(), extended to
compute every indicator as a column in the output DataFrame (one row per
input K-line bar) so each day in the 90-day pre-limit-up window gets a
complete snapshot.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum bars required to produce valid indicators
MIN_BARS = 26


# ── Helpers ────────────────────────────────────────────────────────────

def _last(s: pd.Series):
    """Return last non-NaN value of a series, or None."""
    v = s.dropna()
    return float(v.iloc[-1]) if not v.empty else None


# ── ADX time series ────────────────────────────────────────────────────

def _compute_adx_series(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute ADX, +DI, -DI as full time series using Wilder's smoothing.

    Returns three Series aligned to the input index (NaN for warmup period).
    """
    h = high.values
    l = low.values
    c = close.values
    n = len(h)

    adx_vals = [np.nan] * n
    pdi_vals = [np.nan] * n
    mdi_vals = [np.nan] * n

    if n < period + 2:
        return (
            pd.Series(adx_vals, index=high.index, dtype=float),
            pd.Series(pdi_vals, index=high.index, dtype=float),
            pd.Series(mdi_vals, index=high.index, dtype=float),
        )

    # True Range
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    up_move = h[1:] - h[:-1]
    down_move = l[:-1] - l[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / period
    atr = np.zeros(len(tr)); atr[0] = tr[0]
    pdi_raw = np.zeros(len(tr)); pdi_raw[0] = plus_dm[0]
    mdi_raw = np.zeros(len(tr)); mdi_raw[0] = minus_dm[0]

    for i in range(1, len(tr)):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
        pdi_raw[i] = pdi_raw[i - 1] * (1 - alpha) + plus_dm[i] * alpha
        mdi_raw[i] = mdi_raw[i - 1] * (1 - alpha) + minus_dm[i] * alpha

    pdi = np.where(atr > 0, 100 * pdi_raw / atr, 0.0)
    mdi = np.where(atr > 0, 100 * mdi_raw / atr, 0.0)
    denom = pdi + mdi
    dx = np.zeros(len(pdi))
    mask = denom > 0
    dx[mask] = 100 * np.abs(pdi[mask] - mdi[mask]) / denom[mask]

    adx_raw = np.zeros(len(dx)); adx_raw[0] = dx[0]
    for i in range(1, len(dx)):
        adx_raw[i] = adx_raw[i - 1] * (1 - alpha) + dx[i] * alpha

    # adx/di are offset by 1 relative to input (because tr/dm uses diffs)
    # Place them starting at index 1
    for i in range(1, min(n, len(adx_raw) + 1)):
        pdi_vals[i] = float(pdi[i - 1]) if not np.isnan(pdi[i - 1]) else np.nan
        mdi_vals[i] = float(mdi[i - 1]) if not np.isnan(mdi[i - 1]) else np.nan
        adx_vals[i] = float(adx_raw[i - 1]) if not np.isnan(adx_raw[i - 1]) else np.nan

    return (
        pd.Series(adx_vals, index=high.index, dtype=float),
        pd.Series(pdi_vals, index=high.index, dtype=float),
        pd.Series(mdi_vals, index=high.index, dtype=float),
    )


# ── Divergence series (sliding-window) ─────────────────────────────────

def _find_swings(data: np.ndarray, window: int = 3):
    """Find local peaks and troughs in a 1-d array."""
    peaks, troughs = set(), set()
    n = len(data)
    for i in range(window, n - window):
        left = data[i - window : i]
        right = data[i + 1 : i + window + 1]
        if data[i] >= max(left) and data[i] > max(right):
            peaks.add(i)
        if data[i] <= min(left) and data[i] < min(right):
            troughs.add(i)
    return peaks, troughs


def _detect_divergence_at(
    close_arr: np.ndarray,
    rsi_arr: np.ndarray,
    macd_arr: np.ndarray,
    window: int = 3,
) -> str:
    """Detect divergence in the tail of the given arrays (>=
    20 points)."""
    n = len(close_arr)
    if n < 20:
        return ""

    tail = min(30, n)
    c_tail = close_arr[-tail:]
    r_tail = rsi_arr[-tail:]
    m_tail = macd_arr[-tail:]

    peaks, troughs = _find_swings(c_tail, window)
    _, r_peaks = _find_swings(r_tail, window)
    r_inv = -r_tail
    _, r_troughs = _find_swings(r_inv, window)
    _, m_peaks = _find_swings(m_tail, window)
    m_inv = -m_tail
    _, m_troughs = _find_swings(m_inv, window)

    # Top divergence
    sorted_peaks = sorted(peaks)
    if len(sorted_peaks) >= 2:
        i1, i2 = sorted_peaks[-2], sorted_peaks[-1]
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
    sorted_troughs = sorted(troughs)
    if len(sorted_troughs) >= 2:
        i1, i2 = sorted_troughs[-2], sorted_troughs[-1]
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


def _compute_divergence_series(
    close: pd.Series,
    rsi_s: pd.Series,
    macd_s: pd.Series,
    window: int = 3,
) -> pd.Series:
    """Compute divergence for each row using a sliding window.

    For each position i (i >= 19), we look at rows [i-29, i] (up to 30 rows)
    and detect divergence. Earlier rows get "".
    """
    c_arr = close.values
    r_arr = rsi_s.values
    m_arr = macd_s.values
    n = len(c_arr)

    results = [""] * n
    for i in range(19, n):
        start = max(0, i - 29)
        results[i] = _detect_divergence_at(
            c_arr[start : i + 1],
            r_arr[start : i + 1],
            m_arr[start : i + 1],
            window=window,
        )
    return pd.Series(results, index=close.index, dtype=str)


def _compute_macd_cross_series(macd_hist: pd.Series) -> pd.Series:
    """Detect MACD golden/death cross at each bar using last 5 bars."""
    results = [""] * len(macd_hist)
    h = macd_hist.values
    for i in range(4, len(h)):
        lookback = h[i - 4 : i + 1]
        for j in range(1, min(4, len(lookback))):
            if lookback[j - 1] <= 0 < lookback[j]:
                results[i] = "金叉"
                break
            elif lookback[j - 1] >= 0 > lookback[j]:
                results[i] = "死叉"
                break
    return pd.Series(results, index=macd_hist.index, dtype=str)


# ── Main entry point ───────────────────────────────────────────────────

def compute_indicator_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """Compute all technical indicators as time series columns.

    Args:
        df: DataFrame with columns [date, open, close, high, low, volume].
            Must be sorted ascending by date.

    Returns:
        DataFrame indexed by date with columns:
            open, high, low, close, volume,
            ma5, ma10, ma20,
            macd, macd_signal, macd_hist,
            rsi,
            bb_upper, bb_middle, bb_lower, bb_width_pct,
            k, d, j,
            vol_ratio, vol_trend,
            adx, plus_di, minus_di,
            divergence, macd_cross

        Returns None if fewer than MIN_BARS rows.
    """
    if len(df) < MIN_BARS:
        return None

    # Drop NaNs in price columns but keep positional alignment via raw numpy arrays
    close_arr = pd.to_numeric(df["close"], errors="coerce").values
    high_arr = pd.to_numeric(df["high"], errors="coerce").values
    low_arr = pd.to_numeric(df["low"], errors="coerce").values
    volume_arr = pd.to_numeric(df["volume"], errors="coerce").values
    date_arr = df["date"].values
    n = len(df)

    # Use Series with consistent RangeIndex for rolling computations
    idx = pd.RangeIndex(n)
    close = pd.Series(close_arr, index=idx)
    high = pd.Series(high_arr, index=idx)
    low = pd.Series(low_arr, index=idx)
    volume = pd.Series(volume_arr, index=idx)

    valid_mask = ~pd.isna(close)

    # -- MA --
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()

    # -- MACD --
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # -- RSI(14) --
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # -- Bollinger Bands (20, 2) --
    bb_middle = ma20
    bb_sigma = close.rolling(20).std()
    bb_upper = bb_middle + 2 * bb_sigma
    bb_lower = bb_middle - 2 * bb_sigma
    bb_width_pct = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan) * 100.0

    # -- KDJ (9, 3, 3) —
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv_s = (close - low9) / (high9 - low9).replace(0, np.nan) * 100.0

    k_vals = [50.0] * 8
    d_vals = [50.0] * 8
    j_vals = [50.0] * 8
    for i in range(8, n):
        r = rsv_s.iloc[i]
        if pd.isna(r):
            k_vals.append(k_vals[-1])
            d_vals.append(d_vals[-1])
        else:
            k_vals.append(2 / 3 * k_vals[-1] + 1 / 3 * r)
            d_vals.append(2 / 3 * d_vals[-1] + 1 / 3 * k_vals[-1])
        j_vals.append(3 * k_vals[-1] - 2 * d_vals[-1])

    k_s = pd.Series(k_vals, index=idx, dtype=float)
    d_s = pd.Series(d_vals, index=idx, dtype=float)
    j_s = pd.Series(j_vals, index=idx, dtype=float)

    # -- Volume ratio --
    vol_avg20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg20.replace(0, np.nan)
    vol_trend = pd.Series([
        "放量" if pd.notna(v) and v > 1.2
        else ("缩量" if pd.notna(v) and v < 0.8 else "正常")
        for v in vol_ratio
    ], index=idx)

    # -- ADX --
    adx_s, pdi_s, mdi_s = _compute_adx_series(high, low, close)

    # -- Divergence --
    div_s = _compute_divergence_series(close, rsi, macd)

    # -- MACD cross --
    cross_s = _compute_macd_cross_series(macd_hist)

    # -- Assemble — index by date strings ──
    result = pd.DataFrame(
        {
            "open": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "close": df["close"].values,
            "volume": df["volume"].values,
            "ma5": ma5.values,
            "ma10": ma10.values,
            "ma20": ma20.values,
            "macd": macd.values,
            "macd_signal": macd_signal.values,
            "macd_hist": macd_hist.values,
            "rsi": rsi.values,
            "bb_upper": bb_upper.values,
            "bb_middle": bb_middle.values,
            "bb_lower": bb_lower.values,
            "bb_width_pct": bb_width_pct.values,
            "k": k_s.values,
            "d": d_s.values,
            "j": j_s.values,
            "vol_ratio": vol_ratio.values,
            "vol_trend": vol_trend.values,
            "adx": adx_s.values,
            "plus_di": pdi_s.values,
            "minus_di": mdi_s.values,
            "divergence": div_s.values,
            "macd_cross": cross_s.values,
        },
        index=date_arr,
    )
    result["divergence"] = result["divergence"].replace("", None)
    result["macd_cross"] = result["macd_cross"].replace("", None)

    return result


# ── Snapshot extraction ────────────────────────────────────────────────

def indicators_snapshot_at(
    ind_df: pd.DataFrame,
    row_idx: int,
) -> dict | None:
    """Extract a single indicator snapshot dict from the time series at
    the given row index.  Compatible with compute_indicators() output shape.
    """
    if ind_df is None or row_idx < 0 or row_idx >= len(ind_df):
        return None
    row = ind_df.iloc[row_idx]

    def _v(col: str):
        val = row.get(col)
        if pd.isna(val):
            return None
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.integer,)):
            return int(val)
        return val

    return {
        "ma5": _v("ma5"),
        "ma10": _v("ma10"),
        "ma20": _v("ma20"),
        "macd": _v("macd"),
        "macd_signal": _v("macd_signal"),
        "macd_hist": _v("macd_hist"),
        "rsi": _v("rsi"),
        "bb_upper": _v("bb_upper"),
        "bb_middle": _v("bb_middle"),
        "bb_lower": _v("bb_lower"),
        "bb_width_pct": _v("bb_width_pct"),
        "k": _v("k"),
        "d": _v("d"),
        "j": _v("j"),
        "vol_ratio": _v("vol_ratio"),
        "vol_trend": row.get("vol_trend") or "",
        "adx": _v("adx"),
        "plus_di": _v("plus_di"),
        "minus_di": _v("minus_di"),
        "divergence": row.get("divergence") or "",
        "macd_cross": row.get("macd_cross") or "",
    }


def last_snapshot(ind_df: pd.DataFrame) -> dict | None:
    """Return the indicator snapshot for the last row of the time series."""
    if ind_df is None or len(ind_df) == 0:
        return None
    return indicators_snapshot_at(ind_df, len(ind_df) - 1)
