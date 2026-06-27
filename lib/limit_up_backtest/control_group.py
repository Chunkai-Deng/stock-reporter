"""
Control group construction for weight optimization.

For each trade date in the backtest database, samples non-limit-up stocks
and computes their D-1 technical indicators.  The resulting positive +
negative samples form the training data for weight derivation.

Key design: outputs both binary feature vectors (for statistical methods)
AND full 90-day indicator time series (for neural network training).
"""

from __future__ import annotations

import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

# Reuse from parent module
from .schema import get_connection
from .fetcher import _get_kline_cached, _extract_window, clear_kline_cache
from .indicators import compute_indicator_series, last_snapshot

logger = logging.getLogger(__name__)

# ── Binary feature extraction ─────────────────────────────────────────
# Mirrors the 10 dimensions in cloud_stock_reporter.score_stock()


def extract_binary_features(
    indicators: dict,
    price: float,
    change_pct: float,
    weekly: dict | None = None,
) -> dict[str, float]:
    """Convert an indicator snapshot into the 10 binary score dimensions.

    Each dimension returns:
        +1 = bullish condition met
        -1 = bearish condition met
         0 = neutral / not triggered
    """
    feats: dict[str, float] = {}

    # D1: Price vs MA20
    ma20 = indicators.get("ma20")
    if price and ma20:
        feats["d1_price_above_ma20"] = 1.0 if price > ma20 else -1.0
    else:
        feats["d1_price_above_ma20"] = 0.0

    # D2: MA5 vs MA10
    ma5 = indicators.get("ma5")
    ma10 = indicators.get("ma10")
    if ma5 is not None and ma10 is not None:
        feats["d2_ma5_above_ma10"] = 1.0 if ma5 > ma10 else -1.0
    else:
        feats["d2_ma5_above_ma10"] = 0.0

    # D3: MACD cross
    macd_cross = indicators.get("macd_cross", "")
    if macd_cross == "金叉":
        feats["d3_macd_cross"] = 1.0
    elif macd_cross == "死叉":
        feats["d3_macd_cross"] = -1.0
    else:
        feats["d3_macd_cross"] = 0.0

    # D4: RSI oversold / overbought
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            feats["d4_rsi"] = 1.0
        elif rsi > 70:
            feats["d4_rsi"] = -1.0
        else:
            feats["d4_rsi"] = 0.0
    else:
        feats["d4_rsi"] = 0.0

    # D5: Bollinger position
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    if price and bb_upper and bb_lower:
        if price >= bb_upper * 0.99:
            feats["d5_bollinger"] = -1.0
        elif price <= bb_lower * 1.01:
            feats["d5_bollinger"] = 1.0
        else:
            feats["d5_bollinger"] = 0.0
    else:
        feats["d5_bollinger"] = 0.0

    # D6: Volume trend
    vol_trend = indicators.get("vol_trend", "")
    if vol_trend == "放量":
        feats["d6_volume"] = 1.0 if change_pct >= 0 else -1.0
    else:
        feats["d6_volume"] = 0.0

    # D7: KDJ-J extreme
    j_val = indicators.get("j")
    if j_val is not None:
        if j_val < 0:
            feats["d7_kdj"] = 1.0
        elif j_val > 100:
            feats["d7_kdj"] = -1.0
        else:
            feats["d7_kdj"] = 0.0
    else:
        feats["d7_kdj"] = 0.0

    # D8: ADX trend
    adx = indicators.get("adx")
    plus_di = indicators.get("plus_di")
    minus_di = indicators.get("minus_di")
    if adx is not None and plus_di is not None and minus_di is not None:
        if adx > 40:
            feats["d8_adx"] = 1.0 if plus_di > minus_di else -1.0
        else:
            feats["d8_adx"] = 0.0
    else:
        feats["d8_adx"] = 0.0

    # D9: Divergence
    divergence = indicators.get("divergence", "")
    if divergence == "底背离":
        feats["d9_divergence"] = 2.0  # double weight in original
    elif divergence == "顶背离":
        feats["d9_divergence"] = -2.0
    else:
        feats["d9_divergence"] = 0.0

    # D10: Weekly trend
    wt = (weekly or {}).get("weekly_trend", "")
    if wt == "上涨":
        feats["d10_weekly"] = 1.0
    elif wt == "下跌":
        feats["d10_weekly"] = -1.0
    else:
        feats["d10_weekly"] = 0.0

    return feats


FEATURE_NAMES = [
    "d1_price_above_ma20",
    "d2_ma5_above_ma10",
    "d3_macd_cross",
    "d4_rsi",
    "d5_bollinger",
    "d6_volume",
    "d7_kdj",
    "d8_adx",
    "d9_divergence",
    "d10_weekly",
]


# ── Positive sample extraction ────────────────────────────────────────


def extract_positive_samples(
    conn=None,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Extract D-1 binary features + full 90-day series for all limit-up events.

    Returns:
        (binary_samples, time_series_list)
        binary_samples: list of {feature_name: value, ..., "code": ..., "date": ..., "label": 1}
        time_series_list: list of DataFrames (90 rows × 27 indicator cols)
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection(readonly=True)

    try:
        cur = conn.execute("""
            SELECT e.event_id, e.code, e.name, e.trade_date,
                   e.change_pct, e.close_price
            FROM limit_up_events e
            ORDER BY e.trade_date
        """)
        events = cur.fetchall()

        binary_samples = []
        time_series_list = []

        for evt in events:
            # Fetch D-1 snapshot
            di_cur = conn.execute("""
                SELECT * FROM daily_indicators
                WHERE event_id = ? AND days_before = -1
            """, (evt["event_id"],))
            d1_row = di_cur.fetchone()
            if not d1_row:
                continue

            d1 = dict(d1_row)
            price = float(evt["close_price"] or 0)
            change_pct = float(evt["change_pct"] or 0)

            features = extract_binary_features(d1, price, change_pct)
            features["code"] = evt["code"]
            features["date"] = evt["trade_date"]
            features["label"] = 1  # positive = limit-up

            binary_samples.append(features)

            # Fetch full 90-day series for NN training
            ts_cur = conn.execute("""
                SELECT * FROM daily_indicators
                WHERE event_id = ?
                ORDER BY days_before ASC
            """, (evt["event_id"],))
            ts_rows = ts_cur.fetchall()
            if ts_rows:
                df = pd.DataFrame([dict(r) for r in ts_rows])
                df = df.set_index("days_before")
                df = df.drop(columns=["id", "event_id"], errors="ignore")
                time_series_list.append(df)

        logger.info(
            "Extracted %d positive samples, %d time series",
            len(binary_samples),
            len(time_series_list),
        )
        return binary_samples, time_series_list
    finally:
        if own_conn:
            conn.close()


# ── Control group construction ─────────────────────────────────────────


def get_trade_dates_from_db(conn=None) -> list[str]:
    """Get distinct trade dates from the backtest database."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection(readonly=True)
    try:
        cur = conn.execute(
            "SELECT DISTINCT trade_date FROM limit_up_events ORDER BY trade_date"
        )
        return [r["trade_date"] for r in cur.fetchall()]
    finally:
        if own_conn:
            conn.close()


def _sample_control_codes(
    date_str: str,
    exclude_codes: set[str],
    n_samples: int,
    industry_map: dict[str, str] | None = None,
) -> list[dict]:
    """Sample non-limit-up stocks for a given date via akshare.

    Returns list of {code, name, industry}.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not available; cannot build control group")
        return []

    try:
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        logger.warning("Failed to fetch A-share spot list: %s", e)
        # Try to use cached stock pool from pre_screener
        return _fallback_control_codes(exclude_codes, n_samples)

    if df is None or df.empty:
        return _fallback_control_codes(exclude_codes, n_samples)

    # Filter
    candidates = []
    for _, row in df.iterrows():
        code = str(row.get("代码", "")).strip()
        if not code or len(code) != 6:
            continue
        if code in exclude_codes:
            continue
        name = str(row.get("名称", ""))
        # Skip ST stocks
        if "ST" in name:
            continue
        candidates.append({
            "code": code,
            "name": name,
            "industry": str(row.get("行业", "")),
        })

    if len(candidates) < n_samples:
        return candidates

    # Stratified sampling by industry if industry_map available
    if industry_map:
        # Group by industry, sample proportionally
        by_industry: dict[str, list[dict]] = {}
        for c in candidates:
            ind = c.get("industry", "其他")
            by_industry.setdefault(ind, []).append(c)

        sampled = []
        per_industry = max(1, n_samples // max(len(by_industry), 1))
        for ind, stocks in by_industry.items():
            k = min(per_industry, len(stocks))
            sampled.extend(random.sample(stocks, k))
        # Fill remaining with random
        remaining = n_samples - len(sampled)
        if remaining > 0 and len(candidates) > len(sampled):
            already = {s["code"] for s in sampled}
            remaining_pool = [c for c in candidates if c["code"] not in already]
            sampled.extend(
                random.sample(remaining_pool, min(remaining, len(remaining_pool)))
            )
        return sampled[:n_samples]

    return random.sample(candidates, min(n_samples, len(candidates)))


def _fallback_control_codes(
    exclude_codes: set[str],
    n_samples: int,
) -> list[dict]:
    """Fallback: sample from pre_screened cache or generate random codes."""
    import glob
    import json
    import os

    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data"
    )
    pattern = os.path.join(data_dir, "pre_screened_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    candidates = []
    seen = set()

    for f in files:
        try:
            with open(f) as fh:
                pool = json.load(fh)
            for item in pool:
                code = item.get("code", "")
                if code and code not in exclude_codes and code not in seen:
                    seen.add(code)
                    candidates.append({
                        "code": code,
                        "name": item.get("name", ""),
                        "industry": item.get("industry", ""),
                    })
        except Exception:
            continue

    if len(candidates) >= n_samples:
        return random.sample(candidates, n_samples)
    return candidates


def build_control_group(
    trade_dates: list[str],
    positive_codes_by_date: dict[str, set[str]],
    control_ratio: int = 5,
    max_workers: int = 4,
    rate_delay: float = 0.5,
    progress_callback=None,
) -> tuple[list[dict], list[pd.DataFrame], list[dict]]:
    """Build control group for all trade dates.

    For each date:
      1. Sample non-limit-up stocks
      2. Fetch K-line → compute indicators
      3. Extract D-1 binary features + full 90-day time series

    Returns:
        (binary_samples, time_series_list, errors)
    """
    binary_samples: list[dict] = []
    time_series_list: list[pd.DataFrame] = []
    errors: list[dict] = []

    # ── Phase 1: Sample control codes per date ─────────────────────
    date_controls: dict[str, list[dict]] = {}
    for d in trade_dates:
        exclude = positive_codes_by_date.get(d, set())
        n = min(control_ratio * max(1, len(exclude)), 300)
        if progress_callback:
            progress_callback(f"Sampling control group: {d} (n={n})")
        controls = _sample_control_codes(d, exclude, n)
        date_controls[d] = controls
        logger.info("  %s: %d control stocks sampled", d, len(controls))

    # ── Phase 2: Fetch K-line for all unique control codes ─────────
    all_control_codes: dict[str, str] = {}  # code → latest date needed
    for d, ctrls in date_controls.items():
        for c in ctrls:
            code = c["code"]
            if code not in all_control_codes or d > all_control_codes[code]:
                all_control_codes[code] = d

    logger.info(
        "%d unique control stocks to fetch K-line for",
        len(all_control_codes),
    )

    code_kline_map: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, str] = {}

    def _fetch_one(code: str):
        try:
            sys.path.insert(0, "")
            from cloud_stock_reporter import fetch_weekly_trend
        except ImportError:
            fetch_weekly_trend = None

        try:
            df = _get_kline_cached(code, lookback_days=120, rate_delay=rate_delay)
            if df is not None:
                code_kline_map[code] = df
            else:
                fetch_errors[code] = "Insufficient K-line"
        except Exception as e:
            fetch_errors[code] = str(e)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, c): c for c in all_control_codes}
        for i, fut in enumerate(as_completed(futures)):
            code = futures[fut]
            try:
                fut.result()
            except Exception as e:
                fetch_errors[code] = str(e)
            if progress_callback and (i + 1) % 30 == 0:
                progress_callback(
                    f"Control K-line: {i + 1}/{len(all_control_codes)}"
                )

    logger.info(
        "Control K-line: %d success, %d failed",
        len(code_kline_map),
        len(fetch_errors),
    )

    # ── Phase 3: Extract features per control stock ────────────────
    for d, ctrls in date_controls.items():
        for c in ctrls:
            code = c["code"]
            if code not in code_kline_map:
                errors.append({"date": d, "code": code, "error": "No K-line"})
                continue

            kline = code_kline_map[code]
            window = _extract_window(kline, d, 90)
            if window is None or len(window) < 26:
                errors.append({"date": d, "code": code, "error": "K-line too short"})
                continue

            ind_df = compute_indicator_series(window)
            if ind_df is None:
                errors.append({"date": d, "code": code, "error": "Indicator compute failed"})
                continue

            snapshot = last_snapshot(ind_df)
            if snapshot is None:
                errors.append({"date": d, "code": code, "error": "No snapshot"})
                continue

            price = float(window.iloc[-1]["close"])
            change_pct = 0.0
            if len(window) >= 2:
                prev_close = float(window.iloc[-2]["close"])
                if prev_close > 0:
                    change_pct = (price - prev_close) / prev_close * 100

            # Try weekly trend (best-effort)
            weekly = {}
            try:
                from cloud_stock_reporter import code_prefix, fetch_weekly_trend
                symbol = f"{code_prefix(code)}{code}"
                weekly = fetch_weekly_trend(symbol)
            except Exception:
                pass

            features = extract_binary_features(snapshot, price, change_pct, weekly)
            features["code"] = code
            features["date"] = d
            features["label"] = 0  # negative = not limit-up
            features["name"] = c.get("name", "")

            binary_samples.append(features)

            # Full time series for NN
            time_series_list.append(ind_df)

    logger.info(
        "Control group built: %d binary samples, %d time series, %d errors",
        len(binary_samples),
        len(time_series_list),
        len(errors),
    )
    return binary_samples, time_series_list, errors


# ── Dataset assembly ───────────────────────────────────────────────────


def build_training_dataset(
    control_ratio: int = 5,
    max_workers: int = 4,
    rate_delay: float = 0.5,
    progress_callback=None,
) -> dict:
    """Build the complete training dataset: positives + controls.

    Returns dict with keys:
        binary_samples: list[dict]  — positive + negative, with 'label' field
        time_series: list[pd.DataFrame]  — 90-day indicator DataFrames
        feature_matrix: np.ndarray  — (N, 10) binary feature matrix
        labels: np.ndarray  — (N,) binary labels
    """
    # Positive samples
    if progress_callback:
        progress_callback("Extracting positive samples...")
    pos_binary, pos_ts = extract_positive_samples()

    # Get trade dates and positive code sets
    trade_dates = sorted(set(s["date"] for s in pos_binary))
    pos_codes_by_date: dict[str, set[str]] = {}
    for s in pos_binary:
        pos_codes_by_date.setdefault(s["date"], set()).add(s["code"])

    if progress_callback:
        progress_callback(f"Building control group for {len(trade_dates)} dates...")

    # Control group
    neg_binary, neg_ts, errors = build_control_group(
        trade_dates=trade_dates,
        positive_codes_by_date=pos_codes_by_date,
        control_ratio=control_ratio,
        max_workers=max_workers,
        rate_delay=rate_delay,
        progress_callback=progress_callback,
    )

    # Combine
    all_binary = pos_binary + neg_binary
    all_ts = pos_ts + neg_ts

    # Build feature matrix
    feature_matrix = np.array([
        [s.get(fn, 0.0) for fn in FEATURE_NAMES]
        for s in all_binary
    ], dtype=float)

    labels = np.array([s["label"] for s in all_binary], dtype=int)

    logger.info(
        "Training dataset: %d samples (%d pos, %d neg), feature matrix %s",
        len(all_binary),
        len(pos_binary),
        len(neg_binary),
        feature_matrix.shape,
    )

    return {
        "binary_samples": all_binary,
        "time_series": all_ts,
        "feature_matrix": feature_matrix,
        "labels": labels,
        "pos_count": len(pos_binary),
        "neg_count": len(neg_binary),
        "errors": errors,
    }
