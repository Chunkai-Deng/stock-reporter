"""
Statistical weight optimization engine for technical indicator scoring.

Methods:
  1. Lift analysis — P(feature|limit_up) / P(feature|control)
  2. Logistic regression — coefficients as weights
  3. Mutual information — feature importance ranking
  4. Blending — weighted combination of all methods + NN ensemble importance

Output: a weight_config dict ready for JSON serialization.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

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

DIMENSION_META = {
    "d1_price_above_ma20": {
        "description": "价格 > MA20",
        "positive_condition": "price > ma20",
        "negative_condition": "price <= ma20",
        "direction": "binary",
    },
    "d2_ma5_above_ma10": {
        "description": "MA5 > MA10",
        "positive_condition": "ma5 > ma10",
        "negative_condition": "ma5 <= ma10",
        "direction": "binary",
    },
    "d3_macd_cross": {
        "description": "MACD金叉/死叉",
        "positive_condition": "macd_cross == '金叉'",
        "negative_condition": "macd_cross == '死叉'",
        "direction": "trinary",
    },
    "d4_rsi": {
        "description": "RSI超卖/超买",
        "positive_condition": "rsi < 30",
        "negative_condition": "rsi > 70",
        "direction": "dual_threshold",
    },
    "d5_bollinger": {
        "description": "布林带位置",
        "positive_condition": "price <= bb_lower * 1.01",
        "negative_condition": "price >= bb_upper * 0.99",
        "direction": "dual_threshold",
    },
    "d6_volume": {
        "description": "成交量趋势",
        "positive_condition": "vol_trend == '放量' AND change_pct >= 0",
        "negative_condition": "vol_trend == '放量' AND change_pct < 0",
        "direction": "binary",
    },
    "d7_kdj": {
        "description": "KDJ极端值",
        "positive_condition": "j < 0",
        "negative_condition": "j > 100",
        "direction": "dual_threshold",
    },
    "d8_adx": {
        "description": "ADX趋势强度",
        "positive_condition": "adx > 40 AND plus_di > minus_di",
        "negative_condition": "adx > 40 AND plus_di <= minus_di",
        "direction": "conditional",
    },
    "d9_divergence": {
        "description": "背离信号",
        "positive_condition": "divergence == '底背离'",
        "negative_condition": "divergence == '顶背离'",
        "direction": "trinary",
    },
    "d10_weekly": {
        "description": "周线趋势",
        "positive_condition": "weekly_trend == '上涨'",
        "negative_condition": "weekly_trend == '下跌'",
        "direction": "binary",
    },
}


# ── 1. Lift Analysis ───────────────────────────────────────────────────


def run_lift_analysis(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.01,
    laplace_smoothing: bool = True,
) -> dict:
    """Compute lift ratio for each feature dimension.

    lift_i = P(feature_i > 0 | limit_up) / P(feature_i > 0 | control)

    Returns dict with per-dimension lift metrics.
    """
    pos_mask = y == 1
    neg_mask = y == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    if n_pos == 0 or n_neg == 0:
        logger.warning("Need both positive and negative samples for lift analysis")
        return {}

    results = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]

        # Positive direction (bullish condition met)
        pos_bullish = (col > 0) & pos_mask
        neg_bullish = (col > 0) & neg_mask

        p_pos = pos_bullish.sum() / n_pos
        p_neg = neg_bullish.sum() / n_neg

        if laplace_smoothing:
            p_pos = (pos_bullish.sum() + 1) / (n_pos + 2)
            p_neg = (neg_bullish.sum() + 1) / (n_neg + 2)

        lift = p_pos / max(p_neg, epsilon)
        log_lift = math.log(max(lift, 0.001))

        # Negative direction (bearish condition met)
        pos_bearish = (col < 0) & pos_mask
        neg_bearish = (col < 0) & neg_mask

        p_pos_neg = pos_bearish.sum() / n_pos
        p_neg_neg = neg_bearish.sum() / n_neg

        if laplace_smoothing:
            p_pos_neg = (pos_bearish.sum() + 1) / (n_pos + 2)
            p_neg_neg = (neg_bearish.sum() + 1) / (n_neg + 2)

        lift_neg = p_pos_neg / max(p_neg_neg, epsilon)
        log_lift_neg = -math.log(max(lift_neg, 0.001))

        # Combined weight: bullish lift minus bearish lift
        raw_weight = log_lift + log_lift_neg

        results[name] = {
            "p_pos_bullish": round(p_pos, 4),
            "p_neg_bullish": round(p_neg, 4),
            "lift_bullish": round(lift, 4),
            "p_pos_bearish": round(p_pos_neg, 4),
            "p_neg_bearish": round(p_neg_neg, 4),
            "lift_bearish": round(lift_neg, 4),
            "raw_weight": round(raw_weight, 4),
            "zero_variance": (pos_bullish.sum() == 0 and pos_bearish.sum() == 0),
        }

    return results


# ── 2. Logistic Regression ─────────────────────────────────────────────


def run_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
) -> dict | None:
    """Run L2-regularized logistic regression.

    Returns per-dimension coefficients as weights.
    Falls back to numpy gradient descent if sklearn unavailable.
    """
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )
        model.fit(X, y)

        results = {}
        for i, name in enumerate(FEATURE_NAMES):
            results[name] = {
                "coefficient": round(float(model.coef_[0][i]), 4),
                "intercept": round(float(model.intercept_[0]), 4),
            }

        logger.info(
            "Logistic regression: intercept=%.4f, coefs=%s",
            results[FEATURE_NAMES[0]]["intercept"],
            [results[n]["coefficient"] for n in FEATURE_NAMES[:5]] + ["..."],
        )
        return results
    except ImportError:
        logger.warning("sklearn not available; using numpy gradient descent")
        return _logistic_regression_numpy(X, y)
    except Exception as e:
        logger.warning("Logistic regression failed: %s", e)
        return _logistic_regression_numpy(X, y)


def _logistic_regression_numpy(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
    C: float = 1.0,
) -> dict:
    """Numpy implementation of L2-regularized logistic regression."""
    n, d = X.shape
    # Add intercept
    X_aug = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)

    for epoch in range(epochs):
        z = X_aug @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        grad = (X_aug.T @ (p - y)) / n
        grad[:d] += (1.0 / (C * n)) * w[:d]  # L2 penalty
        w -= lr * grad

        if epoch % 200 == 0:
            loss = -np.mean(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
            logger.debug("  epoch %d: loss=%.4f", epoch, loss)

    results = {}
    for i, name in enumerate(FEATURE_NAMES):
        results[name] = {
            "coefficient": round(float(w[i]), 4),
            "intercept": round(float(w[-1]), 4),
        }
    return results


# ── 3. Mutual Information ──────────────────────────────────────────────


def run_mutual_information(X: np.ndarray, y: np.ndarray) -> dict:
    """Compute mutual information between each feature and the label.

    Uses a simple binning approach (discretizing continuous values).
    Falls back to sklearn if available.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(X, y, random_state=42)
        results = {}
        for i, name in enumerate(FEATURE_NAMES):
            results[name] = round(float(mi[i]), 4)
        return results
    except ImportError:
        return _mutual_information_numpy(X, y)


def _mutual_information_numpy(X: np.ndarray, y: np.ndarray, n_bins: int = 5) -> dict:
    """Numpy mutual information using simple discretization."""
    results = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = X[:, i]
        # Discretize
        bins = np.percentile(col, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            results[name] = 0.0
            continue
        x_disc = np.digitize(col, bins[:-1])

        # Compute MI via contingency table
        mi = 0.0
        for x_val in np.unique(x_disc):
            for y_val in [0, 1]:
                p_xy = np.sum((x_disc == x_val) & (y == y_val)) / len(y)
                p_x = np.sum(x_disc == x_val) / len(y)
                p_y = np.sum(y == y_val) / len(y)
                if p_xy > 0:
                    mi += p_xy * math.log(p_xy / (p_x * p_y))

        results[name] = round(max(0.0, mi), 4)

    return results


# ── 4. Blending ────────────────────────────────────────────────────────


def blend_weights(
    lift_result: dict,
    logreg_result: dict | None,
    mi_result: dict,
    nn_importance: list[float] | None = None,
    blend_config: dict | None = None,
) -> list[float]:
    """Blend weights from multiple methods into final per-dimension weights.

    Args:
        lift_result: from run_lift_analysis()
        logreg_result: from run_logistic_regression()
        mi_result: from run_mutual_information()
        nn_importance: 10-element list from NN ensemble
        blend_config: dict with weights for each method

    Returns:
        list of 10 final weights
    """
    if blend_config is None:
        has_nn = nn_importance is not None and sum(nn_importance) > 0
        blend_config = {
            "lift": 0.4 if has_nn else 0.5,
            "logistic": 0.15 if has_nn else 0.25,
            "mutual_info": 0.1 if has_nn else 0.25,
            "nn": 0.35 if has_nn else 0.0,
        }

    # Normalize each method's output to a common scale
    lift_weights = _normalize_weights(
        [lift_result.get(n, {}).get("raw_weight", 0.0) for n in FEATURE_NAMES],
        target_range=(-2.0, 2.0),
    )

    if logreg_result:
        logreg_weights = _normalize_weights(
            [logreg_result.get(n, {}).get("coefficient", 0.0) for n in FEATURE_NAMES],
            target_range=(-2.0, 2.0),
        )
    else:
        logreg_weights = [0.0] * 10

    mi_weights = _normalize_weights(
        [mi_result.get(n, 0.0) for n in FEATURE_NAMES],
        target_range=(0.5, 2.0),  # MI is always non-negative
    )

    if nn_importance and len(nn_importance) == 10 and sum(nn_importance) > 0:
        nn_weights = _normalize_weights(nn_importance, target_range=(-2.0, 2.0))
    else:
        nn_weights = [0.0] * 10

    # Blend
    final = np.zeros(10)
    final += blend_config["lift"] * np.array(lift_weights)
    final += blend_config["logistic"] * np.array(logreg_weights)
    final += blend_config["mutual_info"] * np.array(mi_weights)
    final += blend_config.get("nn", 0.0) * np.array(nn_weights)

    return [round(float(w), 2) for w in final]


def _normalize_weights(
    raw: list[float],
    target_range: tuple = (-2.0, 2.0),
) -> list[float]:
    """Normalize weights to a target range, preserving sign."""
    arr = np.array(raw)
    lo, hi = target_range

    # Handle zero array
    if np.all(arr == 0):
        return [0.0] * len(raw)

    # Scale magnitude while preserving sign
    abs_max = np.abs(arr).max()
    if abs_max > 0:
        arr = arr / abs_max * hi

    return [round(float(w), 2) for w in arr]


# ── 5. Main entry point ────────────────────────────────────────────────


def optimize_weights(
    X: np.ndarray,
    y: np.ndarray,
    nn_importance: list[float] | None = None,
    method: str = "blended",
) -> dict:
    """Run full weight optimization pipeline.

    Args:
        X: (N, 10) binary feature matrix
        y: (N,) binary labels
        nn_importance: optional 10-element NN importance vector
        method: "lift_only", "logistic_only", "blended"

    Returns:
        Complete weight_config dict
    """
    if len(X) < 10:
        logger.warning("Insufficient samples (%d) for weight optimization", len(X))
        return _default_weight_config()

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    logger.info("Optimizing weights: %d pos, %d neg samples", n_pos, n_neg)

    # Run methods
    lift_result = run_lift_analysis(X, y)

    logreg_result = None
    if method in ("logistic_only", "blended"):
        logreg_result = run_logistic_regression(X, y)

    mi_result = run_mutual_information(X, y)

    # Blend
    if method == "lift_only":
        raw_weights = [
            lift_result.get(n, {}).get("raw_weight", 0.0)
            for n in FEATURE_NAMES
        ]
        final_weights = _normalize_weights(raw_weights, target_range=(-2.0, 2.0))
    elif method == "logistic_only":
        if logreg_result:
            final_weights = _normalize_weights(
                [logreg_result.get(n, {}).get("coefficient", 0.0) for n in FEATURE_NAMES],
                target_range=(-2.0, 2.0),
            )
        else:
            final_weights = [1.0] * 10
    else:
        final_weights = blend_weights(lift_result, logreg_result, mi_result, nn_importance)

    # Build config
    dimensions = {}
    for i, name in enumerate(FEATURE_NAMES):
        meta = DIMENSION_META.get(name, {})
        lift_info = lift_result.get(name, {})

        dim = {
            "description": meta.get("description", name),
            "weight": final_weights[i],
            "direction": meta.get("direction", "binary"),
            "positive_condition": meta.get("positive_condition", ""),
            "negative_condition": meta.get("negative_condition", ""),
            "lift_bullish": lift_info.get("lift_bullish"),
            "lift_bearish": lift_info.get("lift_bearish"),
            "mi_score": mi_result.get(name),
            "confidence": _compute_confidence(lift_info, mi_result.get(name, 0), final_weights[i]),
            "feature_importance_rank": i + 1,  # placeholder, set after sorting
        }

        # Warnings
        warnings = []
        if lift_info.get("zero_variance"):
            warnings.append("零方差特征：该维度在涨停样本中从未触发")
        if dim["confidence"] < 0.5:
            warnings.append("低置信度：建议用连续指标替代二值阈值")
        if warnings:
            dim["warning"] = "; ".join(warnings)

        dimensions[name] = dim

    # Sort by |weight| descending and assign ranks
    sorted_dims = sorted(
        dimensions.items(),
        key=lambda x: abs(x[1]["weight"]),
        reverse=True,
    )
    for rank, (name, _) in enumerate(sorted_dims, 1):
        dimensions[name]["feature_importance_rank"] = rank

    config = {
        "version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "method": method,
        "sample_info": {
            "positive_samples": int(n_pos),
            "control_samples": int(n_neg),
        },
        "dimensions": dimensions,
        "thresholds": {
            "strong_buy": 5.0,
            "mild_buy": 3.0,
            "mild_sell": -3.0,
            "strong_sell": -5.0,
        },
    }

    return config


def _compute_confidence(
    lift_info: dict,
    mi_score: float,
    weight: float,
) -> float:
    """Compute confidence score (0-1) for a dimension's weight."""
    confidence = 0.5  # baseline

    # Higher lift → higher confidence
    lift_bull = lift_info.get("lift_bullish", 1.0) or 1.0
    lift_bear = lift_info.get("lift_bearish", 1.0) or 1.0
    max_lift = max(lift_bull, lift_bear)
    if max_lift > 1.5:
        confidence += 0.2
    elif max_lift > 1.2:
        confidence += 0.1

    # Higher MI → higher confidence
    if mi_score and mi_score > 0.05:
        confidence += 0.15
    if mi_score and mi_score > 0.1:
        confidence += 0.1

    # Non-zero weight → base confidence
    if abs(weight) > 0.5:
        confidence += 0.05

    # Zero variance → very low confidence
    if lift_info.get("zero_variance"):
        confidence = 0.1

    return round(min(1.0, max(0.0, confidence)), 2)


def _default_weight_config() -> dict:
    """Return a default equal-weight config."""
    dimensions = {}
    for name in FEATURE_NAMES:
        meta = DIMENSION_META.get(name, {})
        dimensions[name] = {
            "description": meta.get("description", name),
            "weight": 1.0,
            "confidence": 0.5,
            "direction": meta.get("direction", "binary"),
        }
    return {
        "version": 1,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "method": "default",
        "sample_info": {"positive_samples": 0, "control_samples": 0},
        "dimensions": dimensions,
        "thresholds": {
            "strong_buy": 5.0,
            "mild_buy": 3.0,
            "mild_sell": -3.0,
            "strong_sell": -5.0,
        },
    }
