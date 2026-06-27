"""
Statistical report generation and weight config persistence.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def save_weight_config(
    config: dict,
    output_path: str = "",
) -> str:
    """Save weight configuration to JSON file.

    Args:
        config: Complete weight_config dict
        output_path: Target path. Defaults to data/limit_up_backtest/weight_config.json

    Returns:
        Absolute path to the saved file
    """
    if not output_path:
        project_root = os.path.join(
            os.path.dirname(__file__), "..", ".."
        )
        output_path = os.path.join(
            project_root, "data", "limit_up_backtest", "weight_config.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info("Weight config saved to %s", output_path)
    return output_path


def generate_report(
    weight_config: dict,
    nn_results: dict | None = None,
    lift_detail: dict | None = None,
) -> str:
    """Generate a multi-section text report explaining the weight derivation.

    Args:
        weight_config: from weights.optimize_weights()
        nn_results: from nn_predictor.train_all_models()
        lift_detail: from weights.run_lift_analysis() for per-dim detail

    Returns:
        Multi-line text report string
    """
    lines = []
    _section = lambda title: lines.extend(["", "=" * 60, f"  {title}", "=" * 60, ""])

    info = weight_config.get("sample_info", {})
    dims = weight_config.get("dimensions", {})
    thresholds = weight_config.get("thresholds", {})

    # ── Section 1: Data Summary ────────────────────────────────────
    _section("1. Data Summary")
    lines.append(f"  Method: {weight_config.get('method', 'N/A')}")
    lines.append(f"  Generated: {weight_config.get('generated_at', 'N/A')}")
    lines.append(f"  Positive samples (limit-up): {info.get('positive_samples', 0)}")
    lines.append(f"  Control samples (non-limit-up): {info.get('control_samples', 0)}")

    pos = info.get("positive_samples", 0)
    neg = info.get("control_samples", 0)
    if pos and neg:
        lines.append(f"  Ratio: 1:{neg // max(pos, 1)} (pos:neg)")

    # ── Section 2: Dimension Weights ───────────────────────────────
    _section("2. Optimized Dimension Weights")

    sorted_dims = sorted(
        dims.items(),
        key=lambda x: abs(x[1].get("weight", 0)),
        reverse=True,
    )
    lines.append(
        f"  {'Rank':<5} {'Dimension':<25} {'Weight':>8} {'Confidence':>10} "
        f"{'Lift(Bull)':>12} {'Lift(Bear)':>12} {'MI':>8}"
    )
    lines.append("  " + "-" * 85)

    for name, d in sorted_dims:
        lines.append(
            f"  {d.get('feature_importance_rank', '-'):<5} "
            f"{name:<25} "
            f"{d.get('weight', 0):>+8.2f} "
            f"{d.get('confidence', 0):>10.2f} "
            f"{d.get('lift_bullish') or '-':>12} "
            f"{d.get('lift_bearish') or '-':>12} "
            f"{d.get('mi_score') or 0:>8.4f}"
        )
        w = d.get("warning", "")
        if w:
            lines.append(f"         ⚠ {w}")

    # ── Section 3: Thresholds ──────────────────────────────────────
    _section("3. Signal Thresholds")
    lines.append(f"  Strong Buy : >= {thresholds.get('strong_buy', 5.0)}")
    lines.append(f"  Mild Buy   : >= {thresholds.get('mild_buy', 3.0)}")
    lines.append(f"  Mild Sell  : <= {thresholds.get('mild_sell', -3.0)}")
    lines.append(f"  Strong Sell: <= {thresholds.get('strong_sell', -5.0)}")

    # ── Section 4: NN Results ──────────────────────────────────────
    if nn_results:
        _section("4. Neural Network Results")
        models = nn_results.get("models", {})
        available = nn_results.get("available_models", [])

        lines.append(f"  Available models: {', '.join(available) if available else 'none'}")
        lines.append("")

        for name, m in models.items():
            if m.get("auc") is not None:
                lines.append(f"  {name.upper():<15} AUC: {m['auc']:.4f}")
            else:
                lines.append(f"  {name.upper():<15} (not trained / unavailable)")

        ens_imp = nn_results.get("ensemble_importance")
        if ens_imp and sum(ens_imp) > 0:
            lines.append("")
            lines.append("  Ensemble importance (per dimension):")
            for i, (name, imp) in enumerate(zip(
                ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"],
                ens_imp,
            )):
                lines.append(f"    {name}: {imp:.4f}")

    # ── Section 5: Methodology ─────────────────────────────────────
    _section("5. Methodology Notes")
    lines.append("  Lift Analysis:")
    lines.append("    lift = P(feature|limit_up) / P(feature|control)")
    lines.append("    weight = ln(lift_bullish) - ln(lift_bearish)")
    lines.append("    Laplace smoothing applied for zero-count features")
    lines.append("")
    lines.append("  Logistic Regression:")
    lines.append("    L2-regularized, coefficients as weights")
    lines.append("    Falls back to numpy gradient descent if sklearn unavailable")
    lines.append("")
    lines.append("  Mutual Information:")
    lines.append("    Discretized via percentile binning")
    lines.append("    Non-negative; used for feature ranking")
    lines.append("")
    lines.append("  Neural Network Ensemble:")
    lines.append("    MLP + LSTM + CNN + Transformer")
    lines.append("    Permutation importance maps NN contribution to dimensions")
    lines.append("")
    lines.append("  Blending Formula:")
    if nn_results and nn_results.get("available_models"):
        lines.append("    weight = 0.40*lift + 0.15*logreg + 0.10*mi + 0.35*nn")
    else:
        lines.append("    weight = 0.50*lift + 0.25*logreg + 0.25*mi")

    # ── Section 6: Recommendations ─────────────────────────────────
    _section("6. Recommendations")

    high_confidence = [
        (name, d) for name, d in sorted_dims
        if d.get("confidence", 0) >= 0.7 and abs(d.get("weight", 0)) > 0.3
    ]
    low_confidence = [
        (name, d) for name, d in sorted_dims
        if d.get("confidence", 0) < 0.5
    ]

    if high_confidence:
        lines.append("  High-confidence dimensions (recommend higher weight):")
        for name, d in high_confidence:
            lines.append(
                f"    {d['description']}: weight={d['weight']:.2f}, "
                f"confidence={d['confidence']:.2f}"
            )

    if low_confidence:
        lines.append("")
        lines.append("  Low-confidence dimensions (consider adjusting or replacing):")
        for name, d in low_confidence:
            w = d.get("warning", "low statistical significance")
            lines.append(
                f"    {d['description']}: weight={d['weight']:.2f}, "
                f"confidence={d['confidence']:.2f} — {w}"
            )

    if not high_confidence and not low_confidence:
        lines.append(
            "  Insufficient data for strong recommendations. "
            "Collect more backtest data across a wider date range."
        )

    lines.append("")
    lines.append("  Next steps:")
    lines.append("    1. Apply weights via USE_OPTIMIZED_WEIGHTS=true")
    lines.append("    2. Monitor prediction accuracy over 1-2 weeks")
    lines.append("    3. Re-run --optimize-weights monthly as more data accumulates")
    lines.append("    4. Consider continuous indicators for low-confidence dimensions")

    return "\n".join(lines)


def save_report(report_text: str, output_path: str = "") -> str:
    """Save the text report to a file.

    Returns the file path.
    """
    if not output_path:
        project_root = os.path.join(
            os.path.dirname(__file__), "..", ".."
        )
        output_path = os.path.join(
            project_root,
            "data",
            "limit_up_backtest",
            f"weight_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Report saved to %s", output_path)
    return output_path
