"""
Limit-Up Backtest — 历史涨停股技术面回溯分析。

Provides:
- Standalone CLI (limit_up_backtest.py)
- Query API for downstream consumers (screening, AI analysis, reports)
- Weight optimization (statistical + neural network)
"""

# ── Event queries ──────────────────────────────────────────────────────
from .queries import (
    query_events,
    get_event_detail,
    get_events_for_stock,
)

# ── Indicator queries ──────────────────────────────────────────────────
from .queries import (
    get_pre_limit_up_series,
    get_indicator_snapshot,
)

# ── Aggregate statistics ───────────────────────────────────────────────
from .queries import (
    get_pattern_distribution,
    get_industry_stats,
    get_indicator_averages,
)

# ── Similarity ─────────────────────────────────────────────────────────
from .queries import (
    find_similar_pre_limit_up,
)

# ── Weight optimization ────────────────────────────────────────────────
from .control_group import (
    build_training_dataset,
    extract_positive_samples,
    extract_binary_features,
    FEATURE_NAMES,
)
from .weights import (
    optimize_weights,
    run_lift_analysis,
    run_logistic_regression,
    run_mutual_information,
)
from .nn_predictor import (
    train_all_models,
    importance_to_weights,
)

# ── Raw access ─────────────────────────────────────────────────────────
from .schema import get_connection, get_active_weight_config

__all__ = [
    # Event queries
    "query_events",
    "get_event_detail",
    "get_events_for_stock",
    # Indicator queries
    "get_pre_limit_up_series",
    "get_indicator_snapshot",
    # Stats
    "get_pattern_distribution",
    "get_industry_stats",
    "get_indicator_averages",
    # Similarity
    "find_similar_pre_limit_up",
    # Weight optimization
    "build_training_dataset",
    "extract_positive_samples",
    "extract_binary_features",
    "optimize_weights",
    "run_lift_analysis",
    "run_logistic_regression",
    "run_mutual_information",
    "train_all_models",
    "importance_to_weights",
    "FEATURE_NAMES",
    # Raw
    "get_connection",
    "get_active_weight_config",
]
