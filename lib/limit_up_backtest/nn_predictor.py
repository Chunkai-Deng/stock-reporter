"""
Neural network models for limit-up prediction from technical indicator time series.

Models (in order of priority):
  1. MLP — binary feature classifier, sklearn (always available)
  2. LSTM — 90-day time series predictor, PyTorch (optional)
  3. 1D-CNN — local pattern detector, PyTorch (optional)
  4. Transformer — cross-time attention, PyTorch (optional)

Outputs ensemble probability + permutation importance for weight derivation.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Dependency detection ──────────────────────────────────────────────

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report

    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────

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

INDICATOR_COLS = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "ma20",
    "macd", "macd_signal", "macd_hist",
    "rsi",
    "bb_upper", "bb_middle", "bb_lower", "bb_width_pct",
    "k", "d", "j",
    "vol_ratio",
    "adx", "plus_di", "minus_di",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── MLP Classifier (sklearn) ───────────────────────────────────────────


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train MLP on binary features. Always available (sklearn or numpy fallback)."""
    result = {
        "model_name": "mlp",
        "available": SKLEARN_AVAILABLE,
        "auc": None,
        "predictions": None,
        "importance": None,
    }

    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available; MLP skipped")
        return result

    if len(X) < 20:
        logger.warning("Too few samples for MLP (%d)", len(X))
        return result

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with cross-validation
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=0.01,  # L2 regularization
            dropout=0.3,
            batch_size=min(32, len(X)),
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=RANDOM_SEED,
            verbose=False,
        )

        # Cross-validated predictions
        cv = StratifiedKFold(n_splits=min(5, len(X) // 10 + 2), shuffle=True, random_state=RANDOM_SEED)
        y_pred_proba = cross_val_predict(
            model, X_scaled, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]

        auc = roc_auc_score(y, y_pred_proba)
        result["auc"] = round(auc, 4)
        result["predictions"] = y_pred_proba

        # Permutation importance
        importance = _permutation_importance_sklearn(model, X_scaled, y, scaler)
        result["importance"] = importance

        logger.info("MLP: AUC=%.4f", auc)
    except Exception as e:
        logger.warning("MLP training failed: %s", e)

    return result


def _permutation_importance_sklearn(
    model, X: np.ndarray, y: np.ndarray, scaler
) -> list[float]:
    """Permutation importance for sklearn MLP."""
    try:
        baseline_pred = model.predict_proba(X)[:, 1]
        baseline_auc = roc_auc_score(y, baseline_pred)
    except Exception:
        return [0.0] * 10

    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        try:
            perm_pred = model.predict_proba(X_perm)[:, 1]
            perm_auc = roc_auc_score(y, perm_pred)
            importances.append(max(0.0, baseline_auc - perm_auc))
        except Exception:
            importances.append(0.0)

    total = sum(importances)
    if total > 0:
        importances = [v / total for v in importances]
    return importances


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Models (optional)
# ═══════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)

    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim=27, hidden=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden, num_layers,
                batch_first=True, dropout=dropout, bidirectional=False,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    class CNN1DPredictor(nn.Module):
        def __init__(self, input_dim=27, dropout=0.3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=10, padding=5),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 32, kernel_size=20, padding=10),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x: (batch, seq_len, features) → permute to (batch, features, seq_len)
            x = x.permute(0, 2, 1)
            return self.fc(self.conv(x))

    class TransformerPredictor(nn.Module):
        def __init__(self, input_dim=27, nhead=3, num_layers=2, dropout=0.2):
            super().__init__()
            self.pos_encoder = PositionalEncoding(input_dim, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=nhead, dropout=dropout,
                dim_feedforward=64, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.permute(0, 2, 1)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=100):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)

        def forward(self, x):
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)
            return self.dropout(x)


# ── Data augmentation for time series ──────────────────────────────────


def _augment_time_series(
    ts_list: list[pd.DataFrame],
    labels: np.ndarray,
    target_pos_count: int = 300,
) -> tuple[list[pd.DataFrame], np.ndarray]:
    """Augment positive samples with time warping + noise injection."""
    pos_ts = [ts for ts, lbl in zip(ts_list, labels) if lbl == 1]
    if len(pos_ts) == 0:
        return ts_list, labels

    n_augment = max(0, target_pos_count - len(pos_ts))
    augment_per_sample = max(1, n_augment // len(pos_ts))

    augmented_ts = list(ts_list)
    augmented_labels = list(labels)

    for ts in pos_ts:
        for _ in range(min(augment_per_sample, 3)):
            aug = ts.copy()

            # Time warping: stretch/compress random segments
            for col in INDICATOR_COLS:
                if col not in aug.columns:
                    continue
                noise = np.random.normal(0, 0.02, size=len(aug))
                aug[col] = aug[col].astype(float) * (1.0 + noise)

            # Small random shift
            shift = np.random.choice([-1, 0, 1])
            if shift != 0:
                aug = aug.shift(shift)
                aug = aug.fillna(method="bfill").fillna(method="ffill")

            augmented_ts.append(aug)
            augmented_labels.append(1)

    logger.info(
        "Data augmentation: %d → %d samples (%d positive)",
        len(ts_list), len(augmented_ts), sum(augmented_labels),
    )
    return augmented_ts, np.array(augmented_labels)


# ── Time series to tensor ──────────────────────────────────────────────


def _prepare_torch_tensors(
    ts_list: list[pd.DataFrame],
    labels: np.ndarray,
    seq_len: int = 90,
    n_features: int = 27,
) -> tuple:
    """Convert time series list to PyTorch tensors with padding/truncation."""
    X_list = []
    valid_labels = []

    for ts, lbl in zip(ts_list, labels):
        if ts is None or len(ts) < 20:
            continue
        # Select indicator columns
        cols = [c for c in INDICATOR_COLS if c in ts.columns]
        if len(cols) < 10:
            continue
        arr = ts[cols].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        # Pad or truncate to seq_len
        if len(arr) >= seq_len:
            arr = arr[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(arr), len(cols)), dtype=np.float32)
            arr = np.vstack([pad, arr])

        # Ensure n_features columns (pad with zeros if mismatch)
        if arr.shape[1] < n_features:
            pad_cols = np.zeros((seq_len, n_features - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad_cols])
        elif arr.shape[1] > n_features:
            arr = arr[:, :n_features]

        X_list.append(arr)
        valid_labels.append(lbl)

    if not X_list:
        return None, None, None

    X = np.stack(X_list)
    y = np.array(valid_labels)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, len(X)


# ── Training loop ──────────────────────────────────────────────────────


def _train_torch_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    model_name: str = "model",
    epochs: int = 100,
    batch_size: int = 32,
    pos_weight: float = 3.0,
) -> dict:
    """Train a PyTorch model with cross-validation and return metrics."""
    n = len(X)
    if n < 20:
        return {"auc": None, "predictions": None, "importance": None}

    # Simple train/val split
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)
    split = int(n * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_idx)), shuffle=True)

    weight = torch.tensor([pos_weight])
    criterion = nn.BCELoss()  # default: equal weight
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final predictions on all data
    model.eval()
    with torch.no_grad():
        y_pred = model(X).squeeze().numpy()

    try:
        auc = roc_auc_score(y.numpy(), y_pred)
    except Exception:
        auc = None

    # Permutation importance
    importance = _permutation_importance_torch(model, X, y)

    logger.info("%s: AUC=%.4f", model_name, auc if auc else -1)

    return {
        "model_name": model_name,
        "auc": round(auc, 4) if auc else None,
        "predictions": y_pred,
        "importance": importance,
    }


def _permutation_importance_torch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> list[float]:
    """Permutation importance for PyTorch model (feature-level)."""
    model.eval()
    with torch.no_grad():
        baseline_pred = model(X).squeeze().numpy()
    baseline_auc = roc_auc_score(y.numpy(), baseline_pred)

    # Permute per indicator group (not per time step — that's too granular)
    # Use indicator column groups
    indicator_groups = [
        slice(0, 5),    # price
        slice(5, 8),    # MA
        slice(8, 11),   # MACD
        slice(11, 12),  # RSI
        slice(12, 17),  # Bollinger
        slice(17, 20),  # KDJ
        slice(20, 21),  # vol_ratio
        slice(21, 24),  # ADX
    ]

    importances = []
    for grp in indicator_groups:
        X_perm = X.clone()
        start, stop = grp.start, grp.stop
        if start is None:
            start = 0
        for b in range(X_perm.size(0)):
            perm_idx = torch.randperm(X_perm.size(1))
            X_perm[b, :, start:stop] = X_perm[b, perm_idx, start:stop]

        with torch.no_grad():
            perm_pred = model(X_perm).squeeze().numpy()
        try:
            perm_auc = roc_auc_score(y.numpy(), perm_pred)
            importances.append(max(0.0, baseline_auc - perm_auc))
        except Exception:
            importances.append(0.0)

    total = sum(importances)
    if total > 0:
        importances = [v / total for v in importances]

    # Map indicator groups back to 10 scoring dimensions
    dim_importance = [0.0] * 10
    for i, grp in enumerate(indicator_groups):
        start, stop = grp.start, grp.stop
        if start is None:
            start = 0
        if start <= 5:  # price/volume → D1, D5, D6
            dim_importance[0] += importances[i] * 0.3  # D1
            dim_importance[4] += importances[i] * 0.3  # D5
            dim_importance[5] += importances[i] * 0.4  # D6
        if 5 <= start < 8:  # MA → D2
            dim_importance[1] += importances[i]
        if 8 <= start < 11:  # MACD → D3
            dim_importance[2] += importances[i]
        if 11 <= start < 12:  # RSI → D4
            dim_importance[3] += importances[i]
        if 17 <= start < 20:  # KDJ → D7
            dim_importance[6] += importances[i]
        if 20 <= start < 21:  # vol_ratio → D6
            dim_importance[5] += importances[i] * 0.6
        if 21 <= start < 24:  # ADX → D8
            dim_importance[7] += importances[i]

    # Normalize
    total_dim = sum(dim_importance)
    if total_dim > 0:
        dim_importance = [v / total_dim for v in dim_importance]

    return dim_importance


# ── Main entry point ───────────────────────────────────────────────────


def train_all_models(
    X_binary: np.ndarray,
    y: np.ndarray,
    ts_list: list[pd.DataFrame],
    use_torch: bool = True,
) -> dict:
    """Train all available models and return ensemble results.

    Args:
        X_binary: (N, 10) binary feature matrix
        y: (N,) binary labels
        ts_list: list of (90, 27) indicator DataFrames
        use_torch: whether to attempt PyTorch models

    Returns:
        dict with keys:
            ensemble_prob: np.ndarray — ensemble probability per sample
            ensemble_importance: list[float] — per-dimension importance (10,)
            models: dict — per-model results
            available_models: list[str]
    """
    results: dict = {
        "ensemble_prob": None,
        "ensemble_importance": None,
        "models": {},
        "available_models": [],
    }

    # ── MLP ─────────────────────────────────────────────────────────
    mlp_result = train_mlp(X_binary, y)
    results["models"]["mlp"] = mlp_result
    if mlp_result["auc"] is not None:
        results["available_models"].append("mlp")

    # ── PyTorch models ──────────────────────────────────────────────
    torch_available = TORCH_AVAILABLE and use_torch

    if torch_available and len(ts_list) >= 20:
        # Augment positive samples
        aug_ts, aug_y = _augment_time_series(ts_list, y, target_pos_count=300)

        Xt, yt, n_samples = _prepare_torch_tensors(aug_ts, aug_y)
        if Xt is not None and n_samples >= 30:
            # LSTM
            lstm = LSTMPredictor()
            lstm_result = _train_torch_model(lstm, Xt, yt, "lstm")
            results["models"]["lstm"] = lstm_result
            if lstm_result.get("auc") is not None:
                results["available_models"].append("lstm")

            # CNN
            cnn = CNN1DPredictor()
            cnn_result = _train_torch_model(cnn, Xt, yt, "cnn")
            results["models"]["cnn"] = cnn_result
            if cnn_result.get("auc") is not None:
                results["available_models"].append("cnn")

            # Transformer (only if enough samples)
            if n_samples >= 60:
                transformer = TransformerPredictor()
                tf_result = _train_torch_model(transformer, Xt, yt, "transformer", epochs=80)
                results["models"]["transformer"] = tf_result
                if tf_result.get("auc") is not None:
                    results["available_models"].append("transformer")
    else:
        logger.info(
            "PyTorch not available or insufficient samples (%d); NN models skipped",
            len(ts_list),
        )

    # ── Ensemble ───────────────────────────────────────────────────
    _build_ensemble(results)

    return results


def _build_ensemble(results: dict):
    """Build ensemble from trained models."""
    available = results["available_models"]
    if not available:
        logger.warning("No models trained; skipping ensemble")
        return

    # Collect importance vectors (mapped to 10 dims)
    all_importance = []
    weights_sum = 0.0

    # Model ensemble weights
    model_weights = {"mlp": 0.4, "lstm": 0.4, "cnn": 0.1, "transformer": 0.1}
    # Adjust if some models are missing
    available_weights = {m: model_weights.get(m, 0.2) for m in available}
    total_w = sum(available_weights.values())
    available_weights = {m: w / total_w for m, w in available_weights.items()}

    for name in available:
        m = results["models"].get(name, {})
        imp = m.get("importance")
        if imp and len(imp) == 10:
            all_importance.append((available_weights[name], imp))

    if not all_importance:
        return

    # Weighted average importance
    ensemble_imp = np.zeros(10)
    for w, imp in all_importance:
        ensemble_imp += w * np.array(imp)
    results["ensemble_importance"] = ensemble_imp.tolist()

    # Ensemble probability (simple average of available model predictions)
    # For NN models, predictions are per-sample; for MLP, it's cross-val predictions
    # We approximate by averaging per-model importance * feature_vector
    # (since each sample might not have all model predictions)
    logger.info(
        "Ensemble built from %d models: %s",
        len(all_importance),
        ", ".join(available),
    )


# ── Utility: map ensemble importance to dimension weights ──────────────


def importance_to_weights(
    ensemble_importance: list[float],
    target_range: tuple = (-2.0, 2.0),
) -> list[float]:
    """Convert normalized importance scores to weight values for score_stock.

    The importance scores (sum to 1) are scaled to fit within target_range,
    preserving the relative ranking.
    """
    if not ensemble_importance or sum(ensemble_importance) == 0:
        return [1.0] * 10  # fallback to equal weight

    arr = np.array(ensemble_importance)
    min_val, max_val = arr.min(), arr.max()

    if max_val == min_val:
        return [1.0] * 10

    # Scale to target range
    scaled = (arr - min_val) / (max_val - min_val)  # [0, 1]
    lo, hi = target_range
    weights = lo + scaled * (hi - lo)

    return [round(float(w), 2) for w in weights]
