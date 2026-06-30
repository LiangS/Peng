"""Build the shared windowed dataset once, as plain numpy arrays.

Both `train.py` (TCN) and `train_xgb.py` (XGBoost) consume the output of this
function, guaranteeing they see the identical train/val split, scaling, and
samples — the precondition for a fair model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import Config
from .data import load_prices
from .dataset import build_windowed_data
from .features import FEATURE_COLUMNS, build_dataset_frame


@dataclass
class PreparedData:
    X_train: np.ndarray          # (N, F, window) float32
    y_train: np.ndarray          # (N, H) int64
    X_val: np.ndarray
    y_val: np.ndarray
    class_weights: List[np.ndarray]
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    @property
    def num_features(self) -> int:
        return self.X_train.shape[1]


def prepare_windowed(cfg: Config) -> PreparedData:
    prices = load_prices(cfg.symbol, cfg.start_date, cfg.end_date, cfg.adjust, cfg.cache_dir)
    features, labels = build_dataset_frame(prices, cfg.horizons, cfg.class_thresholds)
    data = build_windowed_data(features, labels, cfg.window, cfg.val_ratio, cfg.num_classes)

    X_tr, y_tr = data.train.tensors
    X_va, y_va = data.val.tensors
    return PreparedData(
        X_train=X_tr.numpy(),
        y_train=y_tr.numpy(),
        X_val=X_va.numpy(),
        y_val=y_va.numpy(),
        class_weights=[w.numpy() for w in data.class_weights],
        feature_names=list(FEATURE_COLUMNS),
        scaler_mean=data.scaler.mean_,
        scaler_scale=data.scaler.scale_,
    )
