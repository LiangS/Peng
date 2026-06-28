"""Sliding-window dataset construction with a time-ordered train/val split.

Windows are built over the full timeline, then assigned to train or val by the
window's *target* date — so validation windows can still use earlier history as
context, but no validation target ever leaks into training. The feature scaler
is fit on the training portion only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


@dataclass
class WindowedData:
    train: TensorDataset
    val: TensorDataset
    scaler: StandardScaler
    class_weights: List[torch.Tensor]   # one (num_classes,) tensor per horizon
    num_features: int


def _make_windows(
    feats: np.ndarray, labels: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, F, window), y (N, H), and target-index array (N,).

    A window ending at index t is kept only if every horizon label at t is
    known (not NaN).
    """
    xs, ys, idx = [], [], []
    n = len(feats)
    for t in range(window - 1, n):
        y = labels[t]
        if np.isnan(y).any():
            continue
        win = feats[t - window + 1 : t + 1]      # (window, F)
        xs.append(win.T)                          # (F, window) for Conv1d
        ys.append(y)
        idx.append(t)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(idx)


def build_windowed_data(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    window: int,
    val_ratio: float,
    num_classes: int,
) -> WindowedData:
    feats = features.to_numpy(dtype=np.float64)
    labs = labels.to_numpy(dtype=np.float64)
    n = len(feats)

    # Time-ordered cutoff: rows before `cutoff` are training territory.
    cutoff = int(n * (1 - val_ratio))

    # Fit scaler on training rows only, then transform everything.
    scaler = StandardScaler().fit(feats[:cutoff])
    feats_scaled = scaler.transform(feats)

    X, y, idx = _make_windows(feats_scaled, labs, window)
    is_train = idx < cutoff

    X_tr, y_tr = X[is_train], y[is_train]
    X_va, y_va = X[~is_train], y[~is_train]

    # Inverse-frequency class weights per horizon (from the training split).
    class_weights = []
    for h in range(y_tr.shape[1]):
        counts = np.bincount(y_tr[:, h], minlength=num_classes).astype(np.float64)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        class_weights.append(torch.tensor(w, dtype=torch.float32))

    train = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))

    return WindowedData(
        train=train,
        val=val,
        scaler=scaler,
        class_weights=class_weights,
        num_features=feats.shape[1],
    )
