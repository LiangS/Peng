"""Turn TCN-style windows into tabular features for tree models (SPEC.md §3/§8).

XGBoost can't consume a raw (features x time) window the way a TCN does, and
flattening all 480 cells produces many collinear laggy columns that trees handle
poorly. Instead we summarise each window into a compact, tree-friendly vector:
every feature sampled at a few lags from the window end, plus its window mean and
std. Crucially this is derived from the *same* windows as the TCN, so the
train/val split and sample alignment are identical — a fair comparison.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Lags measured backwards from the window's final timestep (0 = last/current day).
DEFAULT_LAGS: Tuple[int, ...] = (0, 1, 5, 20)


def window_to_tabular(
    X: np.ndarray,
    feature_names: Sequence[str],
    lags: Sequence[int] = DEFAULT_LAGS,
) -> Tuple[np.ndarray, List[str]]:
    """Map windows to a tabular matrix.

    Args:
        X: array of shape (N, F, window) — same layout the TCN consumes.
        feature_names: length-F names for readable output columns.
        lags: offsets from the window end to sample (must be < window).

    Returns:
        (table, columns) where table is (N, F * (len(lags) + 2)) float32.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (N, F, window), got shape {X.shape}")
    n, f, window = X.shape
    if f != len(feature_names):
        raise ValueError(f"{f} features but {len(feature_names)} names")
    if max(lags) >= window:
        raise ValueError(f"max lag {max(lags)} >= window {window}")

    parts: List[np.ndarray] = []
    columns: List[str] = []

    for lag in lags:
        parts.append(X[:, :, window - 1 - lag])          # (N, F)
        columns += [f"{name}_lag{lag}" for name in feature_names]

    parts.append(X.mean(axis=2))                          # window mean
    columns += [f"{name}_mean" for name in feature_names]
    parts.append(X.std(axis=2))                           # window std
    columns += [f"{name}_std" for name in feature_names]

    return np.concatenate(parts, axis=1).astype(np.float32), columns
