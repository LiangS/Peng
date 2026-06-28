"""Feature engineering and multi-horizon directional labels.

`build_features` -> (feature_df, label_df) aligned on the same date index.
Labels are 3-class direction (0=down, 1=flat, 2=up) per horizon, derived from
the forward return over that horizon.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_ret",
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "vol_10",
    "rsi_14",
    "vol_change",
    "hl_range",
]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model input features from an OHLCV DataFrame."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # Guard the ratio: anything <= 0 becomes NaN (and is dropped later) instead
    # of triggering "invalid value encountered in log".
    ratio = close / close.shift(1)
    out["log_ret"] = np.log(ratio.where(ratio > 0))
    out["ma5_ratio"] = close / close.rolling(5).mean() - 1
    out["ma10_ratio"] = close / close.rolling(10).mean() - 1
    out["ma20_ratio"] = close / close.rolling(20).mean() - 1
    out["vol_10"] = out["log_ret"].rolling(10).std()
    out["rsi_14"] = _rsi(close, 14) / 100.0          # scale to ~[0, 1]
    out["vol_change"] = np.log((df["volume"] + 1) / (df["volume"].shift(1) + 1))
    out["hl_range"] = (df["high"] - df["low"]) / close

    return out[FEATURE_COLUMNS]


def build_labels(
    df: pd.DataFrame, horizons: List[int], flat_threshold: float
) -> pd.DataFrame:
    """3-class forward-direction labels for each horizon.

    For horizon h, the forward return is close[t+h]/close[t] - 1. The "flat"
    band widens with sqrt(h) so longer horizons aren't dominated by drift.
    Rows where the future price isn't known yet are left as NaN.
    """
    close = df["close"]
    labels = pd.DataFrame(index=df.index)

    for h in horizons:
        fwd_ret = close.shift(-h) / close - 1.0
        band = flat_threshold * np.sqrt(h)
        cls = pd.Series(1, index=df.index, dtype="float")  # default flat
        cls[fwd_ret > band] = 2                            # up
        cls[fwd_ret < -band] = 0                           # down
        cls[fwd_ret.isna()] = np.nan                       # unknown future
        labels[f"h{h}"] = cls

    return labels


def _drop_nonpositive_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the contiguous positive-price tail.

    AKShare's qfq (forward-adjusted) prices can be <= 0 in the oldest history
    for stocks with large cumulative dividends. Those rows are unusable and also
    contaminate rolling features, so we drop everything up to and including the
    last non-positive close (this block is always at the start).
    """
    bad = (df["close"] <= 0).to_numpy()
    if bad.any():
        last_bad = int(np.flatnonzero(bad).max())
        df = df.iloc[last_bad + 1 :]
    return df


def build_dataset_frame(
    df: pd.DataFrame, horizons: List[int], flat_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and labels, dropping rows with NaN features (warm-up)."""
    df = _drop_nonpositive_prices(df)
    features = build_features(df)
    labels = build_labels(df, horizons, flat_threshold)

    valid = features.notna().all(axis=1)
    return features[valid], labels[valid]
