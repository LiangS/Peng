"""Feature engineering and multi-horizon directional labels.

`build_features` -> (feature_df, label_df) aligned on the same date index.
Labels are symmetric return buckets per horizon (see `build_labels`): a sorted
list of positive thresholds is mirrored around zero into `2*len+1` classes,
class 0 = biggest drop ... class `2*len` = biggest gain, derived from the
forward return over that horizon.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

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
    df: pd.DataFrame, horizons: List[int], thresholds: Sequence[float]
) -> pd.DataFrame:
    """Symmetric multi-bucket forward-direction labels for each horizon.

    `thresholds` are positive per-day-equivalent return edges (e.g.
    `[0.005, 0.02, 0.05]`). For horizon h the forward return is
    `close[t+h]/close[t] - 1`; each edge is scaled by sqrt(h) and mirrored
    around zero, so a length-k list yields `2k+1` buckets:
    class 0 = biggest drop ... class k = flat ... class 2k = biggest gain.
    Rows where the future price isn't known yet are left as NaN.
    """
    close = df["close"]
    thr = np.sort(np.asarray(list(thresholds), dtype=float))
    if thr.size == 0 or (thr <= 0).any():
        raise ValueError("thresholds must be a non-empty list of positive values")

    labels = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd_ret = (close.shift(-h) / close - 1.0).to_numpy()
        edges = np.concatenate([-thr[::-1], thr]) * np.sqrt(h)  # ascending boundaries
        cls = np.digitize(fwd_ret, edges).astype(float)         # 0 .. 2*len(thr)
        cls[np.isnan(fwd_ret)] = np.nan                         # unknown future
        labels[f"h{h}"] = cls

    return labels


def class_names(thresholds: Sequence[float]) -> List[str]:
    """Human-readable labels for the symmetric buckets, low class -> high.

    For `[0.005, 0.02, 0.05]` returns
    `['down >5%', 'down 2%-5%', 'down 0.5%-2%', 'flat',
      'up 0.5%-2%', 'up 2%-5%', 'up >5%']`.
    """
    thr = sorted(float(t) for t in thresholds)
    pct = lambda x: f"{x * 100:g}%"
    names = [f"down >{pct(thr[-1])}"]
    for hi, lo in zip(thr[:0:-1], thr[-2::-1]):     # widest band inward
        names.append(f"down {pct(lo)}-{pct(hi)}")
    names.append("flat")
    for lo, hi in zip(thr[:-1], thr[1:]):
        names.append(f"up {pct(lo)}-{pct(hi)}")
    names.append(f"up >{pct(thr[-1])}")
    return names


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
    df: pd.DataFrame, horizons: List[int], thresholds: Sequence[float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and labels, dropping rows with NaN features (warm-up)."""
    df = _drop_nonpositive_prices(df)
    features = build_features(df)
    labels = build_labels(df, horizons, thresholds)

    valid = features.notna().all(axis=1)
    return features[valid], labels[valid]
