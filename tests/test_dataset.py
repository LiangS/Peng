"""Windowing / split / scaling contracts from SPEC.md §4 — the no-leakage invariants."""

import numpy as np

from src.dataset import _make_windows, build_windowed_data
from src.features import build_dataset_frame


def test_make_windows_shapes_and_dtypes():
    n, f, h, window = 50, 4, 3, 10
    feats = np.random.RandomState(1).randn(n, f)
    labels = np.tile([0, 1, 2], (n, 1)).astype(float)
    X, y, idx = _make_windows(feats, labels, window)
    assert X.shape == (n - window + 1, f, window)   # (N, F, window) for Conv1d
    assert y.shape == (n - window + 1, h)
    assert idx.min() == window - 1 and idx.max() == n - 1
    assert X.dtype == np.float32 and y.dtype == np.int64


def test_windows_skip_unknown_future_labels():
    feats = np.zeros((20, 2))
    labels = np.tile([1.0, 1.0], (20, 1))
    labels[-3:, 0] = np.nan                          # last 3 future-unknown
    _, _, idx = _make_windows(feats, labels, window=5)
    assert idx.max() == 16                           # 17,18,19 dropped (NaN label)


def test_time_split_no_leakage(trending_ohlcv):
    feats, labels = build_dataset_frame(trending_ohlcv, [1, 5, 20], [0.005])
    window, val_ratio = 30, 0.2
    data = build_windowed_data(feats, labels, window, val_ratio, num_classes=3)

    # Reconstruct window end-indices to assert the split is clean in time.
    _, _, idx = _make_windows(
        feats.to_numpy(), labels.to_numpy(dtype=float), window
    )
    cutoff = int(len(feats) * (1 - val_ratio))
    train_idx, val_idx = idx[idx < cutoff], idx[idx >= cutoff]

    assert len(data.train) == len(train_idx)
    assert len(data.val) == len(val_idx)
    # Strict time order: every training window precedes every validation window.
    assert train_idx.max() < val_idx.min()


def test_scaler_fit_on_train_rows_only(trending_ohlcv):
    feats, labels = build_dataset_frame(trending_ohlcv, [1, 5, 20], [0.005])
    val_ratio = 0.2
    data = build_windowed_data(feats, labels, window=30, val_ratio=val_ratio, num_classes=3)

    cutoff = int(len(feats) * (1 - val_ratio))
    expected_mean = feats.to_numpy()[:cutoff].mean(axis=0)
    np.testing.assert_allclose(data.scaler.mean_, expected_mean, rtol=1e-6)


def test_class_weights_shape(trending_ohlcv):
    feats, labels = build_dataset_frame(trending_ohlcv, [1, 5, 20], [0.005])
    data = build_windowed_data(feats, labels, window=30, val_ratio=0.2, num_classes=3)
    assert len(data.class_weights) == 3              # one per horizon
    for w in data.class_weights:
        assert tuple(w.shape) == (3,)                # one per class
        assert (w >= 0).all()
