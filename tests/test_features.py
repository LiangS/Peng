"""Feature/label contracts from SPEC.md §3."""

import warnings

import numpy as np
import pytest

from src.features import (
    FEATURE_COLUMNS,
    _drop_nonpositive_prices,
    build_dataset_frame,
    build_features,
    build_labels,
)


def test_drop_nonpositive_prices_removes_leading_block(make_ohlcv):
    df = make_ohlcv([-3.0, -1.0, 0.0, 5.0, 6.0, 7.0])
    out = _drop_nonpositive_prices(df)
    assert len(out) == 3
    assert (out["close"] > 0).all()


def test_drop_nonpositive_is_noop_when_all_positive(make_ohlcv):
    df = make_ohlcv([1.0, 2.0, 3.0])
    assert len(_drop_nonpositive_prices(df)) == 3


def test_features_have_expected_columns_and_no_nan(trending_ohlcv):
    feats = build_features(trending_ohlcv).dropna()
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()


@pytest.mark.parametrize(
    "close,expected",
    [
        ([100, 110, 121, 133], 2),   # +10%/step -> up
        ([100, 90, 81, 73], 0),      # -10%/step -> down
        ([100, 100, 100, 100], 1),   # flat
    ],
)
def test_label_direction_h1(make_ohlcv, close, expected):
    df = make_ohlcv(close)
    labels = build_labels(df, horizons=[1], flat_threshold=0.005)
    # first row's 1-day-forward label is well-defined
    assert labels["h1"].iloc[0] == expected


def test_label_tail_is_nan_for_unknown_future(make_ohlcv):
    df = make_ohlcv(np.linspace(100, 120, 30))
    labels = build_labels(df, horizons=[5], flat_threshold=0.005)
    assert labels["h5"].iloc[-5:].isna().all()      # last h rows unknown
    assert labels["h5"].iloc[:-5].notna().all()


def test_no_invalid_log_warning_on_negative_qfq_prices(make_ohlcv):
    # leading negative block (qfq artifact) then a positive series
    close = np.concatenate([np.linspace(-5, -0.5, 8), np.linspace(2, 60, 292)])
    df = make_ohlcv(close)
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # any RuntimeWarning -> failure
        feats, labels = build_dataset_frame(df, [1, 5, 20], 0.005)
    assert not feats.isna().any().any()
    assert np.isfinite(feats.to_numpy()).all()


def test_build_dataset_frame_aligns_features_and_labels(trending_ohlcv):
    feats, labels = build_dataset_frame(trending_ohlcv, [1, 5, 20], 0.005)
    assert feats.index.equals(labels.index)
    assert list(labels.columns) == ["h1", "h5", "h20"]
