"""Tabular feature builder contracts from SPEC.md §3."""

import numpy as np
import pytest

from src.tabular import window_to_tabular


def test_output_shape_and_column_count():
    n, f, window = 7, 8, 60
    X = np.random.RandomState(0).randn(n, f, window).astype(np.float32)
    names = [f"feat{i}" for i in range(f)]
    table, cols = window_to_tabular(X, names, lags=(0, 1, 5, 20))
    # 4 lags + mean + std = 6 blocks of F columns
    assert table.shape == (n, f * 6)
    assert len(cols) == f * 6
    assert table.dtype == np.float32


def test_lag0_is_last_timestep():
    X = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    names = ["a", "b", "c"]
    table, cols = window_to_tabular(X, names, lags=(0,))
    # first F columns are lag0 == X[:, :, -1]
    np.testing.assert_array_equal(table[:, :3], X[:, :, -1])
    assert cols[:3] == ["a_lag0", "b_lag0", "c_lag0"]


def test_mean_and_std_columns_match_numpy():
    X = np.random.RandomState(1).randn(4, 2, 30).astype(np.float32)
    table, cols = window_to_tabular(X, ["a", "b"], lags=(0,))
    mean_block = table[:, 2:4]
    std_block = table[:, 4:6]
    np.testing.assert_allclose(mean_block, X.mean(axis=2), rtol=1e-5)
    np.testing.assert_allclose(std_block, X.std(axis=2), rtol=1e-5)
    assert cols[2:] == ["a_mean", "b_mean", "a_std", "b_std"]


def test_lag_exceeding_window_raises():
    X = np.zeros((1, 2, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        window_to_tabular(X, ["a", "b"], lags=(0, 20))


def test_feature_name_mismatch_raises():
    X = np.zeros((1, 3, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        window_to_tabular(X, ["only_one_name"], lags=(0,))
