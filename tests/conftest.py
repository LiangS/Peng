"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


def _ohlcv(close: np.ndarray, start: str = "2015-01-01") -> pd.DataFrame:
    """Build a minimal valid OHLCV frame from a close-price array."""
    close = np.asarray(close, dtype=float)
    idx = pd.date_range(start, periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        },
        index=idx,
    )


@pytest.fixture
def make_ohlcv():
    """Factory: pass a close-price array, get an OHLCV DataFrame."""
    return _ohlcv


@pytest.fixture
def trending_ohlcv():
    """400 days of mild upward drift plus noise — enough for windowing/splits."""
    rng = np.random.RandomState(0)
    steps = rng.normal(0.0005, 0.02, size=400)
    close = 100.0 * np.exp(np.cumsum(steps))
    return _ohlcv(close)
