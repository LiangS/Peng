"""XGBoost training contract from SPEC.md §5/§6."""

import numpy as np
import pytest

pytest.importorskip("xgboost")  # skip cleanly if xgboost/OpenMP unavailable

from src.tabular import window_to_tabular
from src.train_xgb import run_xgb


def _synthetic(n=200, f=8, window=60, horizons=2, num_classes=3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, f, window).astype(np.float32)
    # make labels weakly depend on a feature so the model can fit something
    signal = X[:, 0, -1]
    y = np.digitize(signal, bins=[-0.5, 0.5])          # 0/1/2
    y = np.repeat(y[:, None], horizons, axis=1).astype(np.int64)
    return X, y


def test_run_xgb_returns_models_preds_proba_report():
    X, y = _synthetic()
    names = [f"f{i}" for i in range(X.shape[1])]
    split = 150
    Xt_tr, _ = window_to_tabular(X[:split], names)
    Xt_va, _ = window_to_tabular(X[split:], names)
    horizons = [1, 5]

    models, preds, proba, report = run_xgb(
        Xt_tr, y[:split], Xt_va, y[split:],
        horizons=horizons, num_classes=3, class_weights=None,
        params={"n_estimators": 20, "random_state": 0}, early_stopping_rounds=10,
    )

    assert len(models) == 2
    assert preds.shape == (len(Xt_va), 2)
    assert all(p.shape == (len(Xt_va), 3) for p in proba)
    assert set(report.keys()) == {"h1", "h5"}
    for m in report.values():
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["macro_f1"] <= 1.0


def test_class_weights_are_accepted():
    X, y = _synthetic()
    names = [f"f{i}" for i in range(X.shape[1])]
    Xt, _ = window_to_tabular(X, names)
    weights = [np.array([1.0, 2.0, 0.5]), np.array([1.0, 1.0, 1.0])]
    models, preds, _, _ = run_xgb(
        Xt[:150], y[:150], Xt[150:], y[150:],
        horizons=[1, 5], num_classes=3, class_weights=weights,
        params={"n_estimators": 15, "random_state": 0}, early_stopping_rounds=10,
    )
    assert preds.shape == (50, 2)
