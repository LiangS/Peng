"""Shared-metrics contracts from SPEC.md §6."""

import numpy as np

from src.metrics import (
    classification_metrics,
    format_comparison,
    horizon_report,
    majority_baseline_accuracy,
)


def test_majority_baseline_is_fraction_of_most_common_class():
    y = np.array([0, 0, 0, 1, 2])           # class 0 is 3/5
    assert majority_baseline_accuracy(y, num_classes=3) == 0.6


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 1, 0])
    m = classification_metrics(y, y.copy(), num_classes=3)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_horizon_report_has_one_entry_per_horizon():
    y_true = np.array([[0, 1], [1, 2], [2, 0], [0, 0]])
    y_pred = y_true.copy()
    rep = horizon_report(y_true, y_pred, horizons=[1, 5], num_classes=3)
    assert set(rep.keys()) == {"h1", "h5"}
    assert rep["h1"]["accuracy"] == 1.0


def test_format_comparison_lists_each_model_and_baseline():
    rep = {"h1": {"accuracy": 0.5, "macro_f1": 0.4, "baseline_acc": 0.45}}
    out = format_comparison({"tcn": rep, "xgboost": rep}, metric="accuracy")
    assert "tcn" in out and "xgboost" in out
    assert "baseline" in out and "h1" in out
