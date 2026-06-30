"""Shared, model-agnostic evaluation metrics (SPEC.md §6).

Both the TCN and XGBoost report through these functions so comparisons are
apples-to-apples. We deliberately report the **majority-class baseline** and
**macro-F1** alongside accuracy: class-weighted training pushes models toward
balanced predictions, so raw accuracy alone can look like chance while the model
is actually doing something useful (or, conversely, can sit below the trivial
baseline).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def majority_baseline_accuracy(y_true: np.ndarray, num_classes: int) -> float:
    """Accuracy of always predicting the most frequent class."""
    if len(y_true) == 0:
        return float("nan")
    counts = np.bincount(y_true, minlength=num_classes)
    return float(counts.max() / len(y_true))


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "baseline_acc": majority_baseline_accuracy(y_true, num_classes),
    }


def horizon_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: Sequence[int],
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    """Per-horizon metrics. `y_true`/`y_pred` are (N, num_horizons) int arrays."""
    return {
        f"h{h}": classification_metrics(y_true[:, i], y_pred[:, i], num_classes)
        for i, h in enumerate(horizons)
    }


def format_report(report: Dict[str, Dict[str, float]], title: str) -> str:
    """Render a per-horizon report as a fixed-width table."""
    lines = [f"=== {title} ===",
             f"{'horizon':>8} {'acc':>8} {'macroF1':>8} {'baseline':>9} {'lift':>7}"]
    for hk, m in report.items():
        lift = m["accuracy"] - m["baseline_acc"]
        lines.append(
            f"{hk:>8} {m['accuracy']:>8.3f} {m['macro_f1']:>8.3f} "
            f"{m['baseline_acc']:>9.3f} {lift:>+7.3f}"
        )
    return "\n".join(lines)


def format_comparison(
    reports: Dict[str, Dict[str, Dict[str, float]]], metric: str = "accuracy"
) -> str:
    """Side-by-side table of one metric across models.

    `reports` maps model name -> per-horizon report (output of `horizon_report`).
    Includes the majority baseline (identical across models) as a reference row.
    """
    model_names: List[str] = list(reports.keys())
    horizons = list(next(iter(reports.values())).keys())

    header = f"{'horizon':>8} {'baseline':>9} " + " ".join(f"{m:>10}" for m in model_names)
    lines = [f"=== model comparison ({metric}) ===", header]
    for hk in horizons:
        baseline = next(iter(reports.values()))[hk]["baseline_acc"]
        row = f"{hk:>8} {baseline:>9.3f} "
        row += " ".join(f"{reports[m][hk][metric]:>10.3f}" for m in model_names)
        lines.append(row)
    return "\n".join(lines)
