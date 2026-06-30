"""Training-process visualization helpers (used in the notebooks).

Kept dependency-light: matplotlib only, no global styling, and each function
returns the Figure so callers can save or further tweak it. These read the
artifacts the trainers already produce — the TCN's per-epoch `history` list and
XGBoost's per-round `evals_result()` — so plotting never re-runs training.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def plot_tcn_history(history: List[Dict], horizons: Sequence[int]):
    """Plot TCN training loss and per-horizon validation accuracy vs epoch.

    `history` is the list filled by `train.train_from_arrays` (also stored in
    the checkpoint under ``"history"``). Returns the matplotlib Figure.
    """
    if not history:
        raise ValueError("empty history — train with history capture enabled first")

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], color="tab:red")
    ax_loss.set(xlabel="epoch", ylabel="train loss", title="Training loss")
    ax_loss.grid(alpha=0.3)

    for i, h in enumerate(horizons):
        ax_acc.plot(epochs, [rec["val_acc"][i] for rec in history], label=f"h{h}")
    ax_acc.plot(epochs, [rec["mean_acc"] for rec in history],
                color="black", linestyle="--", label="mean")
    ax_acc.set(xlabel="epoch", ylabel="val accuracy", title="Validation accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_xgb_evals(models, horizons: Sequence[int], metric: str = "mlogloss"):
    """Plot per-horizon validation `metric` vs boosting round for XGBoost.

    `models` is the list returned by `train_xgb`; each was fit with an
    `eval_set`, so `evals_result()` holds the validation curve. Returns the
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for h, m in zip(horizons, models):
        results = m.evals_result()
        # Single eval_set -> "validation_0"; take its requested metric.
        curve = next(iter(results.values()))[metric]
        ax.plot(range(1, len(curve) + 1), curve, label=f"h{h}")
    ax.set(xlabel="boosting round", ylabel=f"validation {metric}",
           title=f"XGBoost validation {metric}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
