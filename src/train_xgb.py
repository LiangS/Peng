"""Train per-horizon XGBoost classifiers on tabular window features.

One multiclass XGBClassifier per horizon (mirroring the TCN's per-horizon heads),
class-weighted to match the TCN's weighted cross-entropy. Reports through the
shared metrics module so results line up with the TCN.

Usage:
    python -m src.train_xgb --symbol 600519
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb

from .config import Config
from .metrics import format_report, horizon_report
from .prepare import prepare_windowed
from .tabular import window_to_tabular

DEFAULT_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
)


def run_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    horizons: Sequence[int],
    num_classes: int,
    class_weights: Optional[Sequence[np.ndarray]] = None,
    params: Optional[dict] = None,
    early_stopping_rounds: int = 30,
) -> Tuple[List[xgb.XGBClassifier], np.ndarray, List[np.ndarray], Dict]:
    """Fit one classifier per horizon. Returns (models, val_preds, val_proba, report).

    `X_tr`/`X_va` are tabular (N, D); `y_tr`/`y_va` are (N, num_horizons).
    """
    merged = {**DEFAULT_PARAMS, **(params or {}), "num_class": num_classes}
    models: List[xgb.XGBClassifier] = []
    preds = np.zeros_like(y_va)
    proba: List[np.ndarray] = []

    for i in range(len(horizons)):
        weight = None
        if class_weights is not None:
            weight = class_weights[i][y_tr[:, i]]      # per-sample weight by its class
        clf = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds, **merged)
        clf.fit(
            X_tr, y_tr[:, i],
            sample_weight=weight,
            eval_set=[(X_va, y_va[:, i])],
            verbose=False,
        )
        models.append(clf)
        preds[:, i] = clf.predict(X_va)
        proba.append(clf.predict_proba(X_va))

    report = horizon_report(y_va, preds, horizons, num_classes)
    return models, preds, proba, report


def train_xgb(cfg: Config) -> Dict:
    data = prepare_windowed(cfg)
    X_tr, columns = window_to_tabular(data.X_train, data.feature_names)
    X_va, _ = window_to_tabular(data.X_val, data.feature_names)
    print(
        f"XGBoost | symbol {cfg.symbol} | train={len(X_tr)} val={len(X_va)} "
        f"| {len(columns)} tabular features"
    )

    models, _, _, report = run_xgb(
        X_tr, data.y_train, X_va, data.y_val,
        cfg.horizons, cfg.num_classes, data.class_weights,
        params={"random_state": cfg.seed},
    )
    print(format_report(report, f"XGBoost ({cfg.symbol})"))

    outdir = os.path.join(cfg.ckpt_dir, f"xgb_{cfg.symbol}")
    os.makedirs(outdir, exist_ok=True)
    for h, m in zip(cfg.horizons, models):
        m.save_model(os.path.join(outdir, f"h{h}.json"))
    print(f"Saved {len(models)} models to {outdir}/")
    return report


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train per-horizon XGBoost on A-share data")
    p.add_argument("--symbol", default=cfg.symbol)
    p.add_argument("--start-date", default=cfg.start_date)
    p.add_argument("--end-date", default=cfg.end_date)
    p.add_argument("--window", type=int, default=cfg.window)
    args = p.parse_args()
    cfg.symbol, cfg.start_date, cfg.end_date, cfg.window = (
        args.symbol, args.start_date, args.end_date, args.window
    )
    return cfg


if __name__ == "__main__":
    train_xgb(parse_args())
