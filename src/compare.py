"""Train every model on one shared dataset and print a side-by-side comparison.

Both models consume the identical windowed train/val split (via prepare.py), so
differences reflect the models, not the data. Add new models by appending to the
`reports` dict.

Usage:
    python -m src.compare --symbol 600519 --epochs 60
"""

from __future__ import annotations

import argparse

from .config import Config
from .metrics import format_comparison, format_report
from .prepare import prepare_windowed
from .tabular import window_to_tabular
from .train import train_from_arrays
from .train_xgb import run_xgb


def compare(cfg: Config) -> dict:
    data = prepare_windowed(cfg)
    print(
        f"Shared data | symbol {cfg.symbol} | features {data.num_features} "
        f"| train={len(data.X_train)} val={len(data.X_val)}\n"
    )
    reports = {}

    # --- XGBoost (tabular features from the same windows) -------------------
    print(">>> Training XGBoost ...")
    X_tr, columns = window_to_tabular(data.X_train, data.feature_names)
    X_va, _ = window_to_tabular(data.X_val, data.feature_names)
    _, _, _, xgb_report = run_xgb(
        X_tr, data.y_train, X_va, data.y_val,
        cfg.horizons, cfg.num_classes, data.class_weights,
        params={"random_state": cfg.seed},
    )
    reports["xgboost"] = xgb_report
    print(format_report(xgb_report, f"XGBoost ({cfg.symbol})"), "\n")

    # --- TCN (same windows) -------------------------------------------------
    print(">>> Training TCN ...")
    _, tcn_report = train_from_arrays(
        cfg, data.X_train, data.y_train, data.X_val, data.y_val, data.class_weights
    )
    reports["tcn"] = tcn_report
    print(format_report(tcn_report, f"TCN ({cfg.symbol})"), "\n")

    # --- Side-by-side -------------------------------------------------------
    print(format_comparison(reports, "accuracy"))
    print()
    print(format_comparison(reports, "macro_f1"))
    return reports


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Compare models on one shared dataset")
    p.add_argument("--symbol", default=cfg.symbol)
    p.add_argument("--start-date", default=cfg.start_date)
    p.add_argument("--end-date", default=cfg.end_date)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--window", type=int, default=cfg.window)
    args = p.parse_args()
    cfg.symbol, cfg.start_date, cfg.end_date, cfg.epochs, cfg.window = (
        args.symbol, args.start_date, args.end_date, args.epochs, args.window
    )
    return cfg


if __name__ == "__main__":
    compare(parse_args())
