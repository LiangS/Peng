"""Train the multi-horizon TCN.

Loss = mean over horizons of class-weighted cross-entropy. Validation tracks
per-horizon accuracy; we early-stop and checkpoint on the mean val accuracy.

Usage:
    python -m src.train --symbol 600519 --epochs 60
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .metrics import format_report, horizon_report
from .model import MultiHorizonTCN
from .prepare import prepare_windowed


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, horizons, device) -> List[float]:
    """Return per-horizon accuracy on a loader."""
    model.eval()
    correct = np.zeros(len(horizons))
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            for i, lg in enumerate(logits):
                correct[i] += (lg.argmax(-1) == y[:, i]).sum().item()
            total += y.size(0)
    return (correct / max(total, 1)).tolist()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Gather (y_true, y_pred) as (N, num_horizons) int arrays for reporting."""
    model.eval()
    trues, preds = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(torch.stack([lg.argmax(-1).cpu() for lg in logits], dim=1))
        trues.append(y)
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def train_from_arrays(cfg, X_tr, y_tr, X_va, y_va, class_weights, device=None,
                      history: Optional[list] = None):
    """Train the TCN on pre-windowed arrays. Returns (best_model, val_report).

    Kept separate from data loading so `compare.py` can train the TCN and
    XGBoost on the exact same arrays.

    If `history` is a list, one dict per epoch is appended to it
    (`train_loss`, per-horizon `val_acc`, `mean_acc`) for later plotting —
    see `src.plotting.plot_tcn_history`.
    """
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MultiHorizonTCN(
        num_features=X_tr.shape[1],
        horizons=cfg.horizons,
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    criteria = [
        nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        for w in class_weights
    ]
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    best_val, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = sum(criteria[i](logits[i], y[:, i]) for i in range(len(cfg.horizons)))
            loss = loss / len(cfg.horizons)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            epoch_loss += loss.item() * x.size(0)
        scheduler.step()

        val_acc = evaluate(model, val_loader, cfg.horizons, device)
        mean_acc = float(np.mean(val_acc))
        acc_str = " ".join(f"h{h}={a:.3f}" for h, a in zip(cfg.horizons, val_acc))
        train_loss = epoch_loss / len(train_ds)
        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} "
            f"| val_acc {acc_str} | mean {mean_acc:.3f}"
        )
        if history is not None:
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": list(val_acc),
                "mean_acc": mean_acc,
            })

        if mean_acc > best_val:
            best_val, patience = mean_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best mean val_acc={best_val:.3f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred = collect_predictions(model, val_loader, device)
    report = horizon_report(y_true, y_pred, cfg.horizons, cfg.num_classes)
    return model, report


def train(cfg: Config) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | receptive field: {cfg.receptive_field} steps")

    data = prepare_windowed(cfg)
    print(f"Features: {data.num_features} | train={len(data.X_train)} val={len(data.X_val)}")

    history: List[dict] = []
    model, report = train_from_arrays(
        cfg, data.X_train, data.y_train, data.X_val, data.y_val, data.class_weights, device,
        history=history,
    )
    print(format_report(report, f"TCN ({cfg.symbol})"))

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"tcn_{cfg.symbol}.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "scaler_mean": data.scaler_mean,
            "scaler_scale": data.scaler_scale,
            "report": report,
            "history": history,
        },
        ckpt_path,
    )
    print(f"Checkpoint: {ckpt_path}")
    return ckpt_path


def parse_args() -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="Train multi-horizon TCN on A-share data")
    p.add_argument("--symbol", default=cfg.symbol)
    p.add_argument("--start-date", default=cfg.start_date)
    p.add_argument("--end-date", default=cfg.end_date)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--window", type=int, default=cfg.window)
    args = p.parse_args()

    cfg.symbol = args.symbol
    cfg.start_date = args.start_date
    cfg.end_date = args.end_date
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.window = args.window
    return cfg


if __name__ == "__main__":
    train(parse_args())
