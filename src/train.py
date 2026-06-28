"""Train the multi-horizon TCN.

Loss = mean over horizons of class-weighted cross-entropy. Validation tracks
per-horizon accuracy; we early-stop and checkpoint on the mean val accuracy.

Usage:
    python -m src.train --symbol 600519 --epochs 60
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data import load_prices
from .dataset import build_windowed_data
from .features import build_dataset_frame
from .model import MultiHorizonTCN


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


def train(cfg: Config) -> str:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | receptive field: {cfg.receptive_field} steps")

    # --- Data ---------------------------------------------------------------
    prices = load_prices(cfg.symbol, cfg.start_date, cfg.end_date, cfg.adjust, cfg.cache_dir)
    features, labels = build_dataset_frame(prices, cfg.horizons, cfg.flat_threshold)
    data = build_windowed_data(features, labels, cfg.window, cfg.val_ratio, cfg.num_classes)
    print(f"Features: {data.num_features} | train={len(data.train)} val={len(data.val)}")

    train_loader = DataLoader(data.train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(data.val, batch_size=cfg.batch_size, shuffle=False)

    # --- Model / optim ------------------------------------------------------
    model = MultiHorizonTCN(
        num_features=data.num_features,
        horizons=cfg.horizons,
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)

    criteria = [nn.CrossEntropyLoss(weight=w.to(device)) for w in data.class_weights]
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"tcn_{cfg.symbol}.pt")

    best_val = -1.0
    patience = 0

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
        print(
            f"epoch {epoch:3d} | loss {epoch_loss / len(data.train):.4f} "
            f"| val_acc {acc_str} | mean {mean_acc:.3f}"
        )

        if mean_acc > best_val:
            best_val = mean_acc
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "scaler_mean": data.scaler.mean_,
                    "scaler_scale": data.scaler.scale_,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best mean val_acc={best_val:.3f})")
                break

    print(f"Best mean val accuracy: {best_val:.3f} | checkpoint: {ckpt_path}")
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
