"""Central configuration for the TCN stock-trend model.

Everything tunable lives here so the training script and the Colab notebook
share one source of truth. Override fields when constructing `Config(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- Data ---------------------------------------------------------------
    symbol: str = "600519"            # AKShare A-share code, e.g. 600519 = Kweichow Moutai
    start_date: str = "20150101"
    end_date: str = "20241231"
    adjust: str = "qfq"               # qfq = forward-adjusted prices (recommended for modelling)
    cache_dir: str = "data/cache"

    # --- Labels -------------------------------------------------------------
    # Predict directional class at each horizon (trading days into the future).
    horizons: List[int] = field(default_factory=lambda: [1, 5, 20])
    # 3-class: 0=down, 1=flat, 2=up. A move within +/- flat_threshold is "flat".
    # The threshold scales with sqrt(horizon) so longer horizons get a wider band.
    flat_threshold: float = 0.005     # 0.5% per-day-equivalent band
    num_classes: int = 3

    # --- Windowing ----------------------------------------------------------
    window: int = 60                  # look-back length fed to the TCN

    # --- Model --------------------------------------------------------------
    channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2

    # --- Training -----------------------------------------------------------
    val_ratio: float = 0.2            # last 20% of the timeline is validation
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    seed: int = 42

    # --- IO -----------------------------------------------------------------
    ckpt_dir: str = "checkpoints"

    @property
    def receptive_field(self) -> int:
        """How many timesteps the stacked dilated convs can see."""
        return 1 + 2 * (self.kernel_size - 1) * (2 ** len(self.channels) - 1)
