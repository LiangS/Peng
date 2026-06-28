"""Temporal Convolutional Network with one softmax head per horizon.

Architecture follows Bai, Kolter & Koltun (2018), "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks for Sequence Modeling": stacked
dilated *causal* 1D convolutions with residual connections. A shared backbone
feeds N independent linear heads, one per prediction horizon; each head emits
class logits, and `predict_proba` applies softmax to get per-direction
probabilities.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Trim the right padding so each conv stays causal (no future leakage)."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channels: Sequence[int], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = num_inputs if i == 0 else channels[i - 1]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        return self.network(x)


class MultiHorizonTCN(nn.Module):
    """TCN backbone + one classification head per horizon."""

    def __init__(
        self,
        num_features: int,
        horizons: Sequence[int],
        num_classes: int = 3,
        channels: Sequence[int] = (64, 64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.num_classes = num_classes
        self.tcn = TemporalConvNet(num_features, list(channels), kernel_size, dropout)
        feat_dim = channels[-1]
        self.heads = nn.ModuleList(
            [nn.Linear(feat_dim, num_classes) for _ in self.horizons]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """x: (B, num_features, T) -> list of (B, num_classes) logits per horizon."""
        z = self.tcn(x)        # (B, F, T)
        z = z[:, :, -1]        # last timestep summarises the window: (B, F)
        return [head(z) for head in self.heads]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Per-horizon softmax probabilities: list of (B, num_classes)."""
        self.eval()
        return [torch.softmax(logits, dim=-1) for logits in self.forward(x)]
