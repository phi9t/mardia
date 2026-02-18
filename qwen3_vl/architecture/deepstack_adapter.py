from __future__ import annotations

from typing import List

import torch
from torch import nn


class DeepStackAdapter(nn.Module):
    """PaperRef: DeepStack multi-level visual feature fusion."""

    def __init__(self, in_dims: List[int], out_dim: int):
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(d, out_dim) for d in in_dims])
        self.gate = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Sigmoid())

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        fused = 0.0
        for feat, proj in zip(feats, self.projs):
            fused = fused + proj(feat)
        fused = fused / max(len(feats), 1)
        return fused * self.gate(fused)
