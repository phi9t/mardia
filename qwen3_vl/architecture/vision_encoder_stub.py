from __future__ import annotations

import torch
from torch import nn


class VisionEncoderStub(nn.Module):
    """Compact ViT-like patch embedder for educational flow checks."""

    def __init__(self, in_ch: int = 3, hidden: int = 512, patch: int = 16):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(in_ch, hidden, kernel_size=patch, stride=patch)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.embed(images)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
