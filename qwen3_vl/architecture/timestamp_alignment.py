from __future__ import annotations

import torch
from torch import nn


class TimestampAlignment(nn.Module):
    """PaperRef: text-timestamp alignment for temporal grounding."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, text_states: torch.Tensor, video_states: torch.Tensor) -> torch.Tensor:
        q = self.q(text_states)
        k = self.k(video_states)
        v = self.v(video_states)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.size(-1) ** 0.5), dim=-1)
        return attn @ v
