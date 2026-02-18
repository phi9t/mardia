from __future__ import annotations

import math
from typing import Tuple

import torch


def build_axes_positions(seq_len: int, height: int, width: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.arange(seq_len, device=device)
    h = torch.arange(height, device=device).repeat_interleave(max(1, seq_len // max(height, 1)))[:seq_len]
    w = torch.arange(width, device=device).repeat_interleave(max(1, seq_len // max(width, 1)))[:seq_len]
    if h.numel() < seq_len:
        h = torch.nn.functional.pad(h, (0, seq_len - h.numel()))
    if w.numel() < seq_len:
        w = torch.nn.functional.pad(w, (0, seq_len - w.numel()))
    return t, h, w


def apply_interleaved_mrope(x: torch.Tensor, height: int = 32, width: int = 32) -> torch.Tensor:
    """PaperRef: Interleaved-MRoPE with mixed temporal/spatial frequencies.

    x: [B, T, D]
    """
    b, t, d = x.shape
    device = x.device
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]

    pos_t, pos_h, pos_w = build_axes_positions(t, height, width, device)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device) / max(half, 1))
    angles = (
        pos_t[:, None] * freqs[None, :]
        + 0.5 * pos_h[:, None] * freqs[None, :]
        + 0.5 * pos_w[:, None] * freqs[None, :]
    )
    cos = torch.cos(angles)[None, :, :]
    sin = torch.sin(angles)[None, :, :]

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    if d % 2 == 0:
        return torch.cat([y1, y2], dim=-1)
    return torch.cat([y1, y2, x[..., -1:]], dim=-1)
