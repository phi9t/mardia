from __future__ import annotations

import torch


def codec_ce_loss(logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
    # logits: [B, T, V], target: [B, T]
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
