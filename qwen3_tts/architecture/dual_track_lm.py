from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DualTrackConfig:
    vocab_size: int = 1024
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    codebooks: int = 16


class DualTrackLM(nn.Module):
    """PaperRef: dual-track architecture abstraction.

    track 1: linguistic/control context path.
    track 2: acoustic token autoregressive path.
    """

    def __init__(self, cfg: DualTrackConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=cfg.num_heads,
            batch_first=True,
            dim_feedforward=cfg.hidden_size * 4,
        )
        self.linguistic_tower = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.acoustic_tower = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.merge = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)

    def forward(self, text_ids: torch.LongTensor, codec_ids: torch.LongTensor) -> torch.Tensor:
        text_h = self.linguistic_tower(self.token_emb(text_ids))
        codec_h = self.acoustic_tower(self.token_emb(codec_ids))
        h = self.merge(torch.cat([text_h, codec_h], dim=-1))
        return self.head(h)
