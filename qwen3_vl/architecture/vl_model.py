from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .deepstack_adapter import DeepStackAdapter
from .interleaved_mrope import apply_interleaved_mrope
from .timestamp_alignment import TimestampAlignment
from .vision_encoder_stub import VisionEncoderStub


@dataclass
class Qwen3VLEducationalConfig:
    vocab_size: int = 64000
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    image_size: int = 224


class Qwen3VLEducationalModel(nn.Module):
    def __init__(self, cfg: Optional[Qwen3VLEducationalConfig] = None):
        super().__init__()
        self.cfg = cfg or Qwen3VLEducationalConfig()
        self.text_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_size,
            nhead=self.cfg.num_heads,
            batch_first=True,
            dim_feedforward=self.cfg.hidden_size * 4,
        )
        self.decoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.num_layers)
        self.vision = VisionEncoderStub(hidden=self.cfg.hidden_size)
        self.deepstack = DeepStackAdapter([self.cfg.hidden_size] * 3, self.cfg.hidden_size)
        self.temporal = TimestampAlignment(self.cfg.hidden_size)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size)

    def forward(self, text_ids: torch.LongTensor, images: torch.Tensor, video_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        text = self.text_emb(text_ids)
        v1 = self.vision(images)
        v2 = v1 * 0.8
        v3 = v1 * 1.2
        vision = self.deepstack([v1, v2, v3])

        if video_tokens is not None:
            aligned = self.temporal(text, video_tokens)
            text = text + aligned

        joint = torch.cat([vision, text], dim=1)
        joint = apply_interleaved_mrope(joint)
        h = self.decoder(joint)
        out = h[:, -text_ids.size(1) :, :]
        return self.lm_head(out)
