from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from .dual_track_lm import DualTrackConfig, DualTrackLM
from .speech_decoder_convnet import SpeechDecoderConvNet
from .tokenizer_12hz import EncodedAudio, Tokenizer12Hz


@dataclass
class Qwen3TTSEducationalConfig:
    vocab_size: int = 1024
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    codebooks: int = 16


class Qwen3TTSEducationalModel(nn.Module):
    """Educational Qwen3-TTS model with custom/design/clone behaviors."""

    def __init__(self, cfg: Optional[Qwen3TTSEducationalConfig] = None):
        super().__init__()
        self.cfg = cfg or Qwen3TTSEducationalConfig()
        self.tokenizer = Tokenizer12Hz(codebooks=self.cfg.codebooks, bins=self.cfg.vocab_size)
        self.lm = DualTrackLM(
            DualTrackConfig(
                vocab_size=self.cfg.vocab_size,
                hidden_size=self.cfg.hidden_size,
                num_layers=self.cfg.num_layers,
                num_heads=self.cfg.num_heads,
                codebooks=self.cfg.codebooks,
            )
        )
        self.decoder = SpeechDecoderConvNet(codebooks=self.cfg.codebooks)

    def forward(self, text_ids: torch.LongTensor, codec_ids: torch.LongTensor) -> torch.Tensor:
        return self.lm(text_ids, codec_ids)

    def synthesize_from_codes(self, codec_ids: torch.LongTensor) -> torch.Tensor:
        return self.decoder(codec_ids)

    def create_voice_clone_prompt(self, ref_codes: torch.LongTensor) -> dict:
        return {"ref_code": ref_codes}

    def sample_codec(self, text_ids: torch.LongTensor, prompt_codec: Optional[torch.LongTensor] = None) -> torch.LongTensor:
        bsz, tlen = text_ids.shape
        if prompt_codec is None:
            prompt_codec = torch.zeros((bsz, tlen, self.cfg.codebooks), dtype=torch.long, device=text_ids.device)
        logits = self.lm(text_ids, prompt_codec[..., 0])
        ids = torch.argmax(logits, dim=-1)
        codec = torch.stack([(ids + i * 11) % self.cfg.vocab_size for i in range(self.cfg.codebooks)], dim=-1)
        return codec

    def decode_to_numpy(self, codec: torch.LongTensor) -> List[torch.Tensor]:
        wav = self.decoder(codec.float())
        return [w for w in wav]
