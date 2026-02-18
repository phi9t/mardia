from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class EncodedAudio:
    audio_codes: List[torch.LongTensor]


class Tokenizer12Hz:
    """PaperRef: 12Hz discrete speech tokenization interface.

    This educational tokenizer approximates a multi-codebook codec interface.
    It preserves the operational contract used by training/inference code.
    """

    def __init__(self, sample_rate: int = 24000, codebooks: int = 16, bins: int = 1024):
        self.sample_rate = sample_rate
        self.codebooks = codebooks
        self.bins = bins

    def _frame_audio(self, wav: np.ndarray, frame_hz: float = 12.5) -> np.ndarray:
        hop = max(1, int(self.sample_rate / frame_hz))
        n = (len(wav) // hop) * hop
        if n == 0:
            return np.zeros((1, hop), dtype=np.float32)
        framed = wav[:n].reshape(-1, hop)
        return framed

    def encode(self, wavs: Sequence[np.ndarray]) -> EncodedAudio:
        codes: List[torch.LongTensor] = []
        for wav in wavs:
            framed = self._frame_audio(wav)
            energy = np.tanh(np.mean(np.abs(framed), axis=1))
            base = np.clip((energy * (self.bins - 1)).astype(np.int64), 0, self.bins - 1)
            cb = np.stack([(base + i * 7) % self.bins for i in range(self.codebooks)], axis=-1)
            codes.append(torch.from_numpy(cb).long())
        return EncodedAudio(audio_codes=codes)

    def decode(self, encoded: EncodedAudio) -> Tuple[List[np.ndarray], int]:
        out: List[np.ndarray] = []
        hop = int(self.sample_rate / 12.5)
        for code in encoded.audio_codes:
            c = code.float().mean(dim=-1).numpy() / max(self.bins - 1, 1)
            amp = (2.0 * c - 1.0).astype(np.float32)
            wav = np.repeat(amp, hop)
            out.append(wav)
        return out, self.sample_rate
