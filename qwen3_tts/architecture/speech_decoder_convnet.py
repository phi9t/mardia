from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        self.left_pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.left_pad, 0))
        return super().forward(x)


class SpeechDecoderConvNet(nn.Module):
    """PaperRef: lightweight causal ConvNet decode path for low-latency waveform reconstruction."""

    def __init__(self, codebooks: int = 16, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(codebooks, hidden, 3),
            nn.GELU(),
            CausalConv1d(hidden, hidden, 3, dilation=2),
            nn.GELU(),
            CausalConv1d(hidden, 1, 3, dilation=4),
            nn.Tanh(),
        )

    def forward(self, codec_seq: torch.Tensor) -> torch.Tensor:
        # codec_seq: [B, T, Q] -> [B, Q, T]
        x = codec_seq.transpose(1, 2)
        wav = self.net(x).squeeze(1)
        return wav
