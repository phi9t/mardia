from __future__ import annotations

from typing import List

import torch

from .vl_model import Qwen3VLEducationalModel


def _text_to_ids(texts: List[str], vocab_size: int, device: str) -> torch.LongTensor:
    max_len = max(max(len(t), 1) for t in texts)
    ids = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        seq = [ord(c) % vocab_size for c in t[:max_len]] or [0]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return ids


def generate_multimodal(model: Qwen3VLEducationalModel, prompts: List[str], images: torch.Tensor, device: str = "cpu") -> List[str]:
    text_ids = _text_to_ids(prompts, model.cfg.vocab_size, device)
    logits = model(text_ids, images.to(device))
    pred = torch.argmax(logits, dim=-1)
    outs = []
    for row in pred:
        s = "".join(chr(int(x.item()) % 95 + 32) for x in row[:64])
        outs.append(s)
    return outs
