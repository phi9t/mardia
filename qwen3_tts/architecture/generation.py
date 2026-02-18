from __future__ import annotations

from typing import List, Optional

import torch

from .model import Qwen3TTSEducationalModel


def _text_to_ids(texts: List[str], vocab_size: int, device: str) -> torch.LongTensor:
    max_len = max(max(len(t), 1) for t in texts)
    ids = torch.zeros((len(texts), max_len), dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        seq = [ord(c) % vocab_size for c in t[:max_len]] or [0]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return ids


def generate_custom_voice(model: Qwen3TTSEducationalModel, texts: List[str], device: str = "cpu") -> List[torch.Tensor]:
    text_ids = _text_to_ids(texts, model.cfg.vocab_size, device)
    codec = model.sample_codec(text_ids)
    return model.decode_to_numpy(codec)


def generate_voice_design(
    model: Qwen3TTSEducationalModel,
    texts: List[str],
    instructs: Optional[List[str]] = None,
    device: str = "cpu",
) -> List[torch.Tensor]:
    instructs = instructs or ["" for _ in texts]
    mixed = [f"{i} {t}" for i, t in zip(instructs, texts)]
    text_ids = _text_to_ids(mixed, model.cfg.vocab_size, device)
    codec = model.sample_codec(text_ids)
    return model.decode_to_numpy(codec)


def generate_voice_clone(
    model: Qwen3TTSEducationalModel,
    texts: List[str],
    voice_clone_prompt: Optional[dict] = None,
    device: str = "cpu",
) -> List[torch.Tensor]:
    text_ids = _text_to_ids(texts, model.cfg.vocab_size, device)
    prompt = None
    if voice_clone_prompt and "ref_code" in voice_clone_prompt:
        ref = voice_clone_prompt["ref_code"].to(device)
        if ref.dim() == 2:
            ref = ref.unsqueeze(0)
        prompt = ref[:, : text_ids.shape[1], :]
    codec = model.sample_codec(text_ids, prompt_codec=prompt)
    return model.decode_to_numpy(codec)
