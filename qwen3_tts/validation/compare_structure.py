from __future__ import annotations

import torch

from qwen3_tts.architecture.model import Qwen3TTSEducationalModel


def main() -> None:
    model = Qwen3TTSEducationalModel()
    text_ids = torch.randint(0, model.cfg.vocab_size, (2, 32))
    codec_ids = torch.randint(0, model.cfg.vocab_size, (2, 32))
    logits = model(text_ids, codec_ids)
    assert logits.shape == (2, 32, model.cfg.vocab_size)

    sampled = model.sample_codec(text_ids)
    assert sampled.shape == (2, 32, model.cfg.codebooks)

    wavs = model.decode_to_numpy(sampled)
    assert len(wavs) == 2
    print("qwen3_tts structural check passed")


if __name__ == "__main__":
    main()
