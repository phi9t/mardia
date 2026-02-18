from __future__ import annotations

import torch

from qwen3_vl.architecture.vl_model import Qwen3VLEducationalModel


def main() -> None:
    model = Qwen3VLEducationalModel()
    text_ids = torch.randint(0, model.cfg.vocab_size, (2, 32))
    images = torch.randn(2, 3, model.cfg.image_size, model.cfg.image_size)
    out = model(text_ids, images)
    assert out.shape == (2, 32, model.cfg.vocab_size)
    print("qwen3_vl structural check passed")


if __name__ == "__main__":
    main()
