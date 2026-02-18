from __future__ import annotations

import torch

from qwen3_vl.architecture.generation import generate_multimodal
from qwen3_vl.architecture.vl_model import Qwen3VLEducationalModel


def main() -> None:
    model = Qwen3VLEducationalModel()
    images = torch.randn(1, 3, model.cfg.image_size, model.cfg.image_size)
    out = generate_multimodal(model, ["describe this"], images)
    assert len(out) == 1
    print("qwen3_vl educational smoke passed")


if __name__ == "__main__":
    main()
