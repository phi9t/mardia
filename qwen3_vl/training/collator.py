from __future__ import annotations

from typing import Dict, List

import torch


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    text_ids = torch.stack([x["text_ids"] for x in batch], dim=0)
    images = torch.stack([x["image"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    return {"text_ids": text_ids, "images": images, "labels": labels}
