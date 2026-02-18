from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen3_vl.architecture.vl_model import Qwen3VLEducationalModel


def text_to_ids(text: str, vocab: int, length: int) -> torch.LongTensor:
    seq = [ord(c) % vocab for c in text[:length]] or [0]
    out = torch.zeros(length, dtype=torch.long)
    out[: len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    data = json.loads(Path(args.train_json).read_text(encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3VLEducationalModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        total = 0.0
        for sample in data:
            user = next(x for x in sample["conversations"] if x["from"] == "human")["value"]
            assistant = next(x for x in sample["conversations"] if x["from"] == "assistant")["value"]

            text_ids = text_to_ids(user, model.cfg.vocab_size, args.seq_len).unsqueeze(0).to(device)
            labels = text_to_ids(assistant, model.cfg.vocab_size, args.seq_len).unsqueeze(0).to(device)
            image = torch.randn(1, 3, model.cfg.image_size, model.cfg.image_size, device=device)

            logits = model(text_ids, image)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={ep} loss={total / max(len(data), 1):.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "educational_qwen3_vl.pt")


if __name__ == "__main__":
    main()
