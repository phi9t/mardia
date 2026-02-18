from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen3_tts.architecture.model import Qwen3TTSEducationalModel
from qwen3_tts.training.losses import codec_ce_loss


def text_to_ids(text: str, vocab: int, length: int) -> torch.LongTensor:
    seq = [ord(c) % vocab for c in text[:length]]
    if not seq:
        seq = [0]
    out = torch.zeros(length, dtype=torch.long)
    out[: len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq_len", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSEducationalModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rows = [json.loads(x) for x in Path(args.train_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]

    for ep in range(args.epochs):
        total = 0.0
        for row in rows:
            text_ids = text_to_ids(row["text"], model.cfg.vocab_size, args.seq_len).unsqueeze(0).to(device)
            codes = torch.tensor(row["audio_codes"], dtype=torch.long, device=device)
            if codes.size(0) < args.seq_len:
                pad = torch.zeros((args.seq_len - codes.size(0), codes.size(1)), dtype=torch.long, device=device)
                codes = torch.cat([codes, pad], dim=0)
            codes = codes[: args.seq_len].unsqueeze(0)

            logits = model(text_ids, codes[..., 0])
            loss = codec_ce_loss(logits, codes[..., 0])

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={ep} loss={total / max(len(rows), 1):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "educational_qwen3_tts.pt")


if __name__ == "__main__":
    main()
