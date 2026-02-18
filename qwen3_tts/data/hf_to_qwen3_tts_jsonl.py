from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keithito/lj_speech")
    ap.add_argument("--split", default="train[:16]")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            audio_path = row.get("file") or row.get("audio", {}).get("path") or ""
            text = row.get("text") or row.get("sentence") or ""
            obj = {"audio": audio_path, "text": text, "ref_audio": audio_path}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
