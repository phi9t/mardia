from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lmms-lab/DocVQA")
    ap.add_argument("--split", default="train[:16]")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    records = []
    for row in ds:
        img = row.get("image")
        img_path = ""
        if isinstance(img, dict):
            img_path = img.get("path") or ""
        q = row.get("question") or row.get("query") or "Describe this image."
        a = row.get("answers", ["N/A"])
        ans = a[0] if isinstance(a, list) and a else str(a)
        records.append(
            {
                "image": [img_path] if img_path else ["/tmp/placeholder.jpg"],
                "video": [],
                "conversations": [
                    {"from": "human", "value": "<image> " + q},
                    {"from": "assistant", "value": ans},
                ],
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
