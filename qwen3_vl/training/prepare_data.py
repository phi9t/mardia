from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_placeholders(sample: dict) -> None:
    text = "\n".join(x["value"] for x in sample.get("conversations", []))
    n_img = text.count("<image>")
    n_vid = text.count("<video>")
    if n_img != len(sample.get("image", [])):
        raise ValueError(f"image placeholder mismatch: {n_img} vs {len(sample.get('image', []))}")
    if n_vid != len(sample.get("video", [])):
        raise ValueError(f"video placeholder mismatch: {n_vid} vs {len(sample.get('video', []))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    samples = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    for s in samples:
        validate_placeholders(s)
    Path(args.output_json).write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validated {len(samples)} samples")


if __name__ == "__main__":
    main()
