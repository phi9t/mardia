from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from qwen3_tts.architecture.tokenizer_12hz import Tokenizer12Hz

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


def load_wav(path: str, fallback_sr: int = 24000) -> np.ndarray:
    if sf is None:
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        return rng.standard_normal(fallback_sr).astype(np.float32) * 0.01
    wav, _sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    args = ap.parse_args()

    tok = Tokenizer12Hz()

    rows: List[Dict[str, Any]] = []
    with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    with Path(args.output_jsonl).open("w", encoding="utf-8") as out:
        for row in rows:
            wav = load_wav(row["audio"])
            enc = tok.encode([wav]).audio_codes[0].tolist()
            row["audio_codes"] = enc
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
