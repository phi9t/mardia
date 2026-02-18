from __future__ import annotations

import importlib.util


def main() -> None:
    if importlib.util.find_spec("qwen_tts") is None:
        print("qwen_tts is not installed; official smoke skipped")
        return

    try:
        from qwen_tts import Qwen3TTSModel

        model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        print(f"loading official model: {model_id}")
        model = Qwen3TTSModel.from_pretrained(model_id, device_map="cpu")
        wavs, sr = model.generate_custom_voice(text="hello", language="en", speaker="Cherry")
        print(f"official smoke completed: n_wavs={len(wavs)} sr={sr}")
    except Exception as exc:
        print(f"official smoke failed/skipped: {exc}")


if __name__ == "__main__":
    main()
