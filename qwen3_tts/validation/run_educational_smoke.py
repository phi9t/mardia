from __future__ import annotations

from qwen3_tts.architecture.generation import generate_custom_voice, generate_voice_clone, generate_voice_design
from qwen3_tts.architecture.model import Qwen3TTSEducationalModel


def main() -> None:
    model = Qwen3TTSEducationalModel()

    w1 = generate_custom_voice(model, ["hello world"])
    w2 = generate_voice_design(model, ["this is a design"], ["happy tone"])
    prompt = model.create_voice_clone_prompt(model.sample_codec(__import__("torch").randint(0, 128, (1, 16))))
    w3 = generate_voice_clone(model, ["clone this"], prompt)

    assert len(w1) == 1 and len(w2) == 1 and len(w3) == 1
    print("qwen3_tts educational smoke passed")


if __name__ == "__main__":
    main()
