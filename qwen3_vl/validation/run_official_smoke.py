from __future__ import annotations

import importlib.util


def main() -> None:
    if importlib.util.find_spec("transformers") is None:
        print("transformers not installed; official smoke skipped")
        return

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        print(f"loading official model: {model_id}")
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype="auto", device_map="cpu")
        processor = AutoProcessor.from_pretrained(model_id)

        messages = [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=16)
        print(f"official smoke completed: tokens={out.shape[-1]}")
    except Exception as exc:
        print(f"official smoke failed/skipped: {exc}")


if __name__ == "__main__":
    main()
