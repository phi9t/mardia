from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--image_url", default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except Exception as exc:
        print(f"openai client missing: {exc}")
        return

    client = OpenAI(api_key="EMPTY", base_url=args.base_url)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": args.image_url}},
                    {"type": "text", "text": "Describe this image briefly."},
                ],
            }
        ],
        max_tokens=128,
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
