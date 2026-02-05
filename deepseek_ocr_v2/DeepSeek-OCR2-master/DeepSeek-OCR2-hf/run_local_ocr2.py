import os
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

model_name = "deepseek-ai/DeepSeek-OCR-2"

# Use eager attention to avoid FlashAttention2/SDPA requirements (unsupported here).
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fig1.png"))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ocr2_out"))

os.makedirs(output_path, exist_ok=True)

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=768,
    crop_mode=True,
    save_results=True,
)

print(res)
print(f"Saved outputs to: {output_path}")
