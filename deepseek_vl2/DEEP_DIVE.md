# DeepSeek-VL2: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-VL2.git  
> Commit: `ef9f91e2b6426536b83294c11742c27be66361b1` (2025-02-26)

## Overview

DeepSeek-VL2 is an MoE-based Vision-Language model for advanced multimodal understanding, featuring visual grounding, multi-image conversations, and efficient incremental prefilling.

## Model Variants

| Variant | Total Params | Activated | Vision Encoder | LLM |
|---------|-------------|-----------|----------------|-----|
| VL2-Tiny | 3.37B | 1B | SigLIP-400M | DeepSeek-MoE |
| VL2-Small | 16.1B | 2.4B | SigLIP-400M | DeepSeek-MoE |
| VL2 | 27.5B | 4.2B | SigLIP-400M | DeepSeek-MoE |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepSeek-VL2                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Images ──► Vision Encoder ──► Projector ──┐               │
│              (SigLIP-400M)      (MLP)       │               │
│                                              ▼               │
│   Text ──────────────────────────────► MoE Language Model   │
│                                              │               │
│                                              ▼               │
│                                         Response            │
│                                     [+ Grounding Boxes]     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Vision Encoder

- **Model**: SigLIP-400M (ViT-SO400M)
- **Resolution**: Dynamic (multiple scales)
- **Patch size**: 14×14
- **Output**: Visual tokens per image

### MLP Projector

```python
class VisionProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, vision_features):
        return self.proj(vision_features)
```

## Key Capabilities

### 1. Multi-Image Conversations

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "photo1.jpg"},
            {"type": "image", "image": "photo2.jpg"},
            {"type": "text", "text": "Compare these two images"}
        ]
    }
]
response = model.chat(messages)
```

### 2. Visual Grounding

Identify and locate objects with bounding boxes:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "scene.jpg"},
            {"type": "text", "text": "<|ref|>Find all the cars<|/ref|>"}
        ]
    }
]
# Response includes bounding box coordinates
# "The cars are at <|box_start|>(0.12,0.34,0.45,0.67)<|box_end|>, ..."
```

### 3. Document Understanding

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "invoice.pdf"},  # PDF page as image
            {"type": "text", "text": "Extract the total amount and date"}
        ]
    }
]
```

### 4. OCR Capabilities

Built-in text recognition for:
- Scene text
- Documents
- Handwriting
- Multiple languages

## Incremental Prefilling

For memory-constrained deployments (e.g., 40GB GPU):

```python
# Instead of processing all images at once
# Process in chunks to reduce peak memory

model.incremental_prefill(
    images=images,
    chunk_size=1024,  # Tokens per chunk
    device="cuda"
)

# Then generate normally
response = model.generate(prompt)
```

### How It Works

```
Standard Prefill:
[img1_tokens + img2_tokens + img3_tokens + text] → Forward → KV Cache

Incremental Prefill:
[img1_tokens] → Forward → KV Cache (partial)
[img2_tokens] → Forward → KV Cache (partial)
[img3_tokens + text] → Forward → KV Cache (complete)
```

## Usage

### Installation

```bash
pip install -e .
# Or with Flash Attention
pip install -e .[flash_attn]
```

### Basic Usage

```python
from deepseek_vl2 import DeepSeekVL2ForCausalLM, DeepSeekVL2Processor

model = DeepSeekVL2ForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-vl2",
    torch_dtype=torch.bfloat16
).cuda()
processor = DeepSeekVL2Processor.from_pretrained("deepseek-ai/deepseek-vl2")

# Prepare inputs
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "example.jpg"},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

inputs = processor(conversation, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0])
```

### Grounding Example

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "street.jpg"},
            {"type": "text", "text": "<|ref|>Point to all traffic lights<|/ref|>"}
        ]
    }
]

response = model.chat(conversation)
# Parse bounding boxes from response
boxes = parse_grounding_boxes(response)
```

## Benchmark Results

| Benchmark | VL2-Tiny | VL2-Small | VL2 |
|-----------|----------|-----------|-----|
| DocVQA | 86.9 | 91.6 | 93.4 |
| ChartQA | 76.0 | 80.6 | 84.5 |
| TextVQA | 74.1 | 79.5 | 82.3 |
| OCRBench | 72.4 | 80.8 | 83.9 |
| RealWorldQA | 58.7 | 62.1 | 65.4 |

## Deployment

### Transformers

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-vl2",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
```

### vLLM

```bash
vllm serve deepseek-ai/deepseek-vl2-small \
    --tensor-parallel-size 2 \
    --trust-remote-code
```

## Connection to DeepSeek-OCR-2

VL2's architecture influenced OCR-2:
- Both use vision encoder + projector + LLM pattern
- OCR-2 specializes with dual encoder (SAM + Qwen2)
- Both support dynamic resolution

## Key Files

| File | Purpose |
|------|---------|
| `deepseek_vl2/models/` | Model architecture |
| `deepseek_vl2/serve/` | Inference serving |
| `examples/` | Usage examples |
| `README.md` | Documentation |
