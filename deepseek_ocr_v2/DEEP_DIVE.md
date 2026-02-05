# DeepSeek-OCR-2: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-OCR-2.git  
> Commit: `2f3699ebbb96fa8af32212e8c170f2cc28730fad` (2026-02-03)

## Overview

DeepSeek-OCR-2 is a document understanding model that converts images to structured markdown, featuring a dual-encoder architecture (SAM + Qwen2) with dynamic resolution support.

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepSeek-OCR-2                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image ──► Dynamic Crop ──┬──► Global View (1024×1024)     │
│                            │                                 │
│                            └──► Local Crops (N × 768×768)   │
│                                       │                      │
│                                       ▼                      │
│                               ┌───────────────┐             │
│                               │   SAM ViT-B   │             │
│                               └───────┬───────┘             │
│                                       │                      │
│                                       ▼                      │
│                               ┌───────────────┐             │
│                               │ Qwen2 Encoder │             │
│                               └───────┬───────┘             │
│                                       │                      │
│                                       ▼                      │
│                               ┌───────────────┐             │
│                               │ MLP Projector │             │
│                               └───────┬───────┘             │
│                                       │                      │
│                                       ▼                      │
│                               ┌───────────────┐             │
│                               │  DeepSeek LLM │             │
│                               └───────┬───────┘             │
│                                       │                      │
│                                       ▼                      │
│                              Structured Markdown             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Dual Encoder Design

### SAM ViT-B

```python
from deepencoderv2.sam_vary_sdpa import build_sam_vit_b

# SAM encoder extracts visual features
sam_encoder = build_sam_vit_b()
visual_features = sam_encoder(image_patches)
```

**Purpose**: Extract rich visual features from document images

### Qwen2-as-Encoder

```python
from deepencoderv2.qwen2_d2e import build_qwen2_decoder_as_encoder

# Use Qwen2 decoder as an encoder
qwen2_encoder = build_qwen2_decoder_as_encoder()
encoded_features = qwen2_encoder(visual_features)
```

**Purpose**: Add language-aware processing to visual features

### MLP Projector

```python
from deepencoderv2.build_linear import MlpProjector

projector = MlpProjector(encoder_dim, llm_dim)
projected = projector(encoded_features)
```

## Dynamic Resolution

### Configuration

```python
# From config.py
BASE_SIZE = 1024      # Global view resolution
IMAGE_SIZE = 768      # Local crop resolution
MIN_CROPS = 2
MAX_CROPS = 6         # Maximum local views
CROP_MODE = True      # Enable dynamic cropping
```

### Token Calculation

```python
def get_num_image_tokens(image_width, image_height):
    patch_size = 16
    downsample_ratio = 4
    
    # Global view tokens
    h = w = math.ceil((BASE_SIZE // patch_size) / downsample_ratio)
    global_tokens = h * w  # 256 tokens
    
    # Local crop tokens
    if image_width <= 768 and image_height <= 768:
        num_crops = 1
    else:
        crop_ratio = count_tiles(image_width, image_height)
        num_crops = crop_ratio[0] * crop_ratio[1]
    
    local_h = local_w = math.ceil((IMAGE_SIZE // patch_size) / downsample_ratio)
    local_tokens = num_crops * local_h * local_w  # N × 144 tokens
    
    return global_tokens + local_tokens
```

**Example**:
- Global: 1 × 256 = 256 tokens
- Local (4 crops): 4 × 144 = 576 tokens
- Total: 832 tokens

## Output Format

### Structured Markdown

```markdown
# Document Title

## Section 1: Introduction

This is the introduction text with **bold** and *italic* formatting.

### Subsection 1.1

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

## Section 2: Figures

![Figure 1](description of figure)

<box>(0.12, 0.34, 0.56, 0.78)</box>
```

### Grounding Coordinates

For locating elements:
```
<box>(x1, y1, x2, y2)</box>
```
- Normalized coordinates [0, 1]
- (x1, y1): top-left corner
- (x2, y2): bottom-right corner

## Usage

### Single Image Processing

```bash
python run_dpsk_ocr2_image.py \
    --image_path /path/to/document.png \
    --output_path /path/to/output/
```

### PDF Batch Processing

```bash
python run_dpsk_ocr2_pdf.py \
    --input_path /path/to/document.pdf \
    --output_path /path/to/output/
```

### Python API

```python
from process.image_process import DeepseekOCR2Processor

processor = DeepseekOCR2Processor()

# Process image
result = processor.process_image("document.png")

# Get markdown output
markdown = result["markdown"]
```

## vLLM Integration

```python
# deepseek_ocr2.py provides vLLM-compatible model
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-OCR2",
    trust_remote_code=True
)

# Process with vLLM
outputs = llm.generate(
    prompts=[{"image": "document.png", "prompt": "Convert to markdown"}]
)
```

## Comparison with VL2

| Feature | VL2 | OCR-2 |
|---------|-----|-------|
| Focus | General VLM | Document OCR |
| Encoder | SigLIP | SAM + Qwen2 |
| Output | Natural language | Structured markdown |
| Resolution | Dynamic | Dynamic (optimized for docs) |
| Tables | Basic | Precise structure |

## Key Files

| File | Purpose |
|------|---------|
| `DeepSeek-OCR2-vllm/deepseek_ocr2.py` | vLLM model implementation |
| `DeepSeek-OCR2-vllm/config.py` | Resolution configuration |
| `DeepSeek-OCR2-vllm/process/image_process.py` | Image preprocessing |
| `DeepSeek-OCR2-vllm/deepencoderv2/` | Dual encoder components |
| `DeepSeek_OCR2_paper.pdf` | Technical paper |
| `README.md` | Usage guide |
