# DeepSeek-V3: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-V3.git  
> Commit: `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` (2025-08-28)

## Overview

DeepSeek-V3 is a 671B parameter MoE model (37B activated) that achieves GPT-4 level performance while being trained for only **$5.576M** on 14.8T tokens. Key innovations: auxiliary-loss-free load balancing, Multi-Token Prediction (MTP), and validated FP8 mixed-precision training at scale.

## Model Specifications

| Metric | Value |
|--------|-------|
| Total Parameters | 671B (+14B MTP = 685B on HF) |
| Activated Parameters | 37B per token |
| Training Tokens | 14.8T |
| Training Cost | 2.788M H800 GPU hours |
| Context Length | 128K |
| Architecture | MLA + DeepSeekMoE |

## Key Innovations

### 1. Auxiliary-Loss-Free Load Balancing

Traditional MoE uses auxiliary loss to prevent routing collapse, but this degrades model quality.

V3's solution: **Bias-based dynamic balancing**

```python
# Traditional (hurts quality)
loss = lm_loss + α × auxiliary_balance_loss

# V3 (no quality penalty)
def route(x):
    scores = router(x)
    # Dynamic bias adjustment per expert
    scores = scores + self.expert_bias  
    selected = scores.topk(k)
    
    # Update bias based on load (separate from loss)
    with torch.no_grad():
        load = compute_expert_load(selected)
        self.expert_bias -= γ * (load - target_load)
    
    return selected
```

### 2. Multi-Token Prediction (MTP)

Instead of predicting just the next token, V3 predicts multiple future tokens:

```
Standard LM:   P(t_{n+1} | t_1...t_n)
MTP:           P(t_{n+1}, t_{n+2}, ..., t_{n+k} | t_1...t_n)
```

**Training benefit**: Stronger gradients from multi-token supervision

**Inference benefit**: Speculative decoding with 1.5-2× throughput

```python
# MTP for speculative decoding
def generate_with_mtp(prompt, k=4):
    while not done:
        # Draft k tokens using MTP head
        drafts = mtp_head(hidden_state)  # [k tokens]
        
        # Verify in parallel with main model
        verified = verify_batch(drafts)
        
        # Accept correct prefix, reject rest
        output.extend(verified)
```

### 3. FP8 Mixed Precision Training

First validation of FP8 training at 671B scale!

```python
# FP8 Formats
E4M3:  # 4-bit exponent, 3-bit mantissa
       # Range: [-448, 448]
       # Used for: weights, activations (forward)

E5M2:  # 5-bit exponent, 2-bit mantissa  
       # Range: [-57344, 57344]
       # Used for: gradients (backward)
```

**What stays in higher precision**:
- Attention softmax
- Layer normalization
- Loss computation
- Optimizer states (FP32)

**Training stability**: No irrecoverable loss spikes, no rollbacks needed!

## Training Infrastructure

### DualPipe (`dualpipe/`)

Bidirectional pipeline parallelism:

```
PP=8, micro-batches=20

Standard 1F1B bubble: (PP-1)(F+B) = 7 × 2 = 14 slots
DualPipe bubble:      (PP/2-1)(F&B+B-3W) ≈ 3 slots

→ 78% bubble reduction!
```

Key: Run forward and backward in opposite directions simultaneously.

### DeepGEMM (`deep_gemm/`)

FP8 GEMM kernels:

```python
import deep_gemm

# Normal FP8 GEMM
deep_gemm.fp8_gemm_nt(A, B, C, D, scale_a, scale_b)

# MoE grouped GEMM (contiguous layout for prefill)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(...)

# MoE grouped GEMM (masked layout for CUDA graph)
deep_gemm.m_grouped_fp8_gemm_nt_masked(...)
```

Performance: Up to **1550 TFLOPS** on H800

### DeepEP (`deep_ep/`)

Expert parallelism communication:

| Operation | EP=8 | EP=64 |
|-----------|------|-------|
| Dispatch | 153 GB/s (NVLink) | 51 GB/s (RDMA) |
| Combine | 158 GB/s (NVLink) | 50 GB/s (RDMA) |

## Inference

### Hardware Requirements

- **BF16**: 8× 80GB GPUs minimum
- **FP8**: Can fit on fewer GPUs

### SGLang (Recommended)

```bash
# BF16, tensor parallelism = 8
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8

# FP8 with FP8 KV cache
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --quant fp8 \
    --kv-cache-dtype fp8_e5m2
```

### FP8→BF16 Weight Conversion

```bash
cd inference
python fp8_cast_bf16.py \
    --input-fp8-hf-path /path/to/fp8_weights \
    --output-bf16-hf-path /path/to/bf16_weights
```

## Benchmark Results

| Benchmark | V3 | GPT-4 | Claude-3.5 |
|-----------|-----|-------|------------|
| MMLU | 87.1 | 86.4 | 88.3 |
| HumanEval | 82.6 | 67.0 | 92.0 |
| MATH | 52.5 | 52.9 | 71.1 |
| Codeforces | 51.6 | 23.0 | 20.3 |

## Key Files

| File | Purpose |
|------|---------|
| `inference/model.py` | Model architecture |
| `inference/generate.py` | Generation script |
| `inference/fp8_cast_bf16.py` | Weight conversion |
| `inference/configs/` | Model configurations |
