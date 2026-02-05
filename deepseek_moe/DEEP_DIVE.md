# DeepSeekMoE: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-MoE.git  
> Commit: `66edeee5a4f75cbd76e0316229ad101805a90e01` (2024-01-16)

## Overview

DeepSeekMoE introduces a novel Mixture-of-Experts architecture that addresses two fundamental limitations of traditional MoE approaches: **knowledge hybridity** and **knowledge redundancy**.

## Core Innovations

### 1. Fine-Grained Expert Segmentation

Traditional MoE architectures (e.g., GShard) use a small number of large experts (typically 8-16) with top-K routing. This leads to:

- **Knowledge Hybridity**: Each expert must handle diverse, unrelated knowledge
- **Limited Flexibility**: C(16,2) = 120 possible expert combinations with top-2

DeepSeekMoE's solution:

```
Traditional:  16 experts × top-2 → 120 combinations
DeepSeekMoE:  64 experts × top-8 → 4,426,165,368 combinations
```

By splitting each expert FFN into `m` smaller experts while activating `m×K` experts, the model gains exponentially more flexibility in routing decisions.

### 2. Shared Expert Isolation

Some knowledge is universally needed across contexts (common patterns, syntax, etc.). Traditional MoE wastes capacity by learning this redundantly across experts.

DeepSeekMoE dedicates `K_s` experts as **always-on shared experts**:

```python
# DeepSeekMoE Layer Forward
def forward(self, x):
    # Shared experts (always activated)
    shared_out = sum(shared_ffn(x) for shared_ffn in self.shared_experts)
    
    # Routed experts (top-K selection)
    scores = self.router(x)
    topk_experts = scores.topk(self.num_activated)
    routed_out = sum(w * expert(x) for w, expert in topk_experts)
    
    return x + shared_out + routed_out
```

Optimal ratio found: **1 shared : 3 activated routed**

## Architecture Configurations

| Scale | Layers | Hidden | Total Experts | Activated | Expert FFN Ratio | Params |
|-------|--------|--------|---------------|-----------|------------------|--------|
| 2B | 9 | 1280 | 1+63 | 1+7 | 0.25× | 2.0B |
| 16B | 28 | 2048 | 2+64 | 2+6 | 0.25× | 16.4B |
| 145B | 62 | 4096 | 4+128 | 4+12 | 0.125× | 144.6B |

## Load Balancing

Two-level balance loss prevents routing collapse:

### Expert-Level Balance

```python
# Prevents all tokens going to few experts
L_exp = α₁ × Σ(f_i × P_i)
# f_i = fraction of tokens selecting expert i
# P_i = mean routing probability for expert i
# α₁ = 0.01 (2B) or 0.001 (16B)
```

### Device-Level Balance

For distributed training with expert parallelism:

```python
L_dev = α₂ × Σ(f'_i × P'_i)
# Aggregated at device level
# α₂ = 0.05 (145B)
```

## Key Results

| Comparison | Equivalent To | Compute Savings |
|------------|---------------|-----------------|
| DeepSeekMoE 2B | GShard 2.9B | 1.5× expert params |
| DeepSeekMoE 16B | LLaMA2 7B | **60% less compute** |
| DeepSeekMoE 145B | DeepSeek 67B | **71.5% less compute** |

## Fine-Tuning Guide

### Full Fine-Tuning (8×A100 40GB)

```bash
deepspeed finetune/finetune.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --data_path /path/to/data.json \
    --output_dir ./output \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5
```

### QLoRA (1×A100 80GB)

```bash
deepspeed finetune/finetune.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --data_path /path/to/data.json \
    --output_dir ./output \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True \
    --use_lora True \
    --bits 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --quant_type nf4
```

### Data Format

```json
{
    "instruction": "Write a function to calculate factorial",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
}
```

## Connection to Later Work

DeepSeekMoE's innovations directly influenced:

- **DeepSeek-V2**: Combined with Multi-head Latent Attention (MLA)
- **DeepSeek-V3**: Added auxiliary-loss-free load balancing, FP8 training
- **DeepSeek-Coder-V2**: Applied to code domain with 128K context

## Key Files

| File | Purpose |
|------|---------|
| `finetune/finetune.py` | Main fine-tuning script |
| `finetune/configs/ds_config_zero3.json` | DeepSpeed ZeRO-3 config |
| `DeepSeekMoE.pdf` | Technical paper |
