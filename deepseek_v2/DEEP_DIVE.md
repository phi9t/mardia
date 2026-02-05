# DeepSeek-V2: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-V2.git  
> Commit: `ec98ee3cbffc32104cd55dba8af884b3d772602a` (2024-09-25)

## Overview

DeepSeek-V2 introduces **Multi-head Latent Attention (MLA)**, achieving a 93.3% reduction in KV cache while maintaining model quality. Combined with DeepSeekMoE, it delivers 42.5% training cost reduction compared to DeepSeek 67B.

## Model Variants

| Variant | Total Params | Activated | Context | KV Cache/Token |
|---------|-------------|-----------|---------|----------------|
| V2-Lite | 16B | 2.4B | 32K | 576 elements |
| V2 | 236B | 21B | 128K | 576 elements |

## Multi-head Latent Attention (MLA)

### The KV Cache Problem

Standard Multi-Head Attention requires storing K and V for every token:

```
KV_cache = batch × seq_len × num_heads × head_dim × 2
         = 1 × 128K × 128 × 128 × 2
         ≈ 4GB per batch element!
```

### MLA Solution: Low-Rank Compression

Instead of caching full K, V tensors, MLA compresses them into a shared latent:

```python
# Standard MHA
K = X @ W_k    # [seq, heads, head_dim]
V = X @ W_v    # [seq, heads, head_dim]

# MLA
latent = X @ W_down    # [seq, latent_dim] - MUCH smaller!
K = latent @ W_up_k    # Reconstruct on-the-fly
V = latent @ W_up_v    # Reconstruct on-the-fly
```

### V2 Configuration

```python
d_latent = 512       # Compressed KV dimension
d_rope = 64          # RoPE dimension (not compressed)
KV_per_token = 512 + 64 = 576 elements

# 93.3% reduction vs standard attention!
```

### Memory Layout (FP8 KV Cache)

```
Per token: 656 bytes total
├── NoPE part:   512 bytes (512 × FP8_E4M3)
├── Scale:       16 bytes (4 × float32)
└── RoPE part:   128 bytes (64 × bfloat16)
```

## Why MLA Decoding is Compute-Bound

Unlike standard attention (memory-bound), MLA decoding is **compute-bound**:

```
Compute-to-Memory Ratio Analysis:
FLOPs = 2 × h_q × s_q × s_k × (d_k + d_v)
Memory = 2 × s_k × d_k

For h_q=128 (DeepSeek):
Ratio = 256 >> H800 threshold of 128
→ Compute-bound!
```

This enables specialized kernel optimizations (see `flash_mla/`).

## Architecture Details

### Attention Layer

```python
class MLAAttention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(hidden, q_lora_rank)
        self.q_nope_proj = nn.Linear(q_lora_rank, n_heads * qk_nope_dim)
        self.q_rope_proj = nn.Linear(q_lora_rank, n_heads * qk_rope_dim)
        
        self.kv_proj = nn.Linear(hidden, kv_lora_rank + qk_rope_dim)
        self.kv_up_proj = nn.Linear(kv_lora_rank, n_heads * (qk_nope_dim + v_dim))
        
    def forward(self, x, kv_cache=None):
        # Query: decompress from latent
        q_latent = self.q_proj(x)
        q_nope = self.q_nope_proj(q_latent)
        q_rope = self.q_rope_proj(q_latent)
        
        # KV: compress to latent
        kv = self.kv_proj(x)
        kv_latent, k_rope = kv.split([kv_lora_rank, qk_rope_dim], dim=-1)
        
        # Cache only the compressed latent!
        if kv_cache is not None:
            kv_cache.update(kv_latent, k_rope)
        
        # Decompress for attention
        kv_decompressed = self.kv_up_proj(kv_latent)
        # ... attention computation
```

## Inference Options

### SGLang (Recommended)

```bash
# BF16, TP=8
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V2-Chat \
    --tp 8

# FP8 with FP8 KV cache
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V2-Chat \
    --tp 8 \
    --quant fp8 \
    --kv-cache-dtype fp8_e5m2
```

### vLLM

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Chat",
    tensor_parallel_size=8,
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True
)
```

## Performance

| Benchmark | V2-Lite | V2 |
|-----------|---------|-----|
| MMLU | 64.3 | 78.5 |
| HumanEval | 57.3 | 81.1 |
| MATH | 17.1 | 43.6 |
| BBH | 53.0 | 78.9 |

## Connection to Infrastructure

### FlashMLA (`flash_mla/`)

Optimized kernels for MLA attention:
- Dense decoding: 660 TFLOPS
- Sparse decoding: 410 TFLOPS (with DSA)
- Seesaw scheduling for compute-bound workloads

### DeepEP (`deep_ep/`)

Expert parallelism for the MoE layers:
- NVLink: 153 GB/s intranode
- RDMA: 58 GB/s internode

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Usage guide |
| `evaluation/` | Benchmark scripts |
