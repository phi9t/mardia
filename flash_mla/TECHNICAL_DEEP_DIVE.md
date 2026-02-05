# FlashMLA: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/FlashMLA.git  
> Commit: `48c6dc426f045cb7743b18f5c7329f35f1b7ed79` (2026-01-21)

## Overview

FlashMLA provides optimized attention kernels for Multi-head Latent Attention (MLA), the attention mechanism used in DeepSeek-V2/V3. Achieves 660 TFLOPS for dense decoding and supports sparse attention (DSA) for V3.2-Exp.

## Why MLA is Different

### Standard Attention vs MLA

```
Standard MHA:
KV Cache = [batch, seq, num_heads, head_dim] × 2
         = 1 × 128K × 128 × 128 × 2 = 4GB per batch!

MLA:
KV Cache = [batch, seq, latent_dim + rope_dim]
         = 1 × 128K × (512 + 64) = 73MB per batch!
         
93.3% reduction!
```

### Why MLA Decoding is Compute-Bound

```python
# Arithmetic intensity analysis
FLOPs = 2 × h_q × s_q × s_k × (d_k + d_v)
Memory = 2 × s_k × d_k  # Read KV cache

# For DeepSeek (h_q=128 query heads)
Intensity = FLOPs / Memory ≈ 2 × 128 = 256

# H800 compute/memory ratio threshold ≈ 128
# 256 >> 128 → Compute-bound!
```

This is **unusual** - standard attention decoding is memory-bound. FlashMLA exploits this with specialized compute-optimized kernels.

## Performance

### Dense MLA

| Kernel | Performance | Bound |
|--------|-------------|-------|
| Dense Decoding | 660 TFLOPS | Compute |
| Dense Decoding | 3000 GB/s | Memory (alt config) |

### Sparse MLA (DSA)

| Kernel | H800 | B200 |
|--------|------|------|
| Sparse Prefill | 640 TFLOPS | 1450 TFLOPS |
| Sparse Decode (FP8) | 410 TFLOPS | - |

## API Reference

### Metadata Computation

```python
from flash_mla import get_mla_metadata

# Compute tile scheduler metadata (once per batch shape)
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,           # [batch] sequence lengths
    s_q * h_q // h_kv,       # Queries per KV head
    h_kv,                    # Number of KV heads
    h_q,                     # Number of query heads
    is_fp8=True,             # FP8 KV cache?
    topk=None                # For sparse attention
)
```

### Dense MLA with KV Cache

```python
from flash_mla import flash_mla_with_kvcache

# Decoding with cached KV
output, log_sum_exp = flash_mla_with_kvcache(
    q,                       # [batch, seq_q, num_heads, head_dim]
    kvcache,                 # [num_blocks, block_size, latent_dim]
    block_table,             # [batch, max_blocks]
    cache_seqlens,           # [batch]
    dv,                      # Value dimension
    tile_scheduler_metadata,
    num_splits,
    is_causal=True,
    is_fp8_kvcache=True,
    indices=None             # For sparse attention
)
```

### Sparse MLA (DSA)

```python
# Sparse prefill
output = flash_mla.flash_mla_sparse_fwd(
    q,                       # [batch, seq_q, num_heads, head_dim]
    kv,                      # [batch, seq_kv, latent_dim]
    indices,                 # [batch, seq_q, topk] - selected KV indices
    sm_scale                 # Softmax scale
)

# Sparse decode with indices from DSA indexer
output, lse = flash_mla_with_kvcache(
    q, kvcache, block_table, cache_seqlens, dv,
    tile_scheduler_metadata, num_splits,
    is_causal=True,
    is_fp8_kvcache=True,
    indices=indices          # Sparse attention indices
)
```

## KV Cache Format

### BF16 Layout

```python
# Per token: 576 elements
kv_cache = torch.zeros(
    num_blocks,
    block_size,
    latent_dim + rope_dim,  # 512 + 64 = 576
    dtype=torch.bfloat16
)
```

### FP8 Layout (Quantized)

```python
# Per token: 656 bytes
# ├── NoPE part:   512 bytes (512 × FP8_E4M3)
# ├── Scale:       16 bytes (4 × float32)  
# └── RoPE part:   128 bytes (64 × bfloat16)

kv_cache_fp8 = torch.zeros(
    num_blocks,
    block_size,
    656 // element_size,
    dtype=torch.uint8  # Raw bytes
)
```

## Kernel Architecture

### Seesaw Scheduling

Standard ping-pong requires 2 output matrices. MLA's 64×512 output uses 32,768 registers (half of SM's 65,536). Solution: "Seesaw" with 2 warpgroups sharing one split output:

```
Output matrix O (64×512) split:
├── O_L (64×256) → Warpgroup 0
└── O_R (64×256) → Warpgroup 1

Schedule:
1. WG0: Compute P0 = Q @ K0.T     (Tensor Core)
2. WG1: Compute P1 = Q @ K1.T     (Tensor Core)
3. WG0: Softmax + accumulate O_L  (CUDA Core)
4. WG1: Softmax + accumulate O_R  (CUDA Core)
   (CUDA Core and Tensor Core interleaved!)
```

### Fine-Grained TMA Pipelining

```
┌─────────────────────────────────────────────────────────────┐
│                 TMA → GEMM Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 0: [TMA K0] ─────────────────────────────────────►   │
│  Stage 1:          [GEMM K0] [TMA K1] ──────────────────►   │
│  Stage 2:                    [GEMM K1] [TMA K2] ────────►   │
│  Stage 3:                              [GEMM K2] [TMA K3]   │
│                                                              │
│  64×64 blocks for fine-grained overlap                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Cache Hints

```cpp
// L2 cache policy for better hit rate
tma_load<EVICT_FIRST>(k_smem, k_gmem, ...);
// EVICT_FIRST: Deprioritize in L2 (streaming access pattern)
```

### Programmatic Dependent Launch

```cpp
// Overlap splitkv kernel with combine kernel
__global__ void splitkv_kernel(...) {
    // Compute partial results
    ...
    
    // Launch combine kernel from device
    if (threadIdx.x == 0 && is_last_split) {
        cudaLaunchDevice(combine_kernel, ...);
    }
}
```

## Build & Installation

### Requirements

- CUDA 12.3+ (SM90 Hopper)
- Python 3.8+
- PyTorch 2.1+

### Installation

```bash
pip install flash-mla

# Or from source
git clone https://github.com/deepseek-ai/FlashMLA
cd FlashMLA
pip install -e .
```

## Integration with Inference Frameworks

### SGLang

```python
# SGLang uses FlashMLA automatically for DeepSeek models
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8
```

### vLLM

```python
# vLLM integration
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V2-Chat",
    trust_remote_code=True,  # Enables custom attention
)
```

## Debugging & Profiling

### Verify Correctness

```python
# Compare against reference implementation
from flash_mla.testing import verify_mla_attention

verify_mla_attention(
    q, kv_cache, block_table, cache_seqlens,
    atol=1e-3, rtol=1e-3
)
```

### Profile Kernel

```bash
# NSight Compute profiling
ncu --set full python -c "
import flash_mla
# ... kernel call
"
```

## Key Files

| File | Purpose |
|------|---------|
| `flash_mla/__init__.py` | Python API |
| `csrc/flash_mla/` | CUDA kernel source |
| `docs/20250422-new-kernel-deep-dive.md` | Kernel design deep dive |
| `tests/` | Unit tests |
| `benchmarks/` | Performance benchmarks |
