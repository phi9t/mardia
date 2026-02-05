# DeepGEMM: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepGEMM.git  
> Commit: `477618cd51baffca09c4b0b87e97c03fe827ef03` (2026-02-03)

## Overview

DeepGEMM is a library of optimized FP8/BF16 GEMM kernels with JIT compilation, designed for DeepSeek's MoE inference and training. Achieves up to 1550 TFLOPS on H800.

## Supported Operations

### Standard GEMMs

| Operation | Description | Hardware |
|-----------|-------------|----------|
| `fp8_gemm_nt` | FP8 GEMM (NT layout) | SM90, SM100 |
| `bf16_gemm_nt` | BF16 GEMM (NT layout) | SM90, SM100 |
| `fp8_gemm_1d2d` | FP8 with 1D/2D scales | SM90 |

### Grouped GEMMs (for MoE)

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `m_grouped_fp8_gemm_nt_contiguous` | Contiguous layout | Prefill |
| `m_grouped_fp8_gemm_nt_masked` | Masked layout | Decode (CUDA graph) |

### Specialized Kernels

| Operation | Description | Use Case |
|-----------|-------------|----------|
| `fp8_mqa_logits` | MQA logit computation | V3.2-Exp DSA indexer |
| `fp8_paged_mqa_logits` | Paged MQA logits | PagedAttention |
| `clean_logits` | Logit cleaning | Attention masking |
| `layout_transpose` | Layout conversion | Data format changes |

## Performance

### FP8 GEMM (H800)

| M | N | K | TFLOPS |
|---|---|---|--------|
| 4096 | 4096 | 4096 | 1450 |
| 8192 | 8192 | 4096 | 1520 |
| 16384 | 4096 | 4096 | 1550 |

### Grouped GEMM (MoE)

| Experts | Tokens/Expert | Performance |
|---------|---------------|-------------|
| 8 | 256 | ~1400 TFLOPS |
| 64 | 64 | ~1200 TFLOPS |
| 256 | 16 | ~900 TFLOPS |

## API Reference

### Basic FP8 GEMM

```python
import deep_gemm

# D = C + A @ B.T (FP8)
deep_gemm.fp8_gemm_nt(
    A,          # [M, K] FP8
    B,          # [N, K] FP8
    C,          # [M, N] BF16 (optional bias)
    D,          # [M, N] BF16 output
    scale_a,    # [M] or scalar
    scale_b     # [N] or scalar
)
```

### Grouped GEMM (MoE Prefill)

```python
# Contiguous layout for variable-length sequences
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    A,              # [total_tokens, K] FP8
    B,              # [num_experts, N, K] FP8
    C,              # [total_tokens, N] BF16 output
    expert_ids,     # [total_tokens] int32
    m_indices,      # [num_experts+1] cumsum of tokens per expert
    scale_a,        # [total_tokens] or [num_experts]
    scale_b         # [num_experts, N] or [num_experts]
)
```

### Grouped GEMM (MoE Decode with CUDA Graph)

```python
# Masked layout for fixed-size batches (CUDA graph compatible)
deep_gemm.m_grouped_fp8_gemm_nt_masked(
    A,              # [batch, max_tokens, K] FP8
    B,              # [num_experts, N, K] FP8
    C,              # [batch, max_tokens, N] BF16 output
    mask,           # [batch, max_tokens] bool (valid positions)
    expert_ids,     # [batch, max_tokens] int32
    scale_a,
    scale_b
)
```

### DSA Indexer Logits

```python
# For V3.2-Exp sparse attention indexer
# Computes: out[i,j] = ReLU(q[i] @ kv[j]) * weights[i]
deep_gemm.fp8_mqa_logits(
    q,              # [num_queries, d_k] FP8
    kv,             # [num_kv, d_k] FP8
    weights,        # [num_queries] BF16
    out,            # [num_queries, num_kv] BF16
    cu_seq_len_k_start,  # Cumulative sequence starts
    cu_seq_len_k_end     # Cumulative sequence ends
)
```

## JIT Compilation

### How It Works

```python
# First call compiles kernel (cached)
deep_gemm.fp8_gemm_nt(A, B, C, D, scale_a, scale_b)
# Compilation happens here ^

# Subsequent calls use cached kernel
deep_gemm.fp8_gemm_nt(A, B, C, D, scale_a, scale_b)
# Instant ^
```

### Cache Management

```python
# Cache location
# ~/.cache/deep_gemm/kernels/

# Clear cache
import deep_gemm
deep_gemm.clear_cache()

# Warm up specific shapes
deep_gemm.warmup_shapes([
    (4096, 4096, 4096),
    (8192, 4096, 4096),
])
```

## Architecture Details

### SM90 (Hopper) Features

- **Tensor Memory Accelerator (TMA)**: Async data movement
- **Warpgroup MMA**: 128-thread warpgroup GEMM
- **Thread Block Clusters**: Multi-SM cooperation
- **FP8 Tensor Cores**: Native FP8 support

### SM100 (Blackwell) Features

- **Enhanced TMA**: Higher bandwidth
- **Larger shared memory**: More tiling options
- **Improved FP8**: Better precision modes

### Kernel Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     GEMM Kernel                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Global Memory ──► TMA ──► Shared Memory                   │
│                                    │                         │
│                                    ▼                         │
│                            Register Tiles                    │
│                                    │                         │
│                                    ▼                         │
│                          Tensor Core MMA                     │
│                                    │                         │
│                                    ▼                         │
│                          Accumulator (FP32)                  │
│                                    │                         │
│                                    ▼                         │
│                           Epilogue (scale, add C)           │
│                                    │                         │
│                                    ▼                         │
│                    Shared Memory ──► TMA ──► Global         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Build & Installation

### Requirements

- CUDA 12.3+ (SM90) or CUDA 12.8+ (SM100)
- Python 3.8+
- PyTorch 2.1+
- CMake 3.18+

### Installation

```bash
# Quick install
pip install deep-gemm

# From source
git clone https://github.com/deepseek-ai/DeepGEMM
cd DeepGEMM
pip install -e .
```

### Development

```bash
# Build with debug
./develop.sh

# Run tests
python -m pytest tests/
```

## Integration Examples

### MoE Layer

```python
class MoELayer(nn.Module):
    def forward(self, x, expert_ids, expert_weights):
        # Up projection
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            x, self.up_weight, up_out, expert_ids, m_indices, 
            scale_x, scale_up
        )
        
        # Activation
        hidden = F.silu(up_out)
        
        # Down projection
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            hidden, self.down_weight, out, expert_ids, m_indices,
            scale_hidden, scale_down
        )
        
        # Weighted sum
        return (out * expert_weights.unsqueeze(-1)).sum(dim=1)
```

### DSA Indexer

```python
class DSAIndexer(nn.Module):
    def forward(self, q, kv):
        # Compute relevance logits
        deep_gemm.fp8_mqa_logits(
            q, kv, self.weights, logits,
            cu_seq_start, cu_seq_end
        )
        
        # Select top-k
        indices = logits.topk(self.topk, dim=-1).indices
        return indices
```

## Key Files

| File | Purpose |
|------|---------|
| `deep_gemm/__init__.py` | Python API |
| `csrc/jit/` | JIT compilation infrastructure |
| `csrc/jit_kernels/impls/` | Kernel implementations |
| `csrc/jit_kernels/heuristics/` | Auto-tuning heuristics |
| `tests/` | Unit tests and benchmarks |
