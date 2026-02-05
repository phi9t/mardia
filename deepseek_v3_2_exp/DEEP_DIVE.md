# DeepSeek-V3.2-Exp: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp.git  
> Commit: `87e509a2e5a100d221c97df52c6e8be7835f0057` (2025-11-18)

## Overview

DeepSeek-V3.2-Exp introduces **DeepSeek Sparse Attention (DSA)** - the first fine-grained sparse attention mechanism that achieves substantial long-context efficiency improvements without quality degradation.

## Model Specifications

| Metric | Value |
|--------|-------|
| Base Model | V3.1-Terminus |
| Total Parameters | 685B |
| Sparse Attention | DSA (learnable sparsity) |
| Context Length | 128K+ |

## DeepSeek Sparse Attention (DSA)

### Dense vs Sparse Attention

```python
# Dense Attention (O(n²))
attn = softmax(Q @ K.T / sqrt(d)) @ V  # All tokens attend to all

# Sparse Attention (O(n × k))
indices = indexer(Q, K)  # Select top-k tokens per query
attn = sparse_softmax(Q @ K[indices].T / sqrt(d)) @ V[indices]
```

### DSA Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DSA Layer                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Query ──┬──────────────────────────────────────► MLA      │
│           │                                          │       │
│           ▼                                          │       │
│      ┌─────────┐                                     │       │
│      │ Indexer │──► indices [batch, seq_q, topk]     │       │
│      └─────────┘                                     ▼       │
│           │                               Sparse Attention   │
│           │                                     │            │
│   KV Cache ◄──────────────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Indexer Module

The indexer selects which KV tokens each query should attend to:

```python
class DSAIndexer(nn.Module):
    def forward(self, q, kv):
        # Compute relevance scores
        # Uses weighted ReLU MQA logits
        logits = relu_mqa_logits(q, kv, self.weights)
        
        # Select top-k most relevant tokens
        indices = logits.topk(self.topk, dim=-1).indices
        
        return indices  # [batch, seq_q, topk]
```

**Important**: RoPE in indexer uses **non-interleaved** layout, while MLA expects **interleaved** layout.

### Supporting Kernels

**DeepGEMM (Indexer Logits)**:
```python
# PR #200: fp8_mqa_logits for indexer
deep_gemm.fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end)

# Computes: out[i,j] = ReLU(q[i] @ kv[j]) * weights[i] summed over heads
```

**FlashMLA (Sparse Attention)**:
```python
# PR #98: Sparse prefill and decode kernels
flash_mla.flash_mla_sparse_fwd(q, kv, indices, sm_scale)

# Sparse decoding with FP8 KV cache
flash_mla.flash_mla_with_kvcache(..., indices=indices)
```

## Performance

### Quality Preservation

| Benchmark | V3.1-Terminus | V3.2-Exp |
|-----------|---------------|----------|
| AIME 2025 | 88.4 | **89.3** |
| Codeforces | 2046 | **2121** |
| LiveCodeBench | 74.9 | 74.1 |
| GPQA-Diamond | 80.7 | 79.9 |

**Key result**: Virtually identical quality with significant efficiency gains.

### Kernel Performance

| Kernel | Performance |
|--------|-------------|
| Sparse Prefill | 640 TFLOPS (H800), 1450 TFLOPS (B200) |
| Sparse Decode | 410 TFLOPS with FP8 KV cache |

## Deployment

### SGLang

```bash
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2-Exp \
    --tp 8 --dp 8 \
    --enable-dp-attention
```

### Weight Conversion

```bash
cd inference
python convert.py \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --save-path ${SAVE_PATH} \
    --n-experts 256 \
    --model-parallel ${MP}
```

## DSA Benefits

1. **Memory Efficiency**: Only cache selected KV tokens
2. **Compute Efficiency**: O(n×k) instead of O(n²)
3. **Long Context**: Better scaling for 128K+ sequences
4. **Quality**: Learnable sparsity preserves important attention patterns

## Connection to DeepSeekMath-V2

V3.2-Exp-Base serves as the foundation for DeepSeekMath-V2's theorem proving capabilities. The DSA attention enables efficient processing of long mathematical proofs.

## Key Files

| File | Purpose |
|------|---------|
| `inference/convert.py` | Weight conversion |
| `README.md` | Deployment guide |
