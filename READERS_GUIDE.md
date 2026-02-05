# DeepSeek Repository Reader's Guide & Index

> A comprehensive map of all DeepSeek repositories, their core content, and relationships.

---

## Quick Reference Matrix

| Category | Repository | Status | Key Innovation | Params | Paper |
|----------|------------|--------|----------------|--------|-------|
| **Foundation Models** |
| | `deepseek_moe/` | Local | Fine-grained MoE, Shared Experts | 16B (2.4B active) | arXiv:2401.06066 |
| | `deepseek_v2/` | Local | MLA + MoE | 236B (21B active) | arXiv:2405.04434 |
| | `deepseek_v3/` | Local | FP8 Training, MTP, Aux-loss-free | 671B (37B active) | arXiv:2412.19437 |
| | `deepseek_r1/` | Local | RL without SFT, Reasoning | 671B (37B active) | arXiv:2501.12948 |
| **Newer Architectures** |
| | `deepseek_v3_2_exp/` | Local | DeepSeek Sparse Attention (DSA) | 685B | GitHub only |
| | `engram/` | Local | Conditional Memory, N-gram Lookup | Research | arXiv:2601.07372 |
| **Math & Reasoning** |
| | `deepseek_math_v2/` | Local | Self-Verifiable Proofs | 685B | GitHub only |
| **Multi-Modal** |
| | `deepseek_ocr_v2/` | Local | Document OCR, Layout Detection | 3B | arXiv:2601.20552 |
| | DeepSeek-VL2 | External | MoE Vision-Language | 4.5B (activated) | arXiv:2412.10302 |
| | Janus | External | Unified Understanding + Generation | 1.3B-7B | arXiv:2410.13848 |
| **Coding** |
| | DeepSeek-Coder | External | Code LLM, 86 Languages | 1B-33B | arXiv:2401.14196 |
| | DeepSeek-Coder-V2 | External | Code MoE, 338 Languages | 236B (21B active) | arXiv:2406.11931 |
| **Infrastructure** |
| | `deep_ep/` | Local | Expert Parallelism Communication | N/A | GitHub only |
| | `deep_gemm/` | Local | FP8/BF16 GEMM Kernels | N/A | GitHub only |
| | `flash_mla/` | Local | MLA Attention Kernels | N/A | GitHub only |
| | `dualpipe/` | Local | Bidirectional Pipeline Parallelism | N/A | GitHub only |
| | `3fs/` | Local | Distributed File System | N/A | GitHub only |

---

## Part 1: Foundation Models

### 1.1 DeepSeekMoE (`deepseek_moe/`)

**Purpose:** Introduces the foundational MoE architecture used in all subsequent DeepSeek models.

**Key Innovations:**
- Fine-grained expert segmentation (more experts, fewer params each)
- Shared expert isolation (dedicated experts for common knowledge)
- 40% compute of dense model with comparable performance

**What's Included:**
- `finetune/finetune.py` - DeepSpeed fine-tuning script
- `finetune/configs/` - ZeRO-2/3 configurations
- Paper PDF

**What's Missing:**
- Pre-training code
- Data preparation pipeline

**HuggingFace:** `deepseek-ai/deepseek-moe-16b-base`, `deepseek-ai/deepseek-moe-16b-chat`

---

### 1.2 DeepSeek-V2 (`deepseek_v2/`)

**Purpose:** Introduces Multi-head Latent Attention (MLA) for efficient KV-cache.

**Key Innovations:**
- **MLA:** Low-rank KV compression → 93.3% KV cache reduction
- **DeepSeekMoE:** Refined from original MoE
- 128K context length
- 42.5% training cost reduction vs DeepSeek 67B

**Architecture:**
```
Input → MLA Attention → DeepSeekMoE FFN → Output
         ↓                    ↓
    Latent KV              Sparse Experts
    Compression            + Shared Experts
```

**What's Included:**
- README with architecture details
- Inference examples (Transformers, vLLM, SGLang)
- Paper PDF

**What's Missing:**
- Model implementation code (use HuggingFace)
- Training code

**HuggingFace:** `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat`

---

### 1.3 DeepSeek-V3 (`deepseek_v3/`)

**Purpose:** Production-scale model with FP8 training and Multi-Token Prediction.

**Key Innovations:**
- **FP8 Mixed Precision Training:** First validated at 671B scale
- **Auxiliary-loss-free Load Balancing:** No performance degradation from balancing
- **Multi-Token Prediction (MTP):** Predicts multiple future tokens, enables speculative decoding
- 14.8T training tokens, only 2.788M H800 GPU hours

**What's Included:**
- `inference/` - Demo inference code with weight conversion
- `configs/` - Model configurations
- FP8→BF16 conversion script
- Paper PDF

**What's Missing:**
- Training code
- MTP training implementation

**HuggingFace:** `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3`

---

### 1.4 DeepSeek-R1 (`deepseek_r1/`)

**Purpose:** Reasoning model trained via RL without supervised fine-tuning.

**Key Innovations:**
- **R1-Zero:** Pure RL on base model → emergent chain-of-thought
- **R1:** Cold-start data + RL for better reasoning
- Self-verification, reflection behaviors
- Distillation to 1.5B-70B models

**Training Pipeline:**
```
Base Model → RL Stage 1 (Reasoning Patterns)
          → SFT Stage 1 (Seed Capabilities)
          → RL Stage 2 (Human Preference)
          → SFT Stage 2 (Non-reasoning)
```

**What's Included:**
- README with evaluation results
- Usage recommendations
- Prompt templates

**What's Missing:**
- RL training code
- Distillation scripts
- Cold-start data

**HuggingFace:** `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-R1-Zero`, `deepseek-ai/DeepSeek-R1-Distill-*`

---

## Part 2: Newer Architectures

### 2.1 DeepSeek-V3.2-Exp (`deepseek_v3_2_exp/`)

**Purpose:** Experimental model introducing DeepSeek Sparse Attention (DSA).

**Key Innovations:**
- **DSA:** Fine-grained token-level sparse attention
- Substantial long-context training/inference efficiency
- Maintains output quality vs dense attention
- Indexer module for token selection

**What's Included:**
- `inference/` - Updated demo with DSA
- Configuration files
- Benchmark results

**External Kernels:**
- Sparse attention: `flash_mla/` (PR #98)
- Indexer logits: `deep_gemm/` (PR #200)

**HuggingFace:** `deepseek-ai/DeepSeek-V3.2-Exp`

---

### 2.2 Engram (`engram/`)

**Purpose:** Research on conditional memory as complementary sparsity axis.

**Key Innovations:**
- **Engram Module:** O(1) N-gram embedding lookup
- Trade-off between neural computation (MoE) and static memory
- U-shaped scaling law for optimal allocation
- Host memory offloading for massive tables

**What's Included:**
- `engram_demo_v1.py` - Standalone demonstration
- Paper PDF
- Architecture diagrams

**What's Missing:**
- Full training code
- Integration with production models

---

## Part 3: Math & Reasoning

### 3.1 DeepSeekMath-V2 (`deepseek_math_v2/`)

**Purpose:** Self-verifiable mathematical reasoning with theorem proving.

**Key Innovations:**
- Verifier as reward model
- Generator trained to identify/resolve proof issues
- Scaled verification compute for hard proofs
- Gold-level IMO 2025, 118/120 Putnam 2024

**What's Included:**
- `inference/` - Generation scripts
- `inputs/` - Competition problems (IMO, CMO, Putnam)
- `outputs/` - Model predictions
- Paper PDF

**What's Missing:**
- Verifier training code
- RL training pipeline

**HuggingFace:** `deepseek-ai/DeepSeek-Math-V2`

---

## Part 4: Multi-Modal Models

### 4.1 DeepSeek-OCR-2 (`deepseek_ocr_v2/`)

**Purpose:** Document OCR with layout understanding.

**Key Innovations:**
- **Dual Encoder:** SAM ViT-B + Qwen2 decoder-as-encoder
- Dynamic resolution (up to 6×768² crops + 1×1024²)
- Layout detection with grounding
- Document → Markdown conversion

**Architecture:**
```
Image → Dynamic Crop → SAM Encoder → Qwen2 Encoder → MLP Projector → DeepSeek LLM
         ↓                                                              ↓
    Global + Local Views                                         Markdown + Boxes
```

**What's Included:**
- `DeepSeek-OCR2-vllm/` - vLLM inference implementation
- `DeepSeek-OCR2-hf/` - Transformers inference
- `process/` - Image preprocessing
- Paper PDF

**HuggingFace:** `deepseek-ai/DeepSeek-OCR-2`

---

### 4.2 DeepSeek-VL2 (External)

**Purpose:** MoE Vision-Language model for advanced multimodal understanding.

**Key Innovations:**
- MoE language backbone
- Multi-image/interleaved conversations
- Visual grounding with bounding boxes
- Incremental prefilling for memory efficiency

**Models:**
- VL2-Tiny: 3.37B total, 1B active
- VL2-Small: 16.1B total, 2.4B active
- VL2: 27.5B total, 4.2B active

**GitHub:** https://github.com/deepseek-ai/DeepSeek-VL2

**HuggingFace:** `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2`

---

### 4.3 Janus (External)

**Purpose:** Unified multimodal understanding AND generation.

**Key Innovations:**
- **Decoupled Visual Encoding:** Separate paths for understanding vs generation
- Single unified transformer
- Text-to-image generation capability
- JanusFlow: Rectified flow for image generation

**Models:**
- Janus-1.3B, JanusFlow-1.3B
- Janus-Pro-1B, Janus-Pro-7B

**GitHub:** https://github.com/deepseek-ai/Janus

**HuggingFace:** `deepseek-ai/Janus-1.3B`, `deepseek-ai/Janus-Pro-7B`

---

## Part 5: Coding Models

### 5.1 DeepSeek-Coder (External)

**Purpose:** Code-focused LLM trained from scratch.

**Key Innovations:**
- 2T tokens (87% code, 13% NL)
- 86 programming languages
- 16K context with FIM (fill-in-middle)
- Repo-level code completion

**Models:** 1B, 5.7B, 6.7B, 33B (Base + Instruct)

**What's Included:**
- `finetune/` - Fine-tuning scripts
- `Evaluation/` - Benchmark reproduction
- `demo/` - Gradio demo

**GitHub:** https://github.com/deepseek-ai/DeepSeek-Coder

**HuggingFace:** `deepseek-ai/deepseek-coder-*`

---

### 5.2 DeepSeek-Coder-V2 (External)

**Purpose:** MoE code model achieving GPT-4 Turbo parity.

**Key Innovations:**
- MoE architecture (DeepSeek-V2 based)
- 338 programming languages
- 128K context
- Continued pre-training from V2 checkpoint

**Models:**
- Lite: 16B total, 2.4B active
- Full: 236B total, 21B active

**GitHub:** https://github.com/deepseek-ai/DeepSeek-Coder-V2

**HuggingFace:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`, `deepseek-ai/DeepSeek-Coder-V2-Instruct`

---

## Part 6: Infrastructure

### 6.1 DeepEP (`deep_ep/`)

**Purpose:** High-performance MoE dispatch/combine communication.

**Key Features:**
- All-to-all GPU kernels for expert parallelism
- NVLink + RDMA domain forwarding
- Low-latency decoding kernels (pure RDMA)
- Hook-based comm-compute overlap (zero SM)
- FP8 dispatch support

**Performance:**
- Normal: 153 GB/s NVLink, 58 GB/s RDMA
- Low-latency: 77-194 μs dispatch, 39-127 GB/s

**Key Files:**
- `tests/test_intranode.py`, `test_internode.py`, `test_low_latency.py`

**Dependencies:** NVSHMEM, PyTorch 2.1+, CUDA 12.3+ (SM90)

---

### 6.2 DeepGEMM (`deep_gemm/`)

**Purpose:** Clean, efficient GEMM kernels for FP8/BF16.

**Key Features:**
- FP8 and BF16 GEMMs
- Grouped GEMMs for MoE (contiguous + masked layouts)
- JIT compilation (no install-time kernel build)
- SM90 (Hopper) and SM100 (Blackwell) support
- Up to 1550 TFLOPS on H800

**Key Files:**
- `tests/test_core.py` - Core GEMM tests
- `tests/test_attention.py` - MQA logits kernels
- `deep_gemm/legacy/` - Grouped GEMM variants

**Dependencies:** CUDA 12.3+, CUTLASS 4.0+, PyTorch 2.1+

---

### 6.3 FlashMLA (`flash_mla/`)

**Purpose:** Optimized attention kernels for MLA.

**Key Features:**
- Dense MLA decoding: 3000 GB/s memory-bound, 660 TFLOPS compute-bound
- Sparse MLA (DSA): 640 TFLOPS prefill, 410 TFLOPS decode
- FP8 KV cache support
- SM90 and SM100 support

**Key Files:**
- `tests/test_flash_mla_dense_decoding.py`
- `tests/test_flash_mla_sparse_decoding.py`
- `tests/test_flash_mla_sparse_prefill.py`
- `docs/` - Deep-dive kernel documentation

**Dependencies:** CUDA 12.8+, PyTorch 2.0+

---

### 6.4 DualPipe (`dualpipe/`)

**Purpose:** Bidirectional pipeline parallelism with full overlap.

**Key Features:**
- Full forward-backward computation-communication overlap
- Reduced pipeline bubbles
- DualPipeV variant (V-shape schedule)

**Bubble Comparison:**
| Method | Bubble |
|--------|--------|
| 1F1B | (PP-1)(F+B) |
| ZB1P | (PP-1)(F+B-2W) |
| DualPipe | (PP/2-1)(F&B+B-3W) |

**Key Files:**
- `examples/example_dualpipe.py`
- `examples/example_dualpipev.py`

---

### 6.5 3FS (`3fs/`)

**Purpose:** High-performance distributed file system for AI workloads.

**Key Features:**
- 6.6 TiB/s aggregate read throughput
- Disaggregated architecture (SSD + RDMA)
- Strong consistency (CRAQ)
- KVCache offloading for inference
- High-throughput checkpointing

**Use Cases:**
- Training data loading
- Checkpoint saving/loading
- Inference KVCache storage

**Key Files:**
- `src/lib/api/UsrbIo.md` - API reference
- `deploy/README.md` - Setup guide
- `benchmarks/fio_usrbio/` - Benchmarking tools

**Dependencies:** FoundationDB 7.1+, libfuse 3.16+, Rust 1.75+

---

## Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │           Infrastructure                 │
                    │  ┌─────┐ ┌─────────┐ ┌────────┐ ┌─────┐ │
                    │  │3FS  │ │DeepGEMM │ │FlashMLA│ │DeepEP│ │
                    │  └──┬──┘ └────┬────┘ └───┬────┘ └──┬──┘ │
                    │     │         │          │         │     │
                    │     └─────────┴────┬─────┴─────────┘     │
                    │                    │                      │
                    │              ┌─────┴─────┐                │
                    │              │ DualPipe  │                │
                    └──────────────┴─────┬─────┴────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
   ┌────┴────┐                    ┌──────┴──────┐                  ┌──────┴──────┐
   │DeepSeek │                    │  DeepSeek   │                  │  DeepSeek   │
   │  MoE    │──────────────────▶ │     V2      │────────────────▶ │     V3      │
   └─────────┘                    └──────┬──────┘                  └──────┬──────┘
        │                                │                                │
        │                                │                                │
        │                         ┌──────┴──────┐                  ┌──────┴──────┐
        │                         │ Coder V2    │                  │  V3.2-Exp   │
        │                         └─────────────┘                  │   (DSA)     │
        │                                                          └──────┬──────┘
        │                                                                 │
        │                                                          ┌──────┴──────┐
        │                                                          │     R1      │
        │                                                          │  (Reasoning)│
        │                                                          └──────┬──────┘
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┤
                                                                          │
                              ┌───────────────────────────────────────────┼───────┐
                              │                                           │       │
                       ┌──────┴──────┐                             ┌──────┴──────┐│
                       │   Janus     │                             │  Math V2    ││
                       │(Understand  │                             │ (Proofs)    ││
                       │+Generate)   │                             └─────────────┘│
                       └─────────────┘                                            │
                              │                                                   │
                       ┌──────┴──────┐                             ┌──────────────┤
                       │    VL2      │                             │   Engram     │
                       │(Vision MoE) │                             │(Conditional  │
                       └──────┬──────┘                             │  Memory)     │
                              │                                    └──────────────┘
                       ┌──────┴──────┐
                       │   OCR-2     │
                       │ (Document)  │
                       └─────────────┘
```

---

## Getting Started Checklist

### Environment Setup
- [ ] CUDA 12.3+ (12.9 for SM100)
- [ ] PyTorch 2.1+
- [ ] 8×H800/A100 GPUs (for full models)
- [ ] InfiniBand/NVLink for multi-node

### Local Repos Setup
```bash
cd /mnt/data_infra/workspace/dimanchia

# Infrastructure (build these first)
cd deep_gemm && ./develop.sh && cd ..
cd flash_mla && pip install -v . && cd ..
cd deep_ep && python setup.py build && cd ..

# Models (inference)
cd deepseek_v3/inference && pip install -r requirements.txt && cd ../..
```

### External Repos to Clone
```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL2
git clone https://github.com/deepseek-ai/DeepSeek-Coder
git clone https://github.com/deepseek-ai/DeepSeek-Coder-V2
git clone https://github.com/deepseek-ai/Janus
```

---

## Version History

| Model | Release Date | Key Update |
|-------|--------------|------------|
| DeepSeekMoE | Jan 2024 | Initial MoE |
| DeepSeek-Coder | Jan 2024 | Code LLM |
| DeepSeek-V2 | May 2024 | MLA attention |
| DeepSeek-Coder-V2 | Jun 2024 | Code MoE |
| DeepSeek-VL2 | Dec 2024 | Vision MoE |
| DeepSeek-V3 | Dec 2024 | FP8, MTP |
| Janus | Oct 2024 | Unified multimodal |
| DeepSeek-R1 | Jan 2025 | RL reasoning |
| Janus-Pro | Jan 2025 | Scaled Janus |
| DeepSeek-OCR-2 | Jan 2026 | Document OCR |
| DeepSeek-Math-V2 | 2025 | Self-verifiable proofs |
| DeepSeek-V3.2-Exp | 2025 | Sparse Attention |
| Engram | 2025 | Conditional Memory |
