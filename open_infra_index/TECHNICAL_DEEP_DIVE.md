# Open Infrastructure Index: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/open-infra-index.git  
> Commit: `56d86855fcf6e08fdfd45ce6280bd24322c93351` (2025-05-15)

## Overview

The Open Infrastructure Index serves as the central registry and documentation hub for DeepSeek's open-source infrastructure projects. It provides an overview of how different components work together.

## Infrastructure Stack

```
┌─────────────────────────────────────────────────────────────┐
│                  DeepSeek Infrastructure                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Models                            │    │
│  │  V2 │ V3 │ R1 │ Coder │ VL2 │ MoE │ etc.           │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Compute Kernels                        │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │    │
│  │  │ FlashMLA │ │ DeepGEMM │ │  Other   │            │    │
│  │  │(Attention)│ │ (GEMM)   │ │ Kernels  │            │    │
│  │  └──────────┘ └──────────┘ └──────────┘            │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Parallelism & Communication               │    │
│  │  ┌──────────┐ ┌──────────┐                          │    │
│  │  │ DualPipe │ │  DeepEP  │                          │    │
│  │  │   (PP)   │ │   (EP)   │                          │    │
│  │  └──────────┘ └──────────┘                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Storage                           │    │
│  │  ┌──────────┐ ┌──────────┐                          │    │
│  │  │   3FS    │ │smallpond │                          │    │
│  │  │  (DFS)   │ │  (ETL)   │                          │    │
│  │  └──────────┘ └──────────┘                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Project Registry

### Compute Kernels

| Project | Purpose | Key Innovation |
|---------|---------|----------------|
| **FlashMLA** | MLA attention | Seesaw scheduling, 660 TFLOPS |
| **DeepGEMM** | FP8/BF16 GEMM | JIT compilation, 1550 TFLOPS |

### Communication

| Project | Purpose | Key Innovation |
|---------|---------|----------------|
| **DeepEP** | Expert parallelism | Hook-based overlap, 77μs latency |
| **DualPipe** | Pipeline parallelism | Bidirectional, ~78% less bubble |

### Storage

| Project | Purpose | Key Innovation |
|---------|---------|----------------|
| **3FS** | Distributed FS | CRAQ consistency, 6.6 TiB/s |
| **smallpond** | Data processing | DuckDB backend, 3.66 TiB/min sort |

## Component Interactions

### Training Data Flow

```
Raw Data (3FS)
      │
      ▼
smallpond (ETL)
      │ Tokenization, filtering, dedup
      ▼
Processed Data (3FS)
      │
      ▼
Dataloader
      │ Random access via 3FS
      ▼
Training (DualPipe + DeepEP)
      │ PP schedules micro-batches
      │ EP handles MoE communication
      ▼
Checkpoints (3FS)
```

### Inference Data Flow

```
Request
   │
   ▼
Prefill (FlashMLA + DeepGEMM)
   │ MLA attention + MoE GEMM
   ▼
KV Cache (3FS optional)
   │ Offload for large batches
   ▼
Decode (FlashMLA + DeepEP)
   │ Low-latency attention + EP
   ▼
Response
```

## Version Compatibility

### Hardware Requirements

| Component | SM80 (A100) | SM90 (H100/H800) | SM100 (B200) |
|-----------|-------------|------------------|--------------|
| FlashMLA | Partial | Full | Full |
| DeepGEMM | Partial | Full | Full |
| DeepEP | Intranode only | Full | Full |

### Software Requirements

| Component | CUDA | PyTorch | Python |
|-----------|------|---------|--------|
| FlashMLA | 12.3+ | 2.1+ | 3.8+ |
| DeepGEMM | 12.3+ | 2.1+ | 3.8+ |
| DeepEP | 12.3+ | 2.1+ | 3.8+ |
| DualPipe | Any | 2.0+ | 3.8+ |
| 3FS | N/A | N/A | N/A |
| smallpond | N/A | N/A | 3.8+ |

## Getting Started

### Minimal Setup (Inference)

```bash
# Install compute kernels
pip install flash-mla deep-gemm

# Run inference
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --tp 8
```

### Full Setup (Training)

```bash
# Install all components
pip install flash-mla deep-gemm deep-ep dualpipe

# Build 3FS (see 3fs/deploy/README.md)
# Setup smallpond for data processing

# Configure cluster
export NVSHMEM_DIR=/path/to/nvshmem
export THREE_FS_MOUNT=/mnt/3fs
```

### Data Processing Setup

```bash
# Install smallpond
pip install smallpond

# Configure 3FS mount
export SMALLPOND_STORAGE=/mnt/3fs/scratch

# Run ETL pipeline
python my_pipeline.py
```

## Documentation Links

| Project | Documentation |
|---------|---------------|
| FlashMLA | `flash_mla/README.md`, `flash_mla/docs/` |
| DeepGEMM | `deep_gemm/README.md` |
| DeepEP | `deep_ep/README.md` |
| DualPipe | `dualpipe/README.md` |
| 3FS | `3fs/docs/design_notes.md`, `3fs/deploy/README.md` |
| smallpond | `smallpond/docs/` |

## Community Resources

### Issue Tracking

- Report bugs: Each project's GitHub Issues
- Feature requests: GitHub Discussions
- Security issues: security@deepseek.com

### Contributing

1. Fork the relevant repository
2. Create a feature branch
3. Submit a pull request
4. Follow the project's contribution guidelines

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Main index and overview |
| Project-specific docs | Detailed documentation |
