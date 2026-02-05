# Mardia

*Mardi* (French for Tuesday) + *IA* (Intelligence Artificielle) = **Mardia**

A curated collection of open-source AI infrastructure and model implementations. Currently focused on DeepSeek's releases, but may expand to include other notable open-source AI projects.

> **Disclaimer**: All work in this repository belongs to their respective authors and organizations (primarily [DeepSeek](https://github.com/deepseek-ai)). This collection is provided solely as a convenience for study and reference. Please refer to the original repositories for official documentation, updates, and licensing terms.

## Release Timeline

| Date | Release | Key Innovation |
|------|---------|----------------|
| 2024.01 | [DeepSeek-MoE](deepseek_moe/) | Fine-grained expert segmentation, shared expert isolation |
| 2024.01 | [DeepSeek-Coder](deepseek_coder/) | Code LLM with 86 languages, FIM training |
| 2024.05 | [DeepSeek-V2](deepseek_v2/) | Multi-head Latent Attention (MLA), 93% KV cache reduction |
| 2024.06 | [DeepSeek-Coder-V2](deepseek_coder_v2/) | MoE code model, 338 languages, 128K context |
| 2024.12 | [DeepSeek-V3](deepseek_v3/) | 671B MoE, FP8 training, auxiliary-loss-free balancing |
| 2024.12 | [DeepSeek-VL2](deepseek_vl2/) | MoE vision-language model |
| 2025.01 | [DeepSeek-R1](deepseek_r1/) | Reasoning via pure RL, o1-level performance |
| 2025.02 | Open Source Week | FlashMLA, DeepEP, DeepGEMM, DualPipe, 3FS |
| 2025.09 | [DeepSeek-V3.2](deepseek_v3_2_exp/) | DeepSeek Sparse Attention (DSA) |

## Reading Order

**Architecture track** (understand the model evolution):
1. [DeepSeek-MoE](deepseek_moe/DEEP_DIVE.md) - Foundation: fine-grained experts, shared experts
2. [DeepSeek-V2](deepseek_v2/DEEP_DIVE.md) - MLA attention that makes decoding compute-bound
3. [DeepSeek-V3](deepseek_v3/DEEP_DIVE.md) - Full stack: FP8 training, MTP, load balancing
4. [DeepSeek-R1](deepseek_r1/DEEP_DIVE.md) - RL-based reasoning emergence

**Infrastructure track** (understand the systems):
1. [3FS](3fs/TECHNICAL_DEEP_DIVE.md) - Storage layer: CRAQ consistency, 6.6 TiB/s throughput
2. [DeepGEMM](deep_gemm/TECHNICAL_DEEP_DIVE.md) - Compute: FP8 GEMM, 1550 TFLOPS
3. [FlashMLA](flash_mla/TECHNICAL_DEEP_DIVE.md) - Attention: MLA kernels, 660 TFLOPS
4. [DeepEP](deep_ep/TECHNICAL_DEEP_DIVE.md) - Communication: expert parallelism, 77μs latency
5. [DualPipe](dualpipe/TECHNICAL_DEEP_DIVE.md) - Training: bidirectional PP, 78% less bubble

## Infrastructure

| Project | Description |
|---------|-------------|
| [3fs](3fs/) | Fire-Flyer File System - High-performance distributed file system for AI workloads |
| [deep_ep](deep_ep/) | Communication library for Mixture-of-Experts (MoE) and expert parallelism |
| [deep_gemm](deep_gemm/) | Efficient GEMM kernels (FP8/BF16) with JIT compilation |
| [dualpipe](dualpipe/) | Bidirectional pipeline parallelism with full computation-communication overlap |
| [flash_mla](flash_mla/) | Optimized Multi-head Latent Attention kernels for Hopper GPUs |
| [smallpond](smallpond/) | Lightweight data processing framework built on DuckDB and 3FS |
| [engram](engram/) | Conditional memory via scalable N-gram lookup for LLMs |

## Models

| Project | Description |
|---------|-------------|
| [deepseek_v3](deepseek_v3/) | DeepSeek-V3 model implementation |
| [deepseek_v3_2_exp](deepseek_v3_2_exp/) | DeepSeek-V3.2 experimental release |
| [deepseek_r1](deepseek_r1/) | DeepSeek-R1 reasoning model |
| [deepseek_v2](deepseek_v2/) | DeepSeek-V2 model |
| [deepseek_vl2](deepseek_vl2/) | DeepSeek-VL2 vision-language model |
| [deepseek_coder](deepseek_coder/) | DeepSeek-Coder for code generation |
| [deepseek_coder_v2](deepseek_coder_v2/) | DeepSeek-Coder-V2 |
| [deepseek_math_v2](deepseek_math_v2/) | DeepSeek-Math-V2 for mathematical reasoning |
| [deepseek_moe](deepseek_moe/) | DeepSeek-MoE base implementation |
| [deepseek_ocr_v2](deepseek_ocr_v2/) | DeepSeek-OCR-V2 |

## Resources

- [Open Infra Index](open_infra_index/) - Overview of DeepSeek's open-source releases

## License

See individual project directories for specific licenses.
