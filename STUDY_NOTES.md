# DeepSeek Technical Research Curriculum - Study Notes

> Working document capturing key learnings, code analysis, and implementation insights.

---

## Phase 1: Model Foundations

### Module 1.1: DeepSeekMoE - The Building Block

**Status:** Complete

#### Key Paper Insights (arXiv:2401.06066)

**Problem Statement:**
Traditional MoE architectures (e.g., GShard with top-2 routing) suffer from:
1. **Knowledge Hybridity:** Limited experts (8-16) → tokens cover diverse knowledge → experts must learn vastly different knowledge types
2. **Knowledge Redundancy:** Different experts may converge on learning common knowledge → parameter inefficiency

**DeepSeekMoE Solution:**

1. **Fine-Grained Expert Segmentation:**
   - Split each expert FFN into `m` smaller experts (reduce intermediate hidden dim to 1/m)
   - Activate `m×K` experts instead of `K` to maintain compute
   - Example: 16 experts with top-2 → 64 experts with top-8
   - Combinatorial flexibility: C(16,2)=120 → C(64,8)=4,426,165,368 combinations

2. **Shared Expert Isolation:**
   - Dedicate `K_s` experts as always-on shared experts
   - Shared experts capture common knowledge across contexts
   - Reduces redundancy among routed experts
   - Ratio found optimal: 1 shared : 3 activated routed

**Mathematical Formulation:**

```
# Standard MoE Layer
h_t = Σ(g_i,t × FFN_i(u_t)) + u_t
where g_i,t = s_i,t if s_i,t ∈ TopK(affinities), else 0

# DeepSeekMoE Layer  
h_t = Σ(FFN_i(u_t)) for shared experts + Σ(g_i,t × FFN_i(u_t)) for routed + u_t
```

**Architecture Configurations:**

| Scale | Layers | Hidden | Experts | Activated | Relative Size | Params |
|-------|--------|--------|---------|-----------|---------------|--------|
| 2B    | 9      | 1280   | 1+63    | 1+7       | 0.25×FFN      | 2.0B   |
| 16B   | 28     | 2048   | 2+64    | 2+6       | 0.25×FFN      | 16.4B  |
| 145B  | 62     | 4096   | 4+128   | 4+12      | 0.125×FFN     | 144.6B |

**Load Balancing:**

1. **Expert-Level Balance Loss:** Prevents routing collapse
   ```
   L_ExpBal = α₁ × Σ(f_i × P_i)
   where f_i = fraction of tokens selecting expert i
         P_i = mean routing probability for expert i
   ```
   - α₁ = 0.01 (small, for 2B), 0.001 (for 16B)

2. **Device-Level Balance Loss:** For distributed training
   ```
   L_DevBal = α₂ × Σ(f'_i × P'_i)
   ```
   - α₂ = 0.05 (for 145B with expert parallelism)

**Key Results:**
- DeepSeekMoE 2B ≈ GShard 2.9B (1.5× expert params)
- DeepSeekMoE 16B ≈ LLaMA2 7B with **40%** compute
- DeepSeekMoE 145B ≈ DeepSeek 67B with **28.5%** compute

#### Fine-Tuning Code Analysis (`finetune/finetune.py`)

**Key Components:**

1. **Model Arguments:**
```python
@dataclass
class ModelArguments:
    trainable: str = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    lora_rank: int = 8
    lora_dropout: float = 0.1
    lora_alpha: float = 32.
    modules_to_save: str = "embed_tokens,lm_head"
    use_lora: bool = False
    model_name_or_path: str = "deepseek-ai/deepseek-moe-16b"
    attn_implementation: str = "flash_attention_2"
    bits: int = 16  # 4/8 for QLoRA
```

2. **Data Format:** Uses `instruction` + `output` fields
```python
def build_instruction_prompt(instruction: str):
    return '''
You are an AI assistant, developed by DeepSeek Company...
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()
```

3. **DeepSpeed ZeRO-3 Config (`ds_config_zero3.json`):**
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "offload_param": {"device": "cpu", "pin_memory": true},
        "overlap_comm": true,
        "contiguous_gradients": true
    }
}
```

4. **LoRA Integration:**
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "down_proj", "up_proj"],
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    modules_to_save=["embed_tokens", "lm_head"]
)
```

**Training Commands:**

Full fine-tuning (8×A100 40GB):
```bash
deepspeed finetune.py \
    --model_name_or_path deepseek-ai/deepseek-moe-16b-base \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True --use_lora False
```

QLoRA (1×A100 80GB):
```bash
deepspeed finetune.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 True --use_lora True --bits 4 \
    --lora_r 64 --lora_alpha 16 --quant_type nf4
```

#### Infrastructure Connection

**→ DeepEP (`deep_ep/`):** Understanding MoE routing prepares for expert parallelism communication patterns:
- Dispatch: Tokens → Experts (all-to-all)
- Combine: Expert outputs → Original positions (all-to-all)

---

### Module 1.2: DeepSeek-V2 - Multi-head Latent Attention (MLA)

**Status:** Complete

#### Key Architecture (arXiv:2405.04434)

**Model Specifications:**

| Variant | Total Params | Activated | Context | Architecture |
|---------|-------------|-----------|---------|--------------|
| V2-Lite | 16B | 2.4B | 32K | MLA + MoE |
| V2 | 236B | 21B | 128K | MLA + MoE |

**Key Innovations:**
1. **Multi-head Latent Attention (MLA):** 93.3% KV cache reduction
2. **DeepSeekMoE FFN:** Refined from original MoE architecture
3. **42.5% training cost reduction** vs DeepSeek 67B

#### MLA Mechanism Explained

**Problem:** Standard Multi-Head Attention (MHA) KV cache grows as:
```
KV_cache_size = batch × seq_len × num_heads × head_dim × 2 (K and V)
```
For 128K context with 128 heads × 128 dim → ~4GB per batch element just for KV!

**Solution: Low-Rank KV Compression**

Instead of storing full K, V tensors, MLA compresses them into a shared latent representation:

```
Standard MHA:
    K = X @ W_k    # [seq, heads, head_dim]
    V = X @ W_v    # [seq, heads, head_dim]
    
MLA (simplified):
    latent = X @ W_down    # [seq, latent_dim] - much smaller!
    K = latent @ W_up_k    # Reconstruct K on-the-fly
    V = latent @ W_up_v    # Reconstruct V on-the-fly
```

**MLA in DeepSeek-V2:**
- `d_latent = 512` (latent dimension)
- `d_rope = 64` (RoPE dimension, not compressed)
- KV cache per token: **512 + 64 = 576** elements vs thousands in standard attention
- **93.3% reduction** in KV cache size

**Memory Layout (from FlashMLA FP8 format):**
```
Per token KV cache: 656 bytes total
├── Quantized NoPE:  512 bytes (512 × FP8_E4M3)
├── Scale factors:   16 bytes (4 × float32)
└── RoPE part:       128 bytes (64 × bfloat16, not quantized)
```

#### FlashMLA Kernel Analysis

**Performance Achieved:**
- Dense decoding: **660 TFLOPS** (compute-bound), **3000 GB/s** (memory-bound)
- Sparse decoding: **410 TFLOPS** with FP8 KV cache
- Sparse prefill: **640 TFLOPS** (H800), **1450 TFLOPS** (B200)

**Why MLA Decoding is Compute-Bound:**

Compute-to-memory ratio analysis:
```
FLOPs = 2 × h_q × s_q × s_k × (d_k + d_v)
Memory = 2 × s_k × d_k  (read KV cache)
Ratio ≈ 2 × h_q × s_q

For DeepSeek with h_q=128:
    Ratio = 256 >> H800 threshold of 128
    → Kernel is compute-bound!
```

**Seesaw Scheduling (from deep-dive blog):**

Traditional ping-pong requires 2 output matrices, but MLA's 64×512 output uses 32,768 registers (half of SM's 65,536). Solution: "Seesaw" scheduling with 2 warpgroups sharing one output matrix split vertically:

```
Output matrix O (64×512) split into:
├── O_L (64×256) → Warpgroup 0
└── O_R (64×256) → Warpgroup 1

Schedule alternates:
1. WG0: Compute P0 = Q @ K0.T
2. WG1: Compute P1 = Q @ K1.T  
3. WG0: Softmax + accumulate O_L
4. WG1: Softmax + accumulate O_R
   (interleaved to overlap CUDA Core + Tensor Core)
```

**Key Optimizations:**
1. Fine-grained TMA copy → GEMM pipelining (64×64 blocks)
2. Cache hints: `EVICT_FIRST` for better L2 hit rate
3. Programmatic dependent launch for splitkv + combine overlap
4. Tile scheduler for SM load balancing

#### Usage Example

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

# Get tile scheduler metadata once
tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,  # Query heads per KV head
    h_kv,               # Number of KV heads
    h_q,                # Number of query heads
    is_fp8,             # Use FP8 KV cache?
    topk,               # For sparse attention
)

# Decoding loop
for layer in range(num_layers):
    o, lse = flash_mla_with_kvcache(
        q, kvcache, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal=True,
        is_fp8_kvcache=True,
        indices=None,  # For sparse attention
    )
```

#### Inference Recommendations

**SGLang (recommended):**
```bash
# BF16, TP=8
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Chat --tp 8

# FP8 with FP8 KV cache
python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Chat \
    --tp 8 --quant fp8 --kv-cache-dtype fp8_e5m2
```

**vLLM:**
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

#### Infrastructure Connection

**→ FlashMLA (`flash_mla/`):** The optimized kernels that make MLA efficient in production:
- Dense decoding kernel: SM90 (Hopper), BF16
- Sparse decoding kernel: SM90/SM100, FP8 KV cache
- Sparse prefill kernel: SM90/SM100, for DSA

---

### Module 1.3: DeepSeek-V3 - Production Scale FP8 Training

**Status:** Complete

#### Model Specifications (arXiv:2412.19437)

| Metric | Value |
|--------|-------|
| Total Parameters | 671B (+14B MTP module = 685B on HF) |
| Activated Parameters | 37B per token |
| Training Tokens | 14.8T |
| Training Cost | 2.788M H800 GPU hours |
| Context Length | 128K |
| Architecture | MLA + DeepSeekMoE |

#### Key Innovations

**1. Auxiliary-Loss-Free Load Balancing**

Traditional MoE uses auxiliary loss to balance expert loads, but this hurts performance. V3's solution:
- No explicit auxiliary loss term
- Bias-based dynamic balancing during routing
- Maintains load balance without degrading model quality

**2. Multi-Token Prediction (MTP)**

Instead of predicting just the next token, V3 predicts multiple future tokens:

```
Standard LM:   P(t_{n+1} | t_1...t_n)
MTP:           P(t_{n+1}, t_{n+2}, ... t_{n+k} | t_1...t_n)
```

**MTP Architecture:**
- 14B parameter MTP module (separate from main model)
- Used during training for stronger gradients
- Enables speculative decoding during inference

**MTP for Speculative Decoding:**
```
1. Main model generates draft tokens via MTP head
2. Verify draft tokens in parallel
3. Accept correct predictions, reject and regenerate others
4. Throughput improvement: ~1.5-2x
```

**3. FP8 Mixed Precision Training**

First validated at 671B scale! Key aspects:

```
FP8 Formats:
├── E4M3 (4-bit exp, 3-bit mantissa): For weights/activations
│   - Range: [-448, 448]
│   - Best for forward pass
└── E5M2 (5-bit exp, 2-bit mantissa): For gradients
    - Range: [-57344, 57344]  
    - Better dynamic range for backprop
```

**What stays in higher precision:**
- Attention softmax
- Layer normalization
- Loss computation
- Optimizer states (FP32)

**FP8→BF16 Conversion:**
```bash
cd inference
python fp8_cast_bf16.py \
    --input-fp8-hf-path /path/to/fp8_weights \
    --output-bf16-hf-path /path/to/bf16_weights
```

#### Infrastructure Deep Dive

**→ DeepGEMM (`deep_gemm/`):**

FP8 GEMM kernels powering V3 training and inference:

| Feature | Performance |
|---------|-------------|
| FP8 GEMM | Up to 1550 TFLOPS on H800 |
| Grouped GEMM | For MoE (contiguous + masked layouts) |
| JIT Compilation | Runtime kernel compilation |
| Hardware | SM90 (Hopper), SM100 (Blackwell) |

Key interfaces:
```python
import deep_gemm

# Normal FP8 GEMM: D = C + A @ B.T
deep_gemm.fp8_gemm_nt(A, B, C, D, scale_a, scale_b)

# MoE grouped GEMM (contiguous layout for prefill)
deep_gemm.m_grouped_fp8_gemm_nt_contiguous(...)

# MoE grouped GEMM (masked layout for decode with CUDA graph)
deep_gemm.m_grouped_fp8_gemm_nt_masked(...)
```

**→ DualPipe (`dualpipe/`):**

Bidirectional pipeline parallelism for training:

| Method | Bubble | Memory |
|--------|--------|--------|
| 1F1B | (PP-1)(F+B) | 1× params |
| ZB1P | (PP-1)(F+B-2W) | 1× params |
| **DualPipe** | (PP/2-1)(F&B+B-3W) | 2× params |

Key insight: **Full forward-backward overlap** by:
1. Running forward pass in one direction
2. Running backward pass in reverse direction
3. Overlapping computation and communication

```
PP=8, micro-batches=20:
Forward:  [0]->[1]->[2]->[3]->[4]->[5]->[6]->[7]
Backward: [7]->[6]->[5]->[4]->[3]->[2]->[1]->[0]
         (interleaved with forward)
```

**→ DeepEP (`deep_ep/`):**

Expert parallelism communication for MoE:

```
Token flow in MoE:
1. DISPATCH: Tokens → Selected Experts (all-to-all)
2. COMPUTE:  Expert FFN computation
3. COMBINE:  Expert outputs → Original positions (all-to-all)

DeepEP optimizations:
- NVLink domain: 153 GB/s
- RDMA domain: 58 GB/s
- Low-latency decode: 77-194 μs
- Hook-based overlap: Zero SM usage
```

#### Inference Deployment

**Hardware Requirements:**
- BF16: 8× 80GB GPUs (e.g., 8× H100)
- FP8: Can fit on fewer GPUs

**Recommended: SGLang**
```bash
# BF16, tensor parallelism = 8
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 --tp 8

# FP8 with FP8 KV cache
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 --tp 8 \
    --quant fp8 --kv-cache-dtype fp8_e5m2

# Multi-node (2 nodes × 8 GPUs)
# See: https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3
```

**vLLM:**
```python
from vllm import LLM
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    max_model_len=8192,
    trust_remote_code=True,
)
```

#### Training Stability Notes

From the paper:
- **No irrecoverable loss spikes** throughout training
- **No rollbacks required**
- FP8 training proved remarkably stable at 671B scale

This is significant because FP8 training at this scale was previously unvalidated.

---

### Module 1.4: DeepSeek-R1 - Reasoning via Reinforcement Learning

**Status:** Complete

#### Key Innovation (arXiv:2501.12948)

**First model to validate:** Reasoning capabilities can emerge purely through RL, without supervised fine-tuning!

**Model Family:**

| Model | Type | Base | Params |
|-------|------|------|--------|
| R1-Zero | RL only | V3-Base | 671B (37B active) |
| R1 | RL + cold-start | V3-Base | 671B (37B active) |
| R1-Distill-* | Distilled | Qwen/Llama | 1.5B-70B |

#### R1-Zero: Pure RL Reasoning

**Training Approach:**
```
V3-Base Model → Large-scale RL → R1-Zero
               (no SFT!)
```

**Emergent Behaviors:**
1. **Self-verification:** Model checks its own work
2. **Reflection:** Reconsiders approach when stuck
3. **Long Chain-of-Thought:** Extended reasoning traces
4. **Language mixing:** (a challenge to address)

**Challenges with R1-Zero:**
- Endless repetition
- Poor readability
- Language mixing between English/Chinese

#### R1: Full Pipeline

**4-Stage Training Pipeline:**

```
Stage 1: RL Stage 1 (Reasoning Patterns)
         │  - Large-scale RL on base model
         │  - Discovers reasoning patterns
         ↓
Stage 2: SFT Stage 1 (Seed Capabilities)
         │  - Cold-start data injection
         │  - Seeds reasoning + non-reasoning abilities
         ↓
Stage 3: RL Stage 2 (Human Preference)
         │  - Align with human preferences
         │  - Improve output quality
         ↓
Stage 4: SFT Stage 2 (Polish)
         │  - Final capability refinement
         ↓
DeepSeek-R1
```

**Key Insight:** Cold-start data before RL addresses R1-Zero's challenges while preserving emergent reasoning.

#### Distillation: Smaller Models Can Reason

**Distilled Models:**

| Model | Base | AIME 2024 | MATH-500 | Codeforces |
|-------|------|-----------|----------|------------|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 28.9 | 83.9 | 954 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 55.5 | 92.8 | 1189 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | **72.6** | 94.3 | 1691 |
| R1-Distill-Llama-70B | Llama-3.3-70B | 70.0 | **94.5** | 1633 |

**Key Finding:** Distilled models outperform models that discover reasoning through RL directly on small models.

#### Usage Recommendations

**Critical Settings:**
```python
# Temperature: 0.5-0.7 (0.6 recommended)
# NO system prompt!
# Force thinking with "<think>\n" prefix

generation_config = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 32768,
}

# For math problems, add to prompt:
prompt += "\nPlease reason step by step, and put your final answer within \\boxed{}."
```

**Forcing Deep Thinking:**
```python
# Ensure model engages in reasoning
response_prefix = "<think>\n"
```

**Deployment:**
```bash
# Distilled models (vLLM)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager

# Full R1 (SGLang)
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1 \
    --tp 8 \
    --trust-remote-code
```

#### Thinking Tag Structure

```xml
<think>
[Extended reasoning process]
- Step-by-step problem analysis
- Self-verification
- Reflection and correction
- Multiple approaches explored
</think>

[Final answer in natural language]
```

#### Performance Comparison

| Benchmark | o1-mini | o1 | R1 |
|-----------|---------|-----|-----|
| AIME 2024 | 63.6 | 79.2 | **79.8** |
| MATH-500 | 90.0 | 96.4 | **97.3** |
| Codeforces | 1820 | **2061** | 2029 |
| GPQA-Diamond | 60.0 | **75.7** | 71.5 |

#### What's Missing for Reproduction

1. **RL Training Code** - Reference alternatives:
   - OpenRLHF
   - TRL (Hugging Face)
   - veRL
   
2. **Cold-start Data** - Not released

3. **Reward Model** - Not released

---

### Module 1.5: DeepSeekMath-V2 - Self-Verifiable Mathematical Reasoning

**Status:** Complete

#### Core Innovation

**Problem:** Correct answers ≠ correct reasoning. Final-answer-only rewards can't verify proof rigor.

**Solution:** Self-verifiable mathematical reasoning through:
1. LLM-based verifier for theorem proving
2. Generator trained with verifier as reward model
3. Scaled verification compute for hard proofs

#### Key Results

| Competition | Score |
|-------------|-------|
| IMO 2025 | Gold-level |
| CMO 2024 | Gold-level |
| Putnam 2024 | 118/120 |

#### Architecture

**Based on:** DeepSeek-V3.2-Exp-Base (with DSA attention)

**Multi-Round Pipeline:**

```
Round R:
├── Proof Generation
│   ├── Template: proof_gen_template
│   ├── Temperature: 1.0
│   ├── Max tokens: 128K
│   └── Output: proofs with self-evaluation
│
├── Proof Verification  
│   ├── Template: proof_verification_template
│   ├── Multiple verifications per proof (n=4)
│   └── Scores each proof 0-1
│
├── Meta Verification (optional)
│   ├── Verifies low-scoring verifications
│   └── Quality control on ratings
│
└── Proof Refinement → Next Round
    ├── Select best proofs (top n_best)
    ├── Combine with previous ratings
    └── Generate improved proofs
```

#### Inference Code Structure

```
inference/
├── main.py           # Multi-round orchestration
├── generate.py       # API calls for generation
├── math_templates.py # Prompts for gen/verify/refine
├── utils.py          # Helper functions
└── run.sh           # Example launch script
```

**Key Parameters:**
```python
parser.add_argument("--proof_gen_temp", default=1.0)
parser.add_argument("--proof_gen_max_len", default=128*1024)
parser.add_argument("--n_best_proofs_to_sample", default=32)
parser.add_argument("--n_verification_per_proof", default=4)
parser.add_argument("--max_rounds", default=20)
```

#### Input/Output Format

**Input (IMO2025.json):**
```json
{
    "id": 1,
    "question": "A line in the plane is called sunny if...",
    "answer": "k = 0, 1, 3 for all n",
    "contest": "IMO2025",
    "problem_idx": "IMO2025-1"
}
```

**Output:** Proofs with verification scores in `.jsonl` format

#### Self-Evaluation in Proofs

Generator outputs include self-evaluation:
```
<think>
[Extended reasoning]
</think>

[Proof content]

<self_eval>
[Model evaluates own proof]
Score: \boxed{0.85}
</self_eval>
```

**Extraction:**
```python
self_eval = extract_self_eval(proof)
proof_content = extract_solution(proof)
score = float(extract_boxed_answers(self_eval)[-1])
```

#### Scaling Test-Time Compute

Key insight: Scale verification compute for hard problems:
1. Generate many proof candidates
2. Run multiple verifications per proof
3. Use meta-verification for quality control
4. Refine based on aggregated feedback
5. Repeat for multiple rounds (up to 20)

---

## Phase 2: Newer Architectures

### Module 2.1: DeepSeek-V3.2-Exp - DeepSeek Sparse Attention (DSA)

**Status:** Complete

#### Overview

V3.2-Exp introduces **DeepSeek Sparse Attention (DSA)** - the first fine-grained sparse attention achieving substantial long-context efficiency improvements while maintaining output quality.

**Model:** 685B total parameters, built on V3.1-Terminus

#### DSA Mechanism

**Dense vs Sparse Attention:**
```
Dense:  Q @ K.T → Full attention matrix → Softmax → @ V
Sparse: Q @ K.T[selected_indices] → Sparse computation → @ V[selected_indices]
```

**DSA Components:**

1. **Indexer Module:** Selects which tokens each query should attend to
   - Produces `indices` tensor: `[batch, seq_q, topk]`
   - Uses weighted ReLU MQA logits (see DeepGEMM PR #200)

2. **Sparse Attention Kernel:** Computes attention only on selected tokens
   - FlashMLA PR #98 implements these kernels
   - 640 TFLOPS prefill, 410 TFLOPS decode

**Indexer Architecture:**
```
Query → Indexer → Select Top-K Token Indices
                        ↓
                  indices tensor
                        ↓
      Sparse Attention (only attend to selected KV)
```

**Important Implementation Note:**
> RoPE in indexer module requires **non-interleaved** layout, while RoPE in MLA module expects **interleaved** layout.

#### Performance

| Benchmark | V3.1-Terminus | V3.2-Exp |
|-----------|---------------|----------|
| AIME 2025 | 88.4 | 89.3 |
| Codeforces | 2046 | 2121 |
| LiveCodeBench | 74.9 | 74.1 |
| GPQA-Diamond | 80.7 | 79.9 |

**Key Result:** Virtually identical quality with significant efficiency gains.

#### Supporting Kernels

**DeepGEMM (`deep_gemm/`):**
```python
# Indexer logit computation (PR #200)
deep_gemm.fp8_mqa_logits(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end)

# For each query token, computes:
# out[i,j] = ReLU(q[i] @ kv[j]) * weights[i] summed over heads
```

**FlashMLA (`flash_mla/`):**
```python
# Sparse prefill (PR #98)
flash_mla.flash_mla_sparse_fwd(q, kv, indices, sm_scale)

# Sparse decoding with FP8 KV cache
flash_mla.flash_mla_with_kvcache(..., indices=indices)
```

#### Deployment

```bash
# SGLang (recommended)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3.2-Exp \
    --tp 8 --dp 8 \
    --enable-dp-attention

# Weight conversion for demo
cd inference
python convert.py \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --save-path ${SAVE_PATH} \
    --n-experts 256 \
    --model-parallel ${MP}
```

---

### Module 2.2: Engram - Conditional Memory

**Status:** Complete

#### Core Innovation (arXiv:2601.07372)

**Key Insight:** MoE provides conditional *computation*, but Transformers lack conditional *memory* lookup.

**Solution:** Engram module provides O(1) N-gram embedding lookup as a complementary sparsity axis.

#### Architecture

```
Input Tokens → N-gram Hash → Embedding Table Lookup → Fuse with Hidden States
                    ↓                                        ↓
              O(1) lookup                              MoE/Attention
              (deterministic)                          (neural computation)
```

**Trade-off Discovery: U-Shaped Scaling Law**

```
Loss
  │
  │    *           *
  │   * *         * *
  │  *   *       *   *
  │ *     *     *     *
  │*       *****       *
  └──────────────────────→
   100% MoE    ←→    100% Memory
   
Optimal: Mix of both neural compute AND static memory
```

#### Key Results (Engram-27B)

Under iso-parameter and iso-FLOPs constraints:
- Consistent improvements over MoE baselines
- Benefits across knowledge, reasoning, code, and math domains
- Relieves early layers from static pattern reconstruction
- Preserves effective depth for complex reasoning

#### System Efficiency

**Deterministic Addressing Benefits:**
- Massive embedding tables can be offloaded to host memory
- Minimal inference overhead due to predictable access patterns
- No routing computation needed (unlike MoE)

#### Demo Code

```bash
pip install torch numpy transformers sympy
python engram_demo_v1.py
```

**Note:** Demo mocks standard components (Attention/MoE/mHC) to focus on Engram module logic.

---

## Phase 3: Multi-Modal Models

### Module 3.1: DeepSeek-VL2 - Vision-Language MoE

**Status:** Complete

#### Overview (External Repo: github.com/deepseek-ai/DeepSeek-VL2)

MoE-based Vision-Language model for advanced multimodal understanding.

**Model Variants:**

| Variant | Total | Activated | Use Case |
|---------|-------|-----------|----------|
| VL2-Tiny | 3.37B | 1B | Edge deployment |
| VL2-Small | 16.1B | 2.4B | Balanced |
| VL2 | 27.5B | 4.2B | Best quality |

#### Key Capabilities

1. **Multi-image conversations**
2. **Visual grounding** with bounding boxes
3. **Document understanding**
4. **OCR capabilities**
5. **Incremental prefilling** for memory efficiency

#### Architecture

```
Images → Vision Encoder → Projector → MoE Language Model → Response
                                              ↓
                                      [Grounding Boxes]
```

#### Usage (from GitHub)

```python
# Single image
from deepseek_vl2 import DeepSeekVL2

model = DeepSeekVL2.from_pretrained("deepseek-ai/deepseek-vl2")
response = model.chat(
    image="path/to/image.jpg",
    prompt="What's in this image?"
)

# Visual grounding
response = model.chat(
    image="path/to/image.jpg", 
    prompt="<|ref|>Find the red car<|/ref|>"
)
# Returns: coordinates of bounding box
```

#### Incremental Prefilling

For 40GB GPU deployment:
```python
# Process image in chunks to reduce memory
model.incremental_prefill(
    images=images,
    chunk_size=1024
)
```

---

### Module 3.2: DeepSeek-OCR-2 - Document Understanding

**Status:** Complete

#### Overview (arXiv:2601.20552)

Document OCR with layout understanding, converting images to structured markdown.

**Model:** 3B parameters (DeepSeek LLM backbone)

#### Dual Encoder Architecture

```
Image → Dynamic Crop → SAM ViT-B → Qwen2 Encoder → MLP Projector → LLM
             ↓              ↓              ↓
        Global view    Local patches   Feature fusion
       (1024×1024)    (up to 6×768²)
```

**Key Components:**

1. **SAM ViT-B:** Visual feature extraction
2. **Qwen2-as-Encoder:** Process SAM features as a decoder-to-encoder
3. **MLP Projector:** Linear projection to LLM dimension
4. **DeepSeek LLM:** Generate markdown output

#### Dynamic Resolution

```python
# From config.py
BASE_SIZE = 1024      # Global view resolution
IMAGE_SIZE = 768      # Local crop resolution
MIN_CROPS = 2
MAX_CROPS = 6         # Maximum local views
CROP_MODE = True      # Enable dynamic cropping
```

**Token Calculation:**
```python
# Global: 1 view × (1024/16/4)² = 256 tokens
# Local:  N views × (768/16/4)² = N × 144 tokens
# Total:  256 + N×144 (N ≤ 6)
```

#### Usage

**Single Image:**
```bash
python run_dpsk_ocr2_image.py \
    --image_path /path/to/image.png \
    --output_path /path/to/output/
```

**PDF Batch Processing:**
```bash
python run_dpsk_ocr2_pdf.py \
    --input_path /path/to/document.pdf \
    --output_path /path/to/output/
```

#### Output Format

```markdown
# Document Title

## Section 1
Content with **formatting** preserved...

| Table | Headers |
|-------|---------|
| Data  | Values  |

<box>(x1,y1,x2,y2)</box>  # Grounding coordinates
```

---

### Module 3.3: Janus - Unified Understanding + Generation

**Status:** Complete

#### Overview (External Repo: github.com/deepseek-ai/Janus)

First unified multimodal model for both **understanding** AND **generation**.

**Models:**
- Janus-1.3B, JanusFlow-1.3B
- Janus-Pro-1B, Janus-Pro-7B

#### Key Innovation: Decoupled Visual Encoding

```
Understanding Path:
    Image → SigLIP Encoder → Projector → LLM
    
Generation Path:
    LLM → VQ Tokenizer → Decoder → Generated Image
```

**Why Decoupled?**
- Understanding needs semantic features (high-level)
- Generation needs pixel-level details (low-level)
- Different encoders optimize for each task

#### JanusFlow: Rectified Flow

Replaces autoregressive image generation with rectified flow:
- Faster generation
- Better image quality
- More efficient sampling

#### Usage

**Understanding:**
```python
from janus import Janus

model = Janus.from_pretrained("deepseek-ai/Janus-Pro-7B")
response = model.understand(
    image="image.jpg",
    prompt="Describe this image"
)
```

**Generation:**
```python
image = model.generate(
    prompt="A cute cat in a garden",
    cfg_scale=7.5,
    steps=50
)
image.save("generated.png")
```

---

## Phase 4: Agentic Coding

### Module 4.1: DeepSeek-Coder - Foundation Code LLM

**Status:** Complete

#### Overview (External: github.com/deepseek-ai/DeepSeek-Coder)

Code-focused LLM trained from scratch on 2T tokens (87% code, 13% NL).

**Models:** 1B, 5.7B, 6.7B, 33B (Base + Instruct)

#### Key Features

1. **86 programming languages**
2. **16K context length**
3. **Fill-in-the-Middle (FIM)** training
4. **Repo-level code completion**

#### FIM Training

Enables code insertion:
```python
# FIM tokens
<｜fim▁begin｜>  # Start of context
<｜fim▁hole｜>   # Where to insert
<｜fim▁end｜>    # End of context

# Example
prompt = '''<｜fim▁begin｜>def hello():
<｜fim▁hole｜>
    print(greeting)<｜fim▁end｜>'''
# Model generates: '    greeting = "Hello, World!"'
```

#### Data Pipeline

```
GitHub → StarCoder Filter Rules → Parse Dependencies
      → Rearrange by Dependencies → Repo-level Dedup
      → Quality Filter → 2T Tokens
```

#### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct"
)

# Code completion
response = model.generate(
    tokenizer("def quicksort(arr):", return_tensors="pt").input_ids
)
```

---

### Module 4.2: DeepSeek-Coder-V2 - MoE Code Intelligence

**Status:** Complete

#### Overview (External: github.com/deepseek-ai/DeepSeek-Coder-V2)

MoE code model achieving GPT-4 Turbo parity.

**Models:**
- Lite: 16B total, 2.4B activated
- Full: 236B total, 21B activated

#### Key Improvements over V1

| Feature | Coder | Coder-V2 |
|---------|-------|----------|
| Languages | 86 | 338 |
| Context | 16K | 128K |
| Architecture | Dense | MoE |
| Training | 2T tokens | V2 + 6T tokens |

#### Architecture

Based on DeepSeek-V2 (MLA + DeepSeekMoE):
```
Continue pre-training from V2 checkpoint
      + Additional 6T code/math tokens
      → Coder-V2
```

#### Performance

| Benchmark | Coder-V2-Lite | Coder-V2 |
|-----------|---------------|----------|
| HumanEval | 90.2% | 90.2% |
| MBPP | 78.3% | 80.1% |
| LiveCodeBench | 43.4% | - |
| SWE-Bench | - | Competitive |

#### Agentic Capabilities

1. **Code fixing** (Defects4J, SWE-Bench)
2. **Multi-turn coding conversations**
3. **Tool use patterns**

#### Deployment

```bash
# SGLang
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-Coder-V2-Instruct \
    --tp 8

# vLLM  
vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --tensor-parallel-size 2
```

---

## Phase 5: Infrastructure Deep Dives

### Module 5.1: DeepEP - Expert Parallelism Communication

**Status:** Complete

#### Overview

High-throughput, low-latency all-to-all GPU kernels for MoE dispatch/combine.

#### Performance

**Normal Kernels (Training/Prefill):**

| Type | EP Size | Bandwidth |
|------|---------|-----------|
| Intranode | 8 | 153 GB/s (NVLink) |
| Internode | 32 | 58 GB/s (RDMA) |
| Internode | 64 | 51 GB/s (RDMA) |

**Low-Latency Kernels (Decoding):**

| EP Size | Dispatch Latency | Combine Latency |
|---------|------------------|-----------------|
| 8 | 77 μs | 114 μs |
| 64 | 173 μs | 314 μs |
| 256 | 194 μs | 360 μs |

#### Key Interfaces

```python
from deep_ep import Buffer, EventOverlap

# Initialize buffer
buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

# Dispatch (tokens → experts)
recv_x, recv_idx, recv_weights, counts, handle, event = \
    buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights, ...)

# Combine (expert outputs → original positions)
combined_x, _, event = buffer.combine(x, handle, ...)

# Low-latency mode for decoding
recv_hidden, recv_count, handle, event, hook = \
    buffer.low_latency_dispatch(hidden_states, topk_idx, ...)
```

#### Hook-Based Overlap

Zero SM usage for communication-computation overlap:
```python
# Hook returns after RDMA transfer completes
# No GPU compute resources consumed during network transfer
recv_x, ..., hook = buffer.low_latency_dispatch(..., return_recv_hook=True)
# Later: hook() to complete receive
```

---

### Module 5.2: DeepGEMM - Efficient Matrix Operations

**Status:** Complete

*Covered in Module 1.3 (V3) - Summary:*

- **FP8/BF16 GEMMs** with JIT compilation
- **Grouped GEMMs** for MoE (contiguous + masked)
- **Up to 1550 TFLOPS** on H800
- **SM90/SM100 support**

Key APIs: `fp8_gemm_nt`, `m_grouped_fp8_gemm_nt_contiguous`, `m_grouped_fp8_gemm_nt_masked`

---

### Module 5.3: FlashMLA - Attention Kernels

**Status:** Complete

*Covered in Module 1.2 (V2) - Summary:*

- **Dense MLA decoding:** 660 TFLOPS
- **Sparse MLA (DSA):** 640 TFLOPS prefill, 410 TFLOPS decode
- **FP8 KV cache** support
- **Seesaw scheduling** for compute-bound workloads

Key APIs: `get_mla_metadata`, `flash_mla_with_kvcache`, `flash_mla_sparse_fwd`

---

### Module 5.4: DualPipe - Pipeline Parallelism

**Status:** Complete

*Covered in Module 1.3 (V3) - Summary:*

- **Bidirectional scheduling** for forward-backward overlap
- **Reduced pipeline bubbles:** (PP/2-1)(F&B+B-3W)
- **DualPipeV variant:** V-shape schedule

Key benefit: Full computation-communication overlap in training.

---

### Module 5.5: 3FS - Distributed File System

**Status:** Complete

#### Overview

High-performance distributed file system for AI workloads.

**Peak Performance:**
- **6.6 TiB/s** aggregate read throughput (180 nodes)
- **GraySort:** 3.66 TiB/min

#### Key Features

1. **Disaggregated Architecture:** SSDs + RDMA combined
2. **Strong Consistency:** CRAQ (Chain Replication with Apportioned Queries)
3. **File Interface:** Standard POSIX-like API

#### Use Cases

| Workload | Benefit |
|----------|---------|
| **Dataloaders** | Random access to training samples |
| **Checkpointing** | High-throughput parallel saves |
| **KVCache** | Offload inference cache from DRAM |

#### KVCache for Inference

Alternative to DRAM-based caching:
- Much larger capacity (SSD vs DRAM)
- Up to 40 GiB/s per client node
- Cost-effective for inference at scale

#### Setup

```bash
# Clone and build
git clone https://github.com/deepseek-ai/3fs
cd 3fs
git submodule update --init --recursive
./patches/apply.sh

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSHUFFLE_METHOD=g++11
cmake --build build -j 32
```

**Dependencies:**
- FoundationDB 7.1+
- libfuse 3.16+
- Rust 1.75+

---

## Summary: Gaps & External Resources

### Missing from Repos

| Gap | Recommended Resource |
|-----|---------------------|
| Pre-training code | Megatron-LM, NeMo Framework |
| Data preprocessing | DataTrove, RedPajama pipelines |
| RL/PPO training | OpenRLHF, TRL, veRL |
| Evaluation harness | lm-evaluation-harness, HELM |
| Tensor parallelism | Megatron-Core |
| Quantization | llama.cpp, AutoGPTQ, AWQ |

### Research Directions

1. **Scaling Engram:** Integrate with V3.2 sparse attention
2. **Unified Coding Agent:** Coder-V2 + tool use + verification
3. **Multimodal Reasoning:** Apply R1 to VL2/Janus
4. **Efficient Sparse Training:** Train DSA from scratch
5. **Cross-model Distillation:** Transfer OCR to general VLM



