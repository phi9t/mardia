# DeepSeek-Math: Deep Dive

> Paper: arXiv:2402.03300 (February 2024)  
> "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"

## Overview

DeepSeek-Math demonstrates that **continued pre-training on math-specific data** combined with **reinforcement learning** can achieve frontier-level mathematical reasoning in open-source models. The 7B model approaches GPT-4's math performance through three key innovations:

1. **Mathematical corpus curation** (120B tokens from Common Crawl)
2. **Group Relative Policy Optimization (GRPO)** - efficient RL without critic model
3. **Tool-integrated reasoning** via Program-of-Thought

## The Mathematical Pre-Training Corpus

### The Data Challenge

Mathematical reasoning requires exposure to structured, rigorous content. Web data is dominated by informal text, making math content rare:

```
Common Crawl composition:
├── General web text:     ~95%
├── Code:                 ~3%
├── Math-related:         ~0.1%  ← needle in haystack
└── High-quality math:    ~0.01%
```

### DeepSeekMath Corpus Pipeline

```
Common Crawl (hundreds of TB)
         │
         ▼
    ┌─────────────────────────────────────────┐
    │  Step 1: Seed Domain Selection          │
    │  - OpenWebMath domains as positive      │
    │  - Random web pages as negative         │
    │  - Train fastText classifier            │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  Step 2: Iterative Expansion            │
    │  - Apply classifier to Common Crawl     │
    │  - Human review of high-scoring pages   │
    │  - Add new positive domains             │
    │  - Retrain classifier (4 iterations)    │
    └────────────────┬────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────┐
    │  Step 3: Deduplication & Filtering      │
    │  - URL deduplication                    │
    │  - Near-duplicate removal (MinHash)     │
    │  - Quality filtering                    │
    └────────────────┬────────────────────────┘
                     │
                     ▼
         DeepSeekMath Corpus
              120B tokens
```

### Corpus Composition

| Source Type | Examples | Tokens |
|-------------|----------|--------|
| Math Q&A sites | Math StackExchange, AoPS | ~30B |
| Educational | Khan Academy, MIT OCW | ~25B |
| Academic papers | arXiv (math subset) | ~20B |
| Textbooks | Open textbooks | ~15B |
| Code with math | NumPy docs, SciPy | ~15B |
| Other math pages | Encyclopedias, tutorials | ~15B |

**Key insight**: The iterative domain expansion discovered math content that keyword-based filtering would miss (e.g., physics derivations, economics proofs).

## Model Architecture & Training

### Base Model

DeepSeek-Math starts from DeepSeek-Coder-v1.5 7B (not a general LLM):

| Property | Value |
|----------|-------|
| Parameters | 7B |
| Architecture | Dense Transformer |
| Context Length | 4K |
| Vocabulary | 32K tokens |
| Base Model | DeepSeek-Coder-v1.5 |

**Why start from a code model?**
- Code models understand structured syntax
- Math notation shares structure with code
- Better at following precise instructions
- Program-of-Thought naturally integrates

### Pre-Training Details

```python
# Training configuration
config = {
    "tokens": "500B",          # 120B math + 380B general
    "batch_size": 4M tokens,
    "learning_rate": 1e-4,     # Peak
    "lr_schedule": "cosine",
    "warmup": 2000 steps,
    "context_length": 4096,
}

# Data mixture
mixture = {
    "math_corpus": 0.24,       # 120B tokens
    "code": 0.30,              # Maintain coding ability
    "general": 0.46,           # Prevent forgetting
}
```

## Group Relative Policy Optimization (GRPO)

### The RL Challenge

Traditional RLHF requires a reward model + PPO, which is computationally expensive:

```
Standard PPO:
┌──────────────────────────────────────────────────────────────┐
│  Policy Model ──► Generate ──► Reward Model ──► Update       │
│       ↑              │              │              │         │
│       │              │              ▼              │         │
│       │              │        Value Model ◄───────┘         │
│       │              │              │                        │
│       └──────────────┴──────────────┴─── Critic (expensive!) │
└──────────────────────────────────────────────────────────────┘
```

### GRPO Innovation

GRPO eliminates the value/critic model by using **group-relative baselines**:

```
GRPO:
┌──────────────────────────────────────────────────────────────┐
│  For each question q:                                        │
│                                                              │
│  1. Sample G outputs: o₁, o₂, ..., o_G                      │
│                                                              │
│  2. Get rewards: r₁, r₂, ..., r_G                           │
│                                                              │
│  3. Compute group statistics:                                │
│     μ = mean(r₁...r_G)                                       │
│     σ = std(r₁...r_G)                                        │
│                                                              │
│  4. Normalize advantages:                                    │
│     Â_i = (r_i - μ) / σ                                      │
│                                                              │
│  5. Policy gradient:                                         │
│     ∇J = Σ_i Â_i × ∇log π(o_i|q)                            │
└──────────────────────────────────────────────────────────────┘
```

### Mathematical Formulation

```
Standard PPO objective:
J_PPO(θ) = E[min(ρ_t × Â_t, clip(ρ_t, 1-ε, 1+ε) × Â_t)]
where Â_t = r_t + γV(s_{t+1}) - V(s_t)  ← requires value function!

GRPO objective:
J_GRPO(θ) = E_q [E_{o~π_old}[min(ρ × Â, clip(ρ, 1-ε, 1+ε) × Â)]]
where Â = (r - μ_group) / σ_group  ← no value function needed!

ρ = π_θ(o|q) / π_old(o|q)  (importance ratio)
```

### Why GRPO Works for Math

1. **Sparse rewards**: Math has clear right/wrong answers
2. **Group comparison**: Relative ranking within samples is informative
3. **Efficiency**: No separate critic training (saves 50% compute)
4. **Stability**: Group normalization reduces variance

### GRPO Hyperparameters

```python
grpo_config = {
    "group_size": 64,          # Samples per question
    "clip_epsilon": 0.2,       # PPO clip range
    "kl_coef": 0.02,          # KL penalty coefficient
    "learning_rate": 1e-6,     # Small for stability
    "epochs_per_question": 1,
    "reward": "binary",        # 1 if correct, 0 otherwise
}
```

## Tool-Integrated Reasoning

### Program-of-Thought (PoT)

Instead of generating natural language reasoning, the model generates Python code:

```
Chain-of-Thought (CoT):
Q: What is the sum of the first 100 positive integers?
A: The sum of first n integers is n(n+1)/2.
   For n=100: 100 × 101 / 2 = 5050
   Answer: 5050

Program-of-Thought (PoT):
Q: What is the sum of the first 100 positive integers?
A: ```python
   n = 100
   result = n * (n + 1) // 2
   print(result)  # 5050
   ```
   Answer: 5050
```

### When PoT Helps

| Problem Type | CoT | PoT | Winner |
|--------------|-----|-----|--------|
| Arithmetic | Good | Better | PoT |
| Algebra | Good | Better | PoT |
| Geometry | Better | Limited | CoT |
| Combinatorics | Good | Better | PoT |
| Proofs | Better | Limited | CoT |

### Hybrid Approach

DeepSeek-Math learns to choose the right tool:

```python
def solve(question):
    if requires_symbolic_computation(question):
        # Generate Python with SymPy
        code = model.generate_code(question)
        result = execute(code)
    elif requires_numerical(question):
        # Generate Python with NumPy
        code = model.generate_code(question)
        result = execute(code)
    else:
        # Pure reasoning
        result = model.generate_cot(question)
    return result
```

## Training Pipeline

### Three-Stage Process

```
Stage 1: Continued Pre-Training
├── Data: 120B math + 380B general
├── Duration: ~2 weeks on cluster
└── Output: DeepSeekMath-Base

         │
         ▼

Stage 2: Supervised Fine-Tuning (SFT)
├── Data: Chain-of-Thought solutions
│         + Program-of-Thought solutions
├── Sources: GSM8K, MATH, AoPS, synthetic
└── Output: DeepSeekMath-Instruct

         │
         ▼

Stage 3: Reinforcement Learning (GRPO)
├── Reward: Binary (correct/incorrect)
├── Questions: Diverse math problems
├── Group size: 64 samples per question
└── Output: DeepSeekMath-RL
```

### SFT Data Composition

```python
sft_data = {
    # Chain-of-Thought
    "gsm8k_cot": 7473,
    "math_cot": 7500,
    "aops_cot": 25000,
    
    # Program-of-Thought
    "gsm8k_pot": 7473,
    "math_pot": 7500,
    "synthetic_pot": 50000,
    
    # Tool use
    "sympy_examples": 10000,
    "numpy_examples": 5000,
}
```

## Benchmark Results

### Competition Math (MATH Dataset)

| Model | MATH (%) | Improvement |
|-------|----------|-------------|
| GPT-4 | 42.5 | - |
| Gemini Ultra | 53.2 | - |
| **DeepSeekMath-7B-RL** | **51.7** | - |
| Mixtral-8x7B | 28.4 | - |
| LLaMA-2-70B | 13.5 | - |

**7B model achieves 51.7% on MATH, approaching Gemini Ultra!**

### Grade School Math (GSM8K)

| Model | GSM8K (%) |
|-------|-----------|
| DeepSeekMath-7B-RL | 88.2 |
| DeepSeekMath-7B-Instruct | 82.9 |
| DeepSeekMath-7B-Base | 64.2 |
| GPT-4 | 92.0 |

### By Problem Category (MATH)

| Category | Base | Instruct | RL |
|----------|------|----------|-----|
| Algebra | 57.1 | 67.4 | 71.2 |
| Counting/Prob | 40.2 | 51.3 | 56.8 |
| Geometry | 25.4 | 31.2 | 35.6 |
| Intermediate Algebra | 18.9 | 27.1 | 32.4 |
| Number Theory | 41.2 | 52.8 | 58.3 |
| Prealgebra | 72.3 | 81.4 | 85.1 |
| Precalculus | 28.7 | 38.9 | 44.2 |

## Key Insights

### 1. Data Quality > Quantity

The 120B math corpus outperforms mixing in more general data:

```
Experiment: Hold compute constant, vary data mix

Mix A: 120B math + 380B general → 51.7% MATH
Mix B: 60B math + 440B general  → 43.2% MATH
Mix C: 240B math (lower quality) → 45.1% MATH
```

### 2. GRPO vs PPO

```
Same compute budget:
PPO:  45.2% MATH (with critic model)
GRPO: 51.7% MATH (no critic model)

GRPO wins because:
- More training iterations (no critic overhead)
- Group normalization provides stable baselines
- Binary rewards work well for math
```

### 3. Code Pre-Training Matters

```
Starting from different bases (same math training):

DeepSeek-Coder → DeepSeekMath: 51.7% MATH
DeepSeek-LLM → Math variant:   42.1% MATH
LLaMA-2 → Math variant:        38.4% MATH
```

### 4. Tool Use is Complementary

```
DeepSeekMath-RL performance:

CoT only:  47.3% MATH
PoT only:  44.8% MATH
CoT + PoT: 51.7% MATH (best of both)
```

## Connection to Later Work

DeepSeek-Math established foundations used in subsequent models:

| Innovation | Used In |
|------------|---------|
| GRPO algorithm | DeepSeek-R1 |
| Math corpus curation | DeepSeek-V3 training |
| Tool-integrated reasoning | DeepSeek-Coder-V2 |
| Binary reward RL | All DeepSeek RL models |

### Evolution to DeepSeek-Math-V2

| Aspect | Math V1 | Math V2 |
|--------|---------|---------|
| Base Model | 7B dense | V3.2-Exp (671B MoE) |
| Focus | Answer accuracy | Proof verification |
| RL Reward | Binary (correct/wrong) | Verifier score |
| Output | Answer | Full proof |
| Best Result | 51.7% MATH | Gold IMO 2025 |

## Reproduction Notes

### What's Available
- Model weights on HuggingFace: `deepseek-ai/deepseek-math-7b-*`
- Evaluation scripts
- Benchmark datasets

### What's Not Released
- Full 120B math corpus
- GRPO training code
- Data curation pipeline
- SFT training data

### Alternatives for Reproduction
- **Math data**: OpenWebMath, Proof-Pile-2
- **RL training**: OpenRLHF, TRL
- **Evaluation**: lm-evaluation-harness

## Key Files

| Resource | Location |
|----------|----------|
| Paper | arXiv:2402.03300 |
| Weights | `deepseek-ai/deepseek-math-7b-base` |
| | `deepseek-ai/deepseek-math-7b-instruct` |
| | `deepseek-ai/deepseek-math-7b-rl` |

## Summary

DeepSeek-Math demonstrated that:

1. **Targeted pre-training** on curated math data dramatically improves reasoning
2. **GRPO** provides efficient RL without expensive critic models
3. **Tool integration** (code generation) complements pure reasoning
4. A **7B model can approach GPT-4** on mathematical benchmarks

The techniques pioneered here—especially GRPO and tool-integrated reasoning—became foundational for DeepSeek's later reasoning models including R1.
