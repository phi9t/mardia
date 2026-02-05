# DeepSeek-R1: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-R1.git  
> Commit: `0cf78561f1d51c84a21b2190626b21116d5c68bb` (2025-04-09)

## Overview

DeepSeek-R1 is the first model to demonstrate that **reasoning capabilities can emerge purely through reinforcement learning**, without supervised fine-tuning on reasoning examples.

## Model Family

| Model | Type | Base | Parameters |
|-------|------|------|------------|
| R1-Zero | RL only | V3-Base | 671B (37B active) |
| R1 | RL + cold-start | V3-Base | 671B (37B active) |
| R1-Distill-Qwen-1.5B | Distilled | Qwen2.5-Math-1.5B | 1.5B |
| R1-Distill-Qwen-7B | Distilled | Qwen2.5-Math-7B | 7B |
| R1-Distill-Qwen-32B | Distilled | Qwen2.5-32B | 32B |
| R1-Distill-Llama-70B | Distilled | Llama-3.3-70B | 70B |

## R1-Zero: Pure RL Reasoning

### Training Approach

```
V3-Base Model ──► Large-scale RL ──► R1-Zero
                  (no SFT!)
```

### Emergent Behaviors

Through pure RL, R1-Zero spontaneously develops:

1. **Self-verification**: Model checks its own work
2. **Reflection**: Reconsiders approach when stuck
3. **Extended Chain-of-Thought**: Long reasoning traces
4. **Backtracking**: Abandons failed approaches

### Challenges

- Endless repetition
- Poor readability
- Language mixing (English/Chinese)

## R1: Full Training Pipeline

### 4-Stage Process

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: RL Stage 1                                        │
│  ├─ Large-scale RL on V3-Base                              │
│  └─ Discovers reasoning patterns                            │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: SFT Stage 1                                       │
│  ├─ Cold-start data injection                              │
│  └─ Seeds reasoning + non-reasoning abilities               │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: RL Stage 2                                        │
│  ├─ Align with human preferences                           │
│  └─ Improve output quality                                  │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: SFT Stage 2                                       │
│  └─ Final capability refinement                             │
└─────────────────────────────────────────────────────────────┘
                            ▼
                      DeepSeek-R1
```

**Key insight**: Cold-start data before RL fixes R1-Zero's issues while preserving emergent reasoning.

## Distillation

### Why Distillation Works

Direct RL on small models fails to produce strong reasoning. Solution: **distill from R1**.

```python
# Distillation process
def distill(teacher, student, data):
    for prompt in data:
        # Generate reasoning trace with teacher
        teacher_output = teacher.generate(prompt, temperature=0.6)
        
        # Train student to reproduce
        student.train(prompt, teacher_output)
```

### Results

| Model | AIME 2024 | MATH-500 | Codeforces |
|-------|-----------|----------|------------|
| R1-Distill-Qwen-1.5B | 28.9 | 83.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 92.8 | 1189 |
| R1-Distill-Qwen-32B | **72.6** | 94.3 | 1691 |
| R1-Distill-Llama-70B | 70.0 | **94.5** | 1633 |

## Usage Guidelines

### Critical Settings

```python
generation_config = {
    "temperature": 0.6,      # 0.5-0.7 recommended
    "top_p": 0.95,
    "max_tokens": 32768,
}

# NO system prompt!
# Force thinking with prefix
response_prefix = "<think>\n"
```

### Prompt Engineering

```python
# For math problems
prompt = """Solve: Find all positive integers n such that n² + 1 divides n! + 1.

Please reason step by step, and put your final answer within \\boxed{}."""

# Start response with thinking tag
response = model.generate(prompt, prefix="<think>\n")
```

### Output Structure

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

## Deployment

### Distilled Models (vLLM)

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --enforce-eager
```

### Full R1 (SGLang)

```bash
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1 \
    --tp 8 \
    --trust-remote-code
```

## Performance vs OpenAI o1

| Benchmark | o1-mini | o1 | R1 |
|-----------|---------|-----|-----|
| AIME 2024 | 63.6 | 79.2 | **79.8** |
| MATH-500 | 90.0 | 96.4 | **97.3** |
| Codeforces | 1820 | **2061** | 2029 |
| GPQA-Diamond | 60.0 | **75.7** | 71.5 |

## What's Not Released

1. **RL Training Code** - Alternatives: OpenRLHF, TRL, veRL
2. **Cold-start Data** - Not released
3. **Reward Model** - Not released

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Model overview and usage |
| Paper (arXiv:2501.12948) | Full methodology |
