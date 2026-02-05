# DeepSeekMath-V2: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-Math-V2.git  
> Commit: `665c840782baf7faae8a8b244ea313f3cfcc346f` (2025-12-01)

## Overview

DeepSeekMath-V2 introduces **self-verifiable mathematical reasoning** - using an LLM-based verifier as a reward model to ensure proof correctness, not just final answer accuracy.

## Key Results

| Competition | Score |
|-------------|-------|
| IMO 2025 | Gold-level |
| CMO 2024 | Gold-level |
| Putnam 2024 | 118/120 |

## Core Innovation

### The Problem

```
Traditional Reward: correct_answer → +1, wrong_answer → 0

Issue: Correct answer ≠ correct reasoning
       Model may get right answer with flawed proof
```

### The Solution

```
Self-Verifiable Reward:
1. Generate proof with self-evaluation
2. LLM verifier scores proof rigor (0-1)
3. Use verification score as reward
4. Scale verification compute for hard problems
```

## Architecture

**Base Model**: DeepSeek-V3.2-Exp-Base (with DSA attention)

### Multi-Round Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Round R                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐                                       │
│  │ Proof Generation │ Temperature: 1.0                      │
│  │                  │ Max tokens: 128K                      │
│  │                  │ Output: proofs with self-eval         │
│  └────────┬─────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Proof Verification│ Multiple verifications (n=4)         │
│  │                   │ Scores each proof 0-1                │
│  └────────┬──────────┘                                      │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Meta Verification │ (Optional)                           │
│  │                   │ QC on low-scoring verifications      │
│  └────────┬──────────┘                                      │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Proof Refinement  │ Select best (top n_best)            │
│  │                   │ Generate improved proofs             │
│  └──────────────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│       Next Round                                             │
└─────────────────────────────────────────────────────────────┘
```

## Self-Evaluation in Proofs

Generator outputs include embedded self-evaluation:

```xml
<think>
[Extended reasoning and proof development]
</think>

[Proof content with mathematical rigor]

<self_eval>
[Model evaluates own proof]
- Checks logical consistency
- Verifies all steps
- Identifies potential gaps
Score: \boxed{0.85}
</self_eval>
```

### Extraction

```python
def process_proof(proof_text):
    self_eval = extract_self_eval(proof_text)
    proof_content = extract_solution(proof_text)
    score = float(extract_boxed_answers(self_eval)[-1])
    return proof_content, score
```

## Inference Code

### Directory Structure

```
inference/
├── main.py           # Multi-round orchestration
├── generate.py       # API calls for generation
├── math_templates.py # Prompts for gen/verify/refine
├── utils.py          # Helper functions
└── run.sh           # Example launch script
```

### Key Parameters

```python
parser.add_argument("--proof_gen_temp", default=1.0)
parser.add_argument("--proof_gen_max_len", default=128*1024)
parser.add_argument("--n_best_proofs_to_sample", default=32)
parser.add_argument("--n_verification_per_proof", default=4)
parser.add_argument("--max_rounds", default=20)
```

### Running Inference

```bash
python inference/main.py \
    --input_path inputs/IMO2025.json \
    --output_path outputs/ \
    --max_rounds 10 \
    --n_best_proofs_to_sample 32 \
    --n_verification_per_proof 4
```

## Input/Output Format

### Input (IMO2025.json)

```json
{
    "id": 1,
    "question": "A line in the plane is called sunny if...",
    "answer": "k = 0, 1, 3 for all n",
    "contest": "IMO2025",
    "problem_idx": "IMO2025-1"
}
```

### Output

JSONL format with proofs and verification scores:

```json
{
    "problem_idx": "IMO2025-1",
    "round": 5,
    "proof": "...",
    "self_eval_score": 0.92,
    "verification_scores": [0.95, 0.90, 0.88, 0.91],
    "avg_verification": 0.91
}
```

## Scaling Test-Time Compute

Key insight: **Scale verification compute for hard problems**

```python
def solve_with_scaled_compute(problem, difficulty):
    if difficulty == "easy":
        n_candidates = 8
        n_verifications = 2
        max_rounds = 3
    elif difficulty == "medium":
        n_candidates = 16
        n_verifications = 4
        max_rounds = 10
    else:  # hard (IMO-level)
        n_candidates = 32
        n_verifications = 4
        max_rounds = 20
    
    return multi_round_solve(
        problem,
        n_candidates=n_candidates,
        n_verifications=n_verifications,
        max_rounds=max_rounds
    )
```

## Connection to Infrastructure

- **Base Model**: V3.2-Exp with DSA for efficient long-context
- **Inference**: Long proofs (128K tokens) require optimized attention
- **Compute**: Multi-round verification scales with problem difficulty

## Key Files

| File | Purpose |
|------|---------|
| `inference/main.py` | Main orchestration script |
| `inference/math_templates.py` | Generation/verification prompts |
| `inputs/IMO2025.json` | Example IMO problems |
| `README.md` | Usage guide |
