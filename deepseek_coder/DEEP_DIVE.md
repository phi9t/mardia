# DeepSeek-Coder: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-Coder.git  
> Commit: `2f9fd85927c669dae3c0fbb2d607274023af243e` (2025-11-11)

## Overview

DeepSeek-Coder is a code-focused LLM trained from scratch on 2T tokens (87% code, 13% natural language), supporting 86 programming languages with 16K context length.

## Model Variants

| Model | Parameters | Context | Use Case |
|-------|------------|---------|----------|
| deepseek-coder-1.3b-base | 1.3B | 16K | Edge/embedded |
| deepseek-coder-5.7b-mqa-base | 5.7B | 16K | Efficient inference |
| deepseek-coder-6.7b-base | 6.7B | 16K | Balanced |
| deepseek-coder-33b-base | 33B | 16K | Best quality |
| *-instruct variants | Same | 16K | Chat/instruction |

## Key Features

### 1. 86 Programming Languages

Full language support including:
- Systems: C, C++, Rust, Go, Zig
- Web: JavaScript, TypeScript, HTML, CSS
- Data Science: Python, R, Julia
- Enterprise: Java, C#, Kotlin, Scala
- Scripting: Bash, PowerShell, Lua
- And 70+ more...

### 2. Fill-in-the-Middle (FIM) Training

Enables code insertion, not just completion:

```python
# FIM Special Tokens
<｜fim▁begin｜>  # Start of prefix context
<｜fim▁hole｜>   # Insertion point
<｜fim▁end｜>    # End of suffix context

# Example: Complete a function body
prompt = '''<｜fim▁begin｜>def calculate_tax(income, rate):
<｜fim▁hole｜>
    return tax<｜fim▁end｜>'''

# Model generates:
#     tax = income * rate
```

### 3. Repository-Level Completion

Trained on dependency-ordered repos for cross-file context:

```python
# File: utils/math.py
def add(a, b):
    return a + b

# File: main.py (model sees utils/math.py first)
from utils.math import add

result = add(  # Model understands add() signature
```

## Data Pipeline

```
GitHub Crawl
    │
    ▼
StarCoder Filter Rules
    │ - Remove low-quality
    │ - Filter by stars/forks
    ▼
Parse Dependencies
    │ - Extract imports
    │ - Build dependency graph
    ▼
Rearrange by Dependencies
    │ - Sort files topologically
    │ - Dependencies before dependents
    ▼
Repository-Level Dedup
    │ - Near-duplicate removal
    │ - Cross-repo dedup
    ▼
Quality Filter
    │ - Syntax validation
    │ - Style checks
    ▼
2T Training Tokens
    (87% code, 13% NL)
```

## Usage

### Basic Completion

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-base"
)

prompt = "def quicksort(arr):"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### FIM Completion

```python
fim_prompt = """<｜fim▁begin｜>def binary_search(arr, target):
    left, right = 0, len(arr) - 1
<｜fim▁hole｜>
    return -1<｜fim▁end｜>"""

inputs = tokenizer(fim_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

### Instruction Following

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="deepseek-ai/deepseek-coder-6.7b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Write a Python function to find the longest palindrome substring"}
]
response = pipe(messages, max_new_tokens=512)
```

## Benchmark Results

| Benchmark | 6.7B-Base | 6.7B-Instruct | 33B-Instruct |
|-----------|-----------|---------------|--------------|
| HumanEval | 47.6 | 73.8 | 79.3 |
| MBPP | 57.4 | 65.4 | 70.0 |
| DS-1000 | 20.4 | 39.2 | 43.3 |

## Fine-Tuning

### LoRA Fine-Tuning

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### Data Format

```json
{
    "instruction": "Write a function to check if a string is a palindrome",
    "input": "",
    "output": "def is_palindrome(s):\n    return s == s[::-1]"
}
```

## Deployment

### vLLM

```bash
vllm serve deepseek-ai/deepseek-coder-6.7b-instruct \
    --tensor-parallel-size 1 \
    --max-model-len 16384
```

### Text Generation Inference

```bash
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id deepseek-ai/deepseek-coder-6.7b-instruct
```

## Evolution to Coder-V2

DeepSeek-Coder laid the foundation for Coder-V2:

| Feature | Coder | Coder-V2 |
|---------|-------|----------|
| Languages | 86 | 338 |
| Context | 16K | 128K |
| Architecture | Dense | MoE (MLA + DeepSeekMoE) |
| Training | 2T tokens | V2 checkpoint + 6T tokens |

## Key Files

| File | Purpose |
|------|---------|
| `Evaluation/` | Benchmark evaluation scripts |
| `finetune/` | Fine-tuning code and configs |
| `demo/` | Usage examples |
| `README.md` | Model documentation |
