# DeepSeek-Coder-V2: Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepSeek-Coder-V2.git  
> Commit: `a2b4e0a25b5dab1ee87e8080f76e4512b0725b7b` (2025-11-11)

## Overview

DeepSeek-Coder-V2 is an MoE code model achieving GPT-4 Turbo parity on coding benchmarks, supporting 338 programming languages with 128K context.

## Model Variants

| Variant | Total Params | Activated | Context |
|---------|-------------|-----------|---------|
| Coder-V2-Lite | 16B | 2.4B | 128K |
| Coder-V2 | 236B | 21B | 128K |

## Key Improvements Over V1

| Feature | Coder (V1) | Coder-V2 |
|---------|------------|----------|
| Languages | 86 | **338** |
| Context | 16K | **128K** |
| Architecture | Dense | **MoE** |
| Attention | Standard | **MLA** |
| Training | 2T tokens | V2 + **6T** tokens |

## Architecture

Based on DeepSeek-V2 (MLA + DeepSeekMoE):

```
DeepSeek-V2 Checkpoint
        │
        ▼
Continue Pre-training
        │ + 6T code/math tokens
        │ + 338 languages
        ▼
DeepSeek-Coder-V2
```

### MLA for Code

Multi-head Latent Attention provides:
- **93% KV cache reduction** → Longer context for repo-level code
- **Efficient long-range attention** → Better cross-file understanding

### MoE for Code

DeepSeekMoE architecture:
- **Fine-grained experts** → Language-specific specialization
- **Shared experts** → Common programming patterns
- **Efficient routing** → Only 2.4B/21B params active per token

## Capabilities

### 1. 338 Programming Languages

Expanded from 86 to include:
- Esoteric: Brainfuck, Whitespace, LOLCODE
- Domain-specific: VHDL, Verilog, MATLAB, R
- Legacy: COBOL, Fortran, Pascal
- Modern: Zig, Nim, Crystal
- Configuration: Terraform, Ansible, Kubernetes YAML

### 2. 128K Context

Enables true repository-level understanding:

```python
# Can process entire codebases
context = ""
for file in repo.files[:100]:  # Up to ~100 files
    context += f"# {file.path}\n{file.content}\n\n"

# Ask questions about the entire repo
response = model.chat(
    context + "\nWhat design patterns are used in this codebase?"
)
```

### 3. Code Fixing (SWE-Bench)

```python
# Given a bug report and codebase
prompt = f"""
Repository: {repo_context}

Issue: {bug_report}

Generate a patch to fix this issue.
"""

patch = model.generate(prompt)
```

## Benchmark Results

| Benchmark | Coder-V2-Lite | Coder-V2 | GPT-4 Turbo |
|-----------|---------------|----------|-------------|
| HumanEval | 90.2% | 90.2% | 90.2% |
| MBPP | 78.3% | 80.1% | 80.1% |
| LiveCodeBench | 43.4% | - | 43.8% |
| Aider Polyglot | 7.5% | 9.3% | 9.1% |

## Usage

### Basic Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).cuda()
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
)

messages = [
    {"role": "user", "content": "Implement a B+ tree in Rust"}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt"
).to("cuda")
outputs = model.generate(inputs, max_new_tokens=2048)
```

### FIM (Fill-in-Middle)

```python
fim_prompt = """<｜fim▁begin｜>impl<K: Ord, V> BTreeMap<K, V> {
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
<｜fim▁hole｜>
    }
}<｜fim▁end｜>"""
```

## Deployment

### SGLang (Recommended for Full Model)

```bash
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-Coder-V2-Instruct \
    --tp 8
```

### vLLM (Lite Model)

```bash
vllm serve deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768
```

## Agentic Coding Patterns

### Tool Use

```python
tools = [
    {"name": "read_file", "description": "Read a file from the repo"},
    {"name": "write_file", "description": "Write content to a file"},
    {"name": "run_tests", "description": "Run the test suite"},
    {"name": "search_code", "description": "Search for code patterns"},
]

messages = [
    {"role": "system", "content": f"You have access to: {tools}"},
    {"role": "user", "content": "Fix the failing tests in auth.py"}
]
```

### Multi-Turn Code Editing

```python
conversation = [
    {"role": "user", "content": "Add input validation to this function"},
    {"role": "assistant", "content": "```python\ndef process(data):\n    if not data:\n        raise ValueError('Empty data')\n    ...```"},
    {"role": "user", "content": "Also add type hints"},
    {"role": "assistant", "content": "```python\ndef process(data: dict) -> dict:\n    if not data:\n        raise ValueError('Empty data')\n    ...```"},
]
```

## Connection to Infrastructure

- **MLA** (`flash_mla/`): Enables 128K context efficiently
- **MoE** (`deep_ep/`): Expert parallelism for distributed inference
- **Inference**: Leverages DeepSeek-V2 optimizations

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Model documentation |
| Paper (arXiv) | Technical details |
