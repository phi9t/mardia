# Engram: Deep Dive

> Vendored from: https://github.com/deepseek-ai/Engram.git  
> Commit: `fb7f84a21f91223715394a33a1dc24bbfb7f788e` (2026-01-14)

## Overview

Engram introduces **conditional memory** - an O(1) N-gram embedding lookup mechanism that complements MoE's conditional computation, discovering a U-shaped scaling law between neural compute and static memory.

## Core Innovation

### The Insight

MoE provides **conditional computation** (different experts for different tokens), but Transformers lack **conditional memory** (static pattern lookup).

```
Traditional Transformer:
    Token → Embedding → [Attention + FFN]×L → Output
                              ↑
                        All computation is neural

Engram Transformer:
    Token → N-gram Hash → Embedding Lookup (O(1)) ──┐
                                                     ▼
    Token → Embedding → [Attention + MoE]×L ──► Fuse → Output
                              ↑
                   Neural computation + Static memory
```

### Why This Matters

Some patterns are **deterministic** (common phrases, syntax):
- "New York" → always refers to the city
- "def __init__" → always a Python constructor
- Language-specific idioms

Neural networks waste capacity learning these static patterns. Engram offloads them to a hash table.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Engram Layer                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Input Tokens ──┬──► N-gram Hash ──► Embedding Table       │
│                  │         │                │                │
│                  │         │    O(1) lookup │                │
│                  │         │                ▼                │
│                  │         │         Memory Output           │
│                  │         │                │                │
│                  │         │                │                │
│                  └──► Neural Path ──────────┤                │
│                       (Attention/MoE)       │                │
│                             │               │                │
│                             ▼               ▼                │
│                         Fuse (concat/add/gate)               │
│                                   │                          │
│                                   ▼                          │
│                              Output                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### N-gram Hashing

```python
def ngram_hash(tokens, n=3):
    """Hash n consecutive tokens to embedding index."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        hash_val = hash(ngram) % table_size
        ngrams.append(hash_val)
    return ngrams

# Lookup is O(1) - just array indexing
def lookup(indices, embedding_table):
    return embedding_table[indices]
```

## U-Shaped Scaling Law

Key discovery: There's an **optimal mix** of neural compute and static memory.

```
Loss
  │
  │    *           *
  │   * *         * *
  │  *   *       *   *
  │ *     *     *     *
  │*       *****       *
  └──────────────────────→
   100% Neural    100% Memory
   
Optimal: Mix of both!
```

### Implications

1. **Too much neural**: Wastes compute on learnable patterns
2. **Too much memory**: Loses generalization
3. **Optimal mix**: Best of both worlds

## Engram-27B Results

Under iso-parameter and iso-FLOPs constraints:

| Domain | Improvement |
|--------|-------------|
| Knowledge | Consistent gains |
| Reasoning | Preserved |
| Code | Improved |
| Math | Improved |

### Why It Works

- **Early layers**: Relieved from static pattern reconstruction
- **Later layers**: Can focus on complex reasoning
- **Effective depth**: Preserved for hard tasks

## System Efficiency

### Deterministic Addressing

Unlike MoE routing (needs softmax), Engram lookup is:
- **Deterministic**: Hash function, no learned routing
- **O(1)**: Single array access
- **Predictable**: No load balancing issues

### Host Memory Offloading

```python
# Massive embedding tables can live in CPU memory
class EngramModule:
    def __init__(self, table_size, embed_dim):
        # Table in CPU memory (can be huge)
        self.table = torch.zeros(table_size, embed_dim, device="cpu")
    
    def forward(self, indices):
        # Only fetch needed embeddings to GPU
        # Access pattern is known ahead of time
        embeddings = self.table[indices].to("cuda")
        return embeddings
```

**Benefits**:
- Table can be 100s of GB (CPU RAM is cheap)
- Minimal GPU memory footprint
- Predictable memory access patterns

## Demo Code

```bash
# Run the demonstration
pip install torch numpy transformers sympy
python engram_demo_v1.py
```

**Note**: Demo mocks standard components (Attention/MoE/mHC) to focus on Engram module logic.

### Demo Structure

```python
# engram_demo_v1.py (simplified)
class EngramDemo:
    def __init__(self):
        self.ngram_table = {}  # N-gram → embedding
        self.neural_model = MockTransformer()
    
    def forward(self, tokens):
        # Memory path
        ngrams = extract_ngrams(tokens, n=3)
        memory_out = self.lookup(ngrams)
        
        # Neural path
        neural_out = self.neural_model(tokens)
        
        # Fuse
        return self.fuse(memory_out, neural_out)
```

## Connection to DeepSeek Models

### Complementary to MoE

```
MoE:    Conditional Computation (which expert?)
Engram: Conditional Memory (which pattern?)

Combined: Both sparse computation AND sparse memory
```

### Potential Integration

- V3 + Engram: More efficient for common patterns
- Coder + Engram: Programming idiom lookup
- R1 + Engram: Reasoning pattern shortcuts

## Research Directions

1. **Scale the table**: How large can it get?
2. **Dynamic updates**: Online learning of new patterns
3. **Multi-level**: Different N for different layers
4. **Sparse tables**: Hash collision handling

## Key Files

| File | Purpose |
|------|---------|
| `engram_demo_v1.py` | Demo implementation |
| `README.md` | Overview and results |
| Paper (arXiv:2601.07372) | Full technical details |
