# DualPipe: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/DualPipe.git  
> Commit: `030ce4325f4ebeb437da4ebc6d00a70469dd58ae` (2026-01-14)

## Overview

DualPipe is a bidirectional pipeline parallelism algorithm that achieves near-zero pipeline bubbles by running forward and backward passes in opposite directions simultaneously.

## The Pipeline Bubble Problem

### Standard 1F1B Schedule

```
PP=4, micro-batches=8

       │ Stage 0 │ Stage 1 │ Stage 2 │ Stage 3 │
───────┼─────────┼─────────┼─────────┼─────────┤
 t=0   │ F0      │ ░░░░░░░ │ ░░░░░░░ │ ░░░░░░░ │  ← Bubble
 t=1   │ F1      │ F0      │ ░░░░░░░ │ ░░░░░░░ │
 t=2   │ F2      │ F1      │ F0      │ ░░░░░░░ │
 t=3   │ F3      │ F2      │ F1      │ F0      │  ← Steady state
 t=4   │ B0      │ F3      │ F2      │ F1      │
 ...   │ ...     │ ...     │ ...     │ ...     │
 t=n   │ ░░░░░░░ │ ░░░░░░░ │ ░░░░░░░ │ B7      │  ← Bubble

Bubble ratio = (PP - 1) / num_microbatches
For PP=8, mb=20: 35% bubble!
```

### DualPipe Solution

Run forward and backward in **opposite directions**:

```
PP=4, micro-batches=8

       │ Stage 0 │ Stage 1 │ Stage 2 │ Stage 3 │
───────┼─────────┼─────────┼─────────┼─────────┤
 t=0   │ F0→     │         │         │     ←B0'│  (B0' from other stream)
 t=1   │ F1→     │ F0→     │     ←B0'│     ←B1'│
 t=2   │ F2→     │ F1→ B0'←│ F0→ B1'←│     ←B2'│
 t=3   │ F3→ B0'←│ F2→ B1'←│ F1→ B2'←│ F0→ B3'←│
       │ ...     │ ...     │ ...     │ ...     │

Two streams:
Stream A: F0→F1→F2→... (forward direction →)
Stream B: B0'←B1'←B2'←... (backward direction ←)
```

## Bubble Comparison

| Method | Bubble Formula | PP=8, mb=20 |
|--------|----------------|-------------|
| 1F1B | (PP-1)(F+B) | 14 slots |
| ZB1P | (PP-1)(F+B-2W) | 12 slots |
| **DualPipe** | (PP/2-1)(F&B+B-3W) | **~3 slots** |

Where:
- F = Forward time
- B = Backward time
- W = Weight update time
- F&B = Overlapped forward-backward

## How It Works

### Two-Stream Execution

```python
class DualPipeScheduler:
    def __init__(self, num_stages, num_microbatches):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        
    def schedule(self, rank):
        """
        Each rank processes two streams:
        - Forward stream: micro-batches 0, 1, 2, ...
        - Backward stream: micro-batches from opposite direction
        """
        for step in range(self.total_steps):
            # Forward from one direction
            if self.has_forward_work(rank, step):
                yield ('forward', self.get_forward_mb(rank, step))
            
            # Backward from opposite direction
            if self.has_backward_work(rank, step):
                yield ('backward', self.get_backward_mb(rank, step))
```

### Overlapping Forward and Backward

```
┌─────────────────────────────────────────────────────────────┐
│                  Single Stage Execution                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Time →                                                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │Forward mb0│  │Forward mb1│  │Forward mb2│                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                          │
│       ▼             ▼             ▼                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │Backward  │  │Backward  │  │Backward  │                   │
│  │(other mb)│  │(other mb)│  │(other mb)│                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
│                                                              │
│  Computation and communication overlap!                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Basic Usage

```python
from dualpipe import DualPipeScheduler

# Initialize scheduler
scheduler = DualPipeScheduler(
    num_stages=8,           # Pipeline parallel size
    num_microbatches=20,    # Micro-batches per step
)

# Get schedule for current rank
for op_type, mb_idx in scheduler.schedule(rank=my_rank):
    if op_type == 'forward':
        output = model.forward(inputs[mb_idx])
        send_activation(output, next_rank)
    elif op_type == 'backward':
        grad = recv_gradient(next_rank)
        model.backward(grad)
        send_gradient(grad_input, prev_rank)
```

### With Communication Overlap

```python
from dualpipe import DualPipeScheduler, overlap_comm_compute

scheduler = DualPipeScheduler(num_stages=8, num_microbatches=20)

# Overlap communication with computation
for ops in scheduler.schedule_with_overlap(rank):
    with overlap_comm_compute():
        for op in ops:
            execute(op)
```

## DualPipeV Variant

V-shaped schedule for even better overlap:

```
Standard DualPipe:
Stage 0: F→F→F→B←B←B←
Stage 1:  F→F→B←B←F→B←

DualPipeV:
Stage 0: F→F→B←F→B←B←
Stage 1:  F→B←F→B←F→B←
         (V-shaped interleaving)
```

Benefits:
- Better memory efficiency
- More uniform compute distribution
- Slightly lower peak memory

## Memory Considerations

### Activation Memory

```
DualPipe requires storing activations for:
- Forward stream: ~PP/2 micro-batches
- Backward stream: ~PP/2 micro-batches
- Total: ~PP micro-batches (2× standard 1F1B)

Trade-off: 2× activation memory for ~78% less bubble
```

### Gradient Accumulation

```python
# Gradient accumulation compatible
for micro_step in range(grad_accum_steps):
    for op_type, mb_idx in scheduler.schedule(rank):
        if op_type == 'forward':
            output = model.forward(inputs[mb_idx])
        elif op_type == 'backward':
            model.backward(grad, accumulate=True)
    
    # Sync and update after all micro-steps
    optimizer.step()
```

## Integration with DeepSeek

### V3 Training Configuration

```python
# DeepSeek-V3 training setup
config = {
    'pipeline_parallel': 8,
    'tensor_parallel': 8,
    'expert_parallel': 64,
    'micro_batches': 20,
    'scheduler': 'dualpipe',
}
```

### With DeepEP

```python
# Combine pipeline parallelism with expert parallelism
from dualpipe import DualPipeScheduler
from deep_ep import Buffer

scheduler = DualPipeScheduler(pp_size, num_microbatches)
ep_buffer = Buffer(ep_group, nvl_bytes, rdma_bytes)

for op_type, mb_idx in scheduler.schedule(rank):
    if op_type == 'forward':
        # MoE layer with expert parallelism
        hidden = attention(inputs[mb_idx])
        hidden = moe_forward_with_ep(hidden, ep_buffer)
        ...
```

## Debugging

### Visualization

```python
from dualpipe import visualize_schedule

# Generate schedule visualization
visualize_schedule(
    num_stages=8,
    num_microbatches=20,
    output_file='schedule.png'
)
```

### Profiling

```python
from dualpipe import profile_schedule

# Profile actual execution
stats = profile_schedule(
    scheduler,
    model,
    inputs,
    num_iterations=10
)

print(f"Bubble ratio: {stats['bubble_ratio']:.2%}")
print(f"Throughput: {stats['throughput']:.2f} samples/sec")
```

## Key Files

| File | Purpose |
|------|---------|
| `dualpipe/scheduler.py` | Main scheduler implementation |
| `dualpipe/v_schedule.py` | DualPipeV variant |
| `examples/` | Usage examples |
| `tests/` | Unit tests |
