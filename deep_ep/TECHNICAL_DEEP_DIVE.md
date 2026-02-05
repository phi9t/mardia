# DeepEP: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/DeepEP.git  
> Commit: `567632dd59810d77b3cc05553df953cc0f779799` (2026-02-03)

## Overview

DeepEP is a high-performance communication library for Mixture-of-Experts (MoE) expert parallelism, providing optimized all-to-all GPU kernels for MoE dispatch and combine operations.

## Core Operations

### MoE Communication Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    MoE Forward Pass                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Tokens ──► Router ──► Top-K Expert Selection              │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                      │
│                    │    DISPATCH     │  All-to-All          │
│                    │  (Tokens→Experts)│                      │
│                    └────────┬────────┘                      │
│                              │                               │
│                              ▼                               │
│                    Expert Computation                        │
│                    (FFN on each GPU)                        │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                      │
│                    │    COMBINE      │  All-to-All          │
│                    │ (Outputs→Original)│                     │
│                    └─────────────────┘                      │
│                              │                               │
│                              ▼                               │
│                    Weighted Sum Output                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Benchmarks

### Normal Kernels (Training/Prefill)

Tested on H800 (~160 GB/s NVLink, CX7 400 Gb/s RDMA):

| Type | EP Size | Dispatch BW | Combine BW |
|------|---------|-------------|------------|
| Intranode | 8 | 153 GB/s (NVLink) | 158 GB/s (NVLink) |
| Internode | 16 | 43 GB/s (RDMA) | 43 GB/s (RDMA) |
| Internode | 32 | 58 GB/s (RDMA) | 57 GB/s (RDMA) |
| Internode | 64 | 51 GB/s (RDMA) | 50 GB/s (RDMA) |

### Low-Latency Kernels (Decoding)

Pure RDMA for minimal latency:

| EP Size | Dispatch | Combine |
|---------|----------|---------|
| 8 | 77 μs | 114 μs |
| 16 | 118 μs | 195 μs |
| 32 | 155 μs | 273 μs |
| 64 | 173 μs | 314 μs |
| 128 | 192 μs | 369 μs |
| 256 | 194 μs | 360 μs |

## API Reference

### Buffer Initialization

```python
from deep_ep import Buffer, EventOverlap

# Normal mode (training/prefill)
buffer = Buffer(
    group=dist.ProcessGroup,     # Torch distributed group
    num_nvl_bytes=nvl_size,      # NVLink buffer size
    num_rdma_bytes=rdma_size     # RDMA buffer size
)

# Low-latency mode (decoding)
buffer = Buffer(
    group=group,
    num_nvl_bytes=0,
    num_rdma_bytes=rdma_size,
    low_latency_mode=True,
    num_qps_per_rank=num_experts // group.size()
)

# Control SM usage (static)
Buffer.set_num_sms(24)
```

### Dispatch Operation

```python
def dispatch_forward(x, topk_idx, topk_weights, num_experts, previous_event=None):
    """
    Send tokens to their selected experts.
    
    Args:
        x: Input tensor [num_tokens, hidden_dim]
        topk_idx: Expert indices [num_tokens, top_k]
        topk_weights: Expert weights [num_tokens, top_k]
        num_experts: Total number of experts
        previous_event: Optional CUDA event for dependency
    
    Returns:
        recv_x: Received tokens for local experts
        recv_topk_idx: Expert indices for received tokens
        recv_topk_weights: Weights for received tokens
        num_recv_tokens_per_expert: Count per expert
        handle: State for combine operation
        event: CUDA event for synchronization
    """
    # Calculate layout
    layout = buffer.get_dispatch_layout(
        topk_idx, num_experts,
        previous_event=previous_event,
        async_finish=True
    )
    
    # Execute dispatch
    return buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_tokens_per_rank=layout.num_tokens_per_rank,
        num_tokens_per_rdma_rank=layout.num_tokens_per_rdma_rank,
        is_token_in_rank=layout.is_token_in_rank,
        num_tokens_per_expert=layout.num_tokens_per_expert,
        previous_event=previous_event,
        async_finish=True
    )
```

### Combine Operation

```python
def combine_forward(x, handle, previous_event=None):
    """
    Gather expert outputs back to original positions.
    
    Args:
        x: Expert outputs [num_recv_tokens, hidden_dim]
        handle: State from dispatch
        previous_event: Optional CUDA event
    
    Returns:
        combined_x: Combined output [num_tokens, hidden_dim]
        event: CUDA event
    """
    return buffer.combine(
        x, handle,
        async_finish=True,
        previous_event=previous_event
    )
```

### Low-Latency Mode

```python
def low_latency_dispatch(hidden_states, topk_idx, num_max_tokens, num_experts):
    """
    Dispatch for inference decoding with minimal latency.
    Compatible with CUDA graph.
    """
    recv_hidden, recv_count, handle, event, hook = buffer.low_latency_dispatch(
        hidden_states,
        topk_idx,
        num_max_tokens,
        num_experts,
        async_finish=False,
        return_recv_hook=True  # Hook-based overlap
    )
    
    # hook() triggers actual receive - useful for overlap
    return recv_hidden, recv_count, handle, event, hook

def low_latency_combine(hidden_states, topk_idx, topk_weights, handle):
    """Combine for inference decoding."""
    return buffer.low_latency_combine(
        hidden_states,
        topk_idx,
        topk_weights,
        handle,
        async_finish=False,
        return_recv_hook=True
    )
```

## Hook-Based Communication Overlap

Zero SM usage overlap pattern:

```
┌─────────────────────────────────────────────────────────────┐
│              Two-Micro-Batch Overlap                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Batch 1:  [Attention] ─► [Dispatch] ─► [MoE] ─► [Combine]  │
│                              │                     │         │
│  Batch 2:        [Attention] ─► [Dispatch] ─► [MoE] ─►      │
│                                    │                         │
│                              RDMA happens in background      │
│                              (no SM usage)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
# Start dispatch, get hook
recv_x, ..., hook = buffer.low_latency_dispatch(
    ..., return_recv_hook=True
)

# Do other compute (attention for next batch)
other_output = attention(other_batch)

# Now trigger receive completion
hook()

# Continue with MoE computation
```

## Network Configuration

### Traffic Isolation (Virtual Lanes)

```bash
# Separate traffic types
export NVSHMEM_IB_SL=0  # Normal kernels
export NVSHMEM_IB_SL=1  # Low-latency kernels
export NVSHMEM_IB_SL=2  # Other workloads
```

### Adaptive Routing

- **Heavy loads**: Enable adaptive routing (distribute across paths)
- **Light loads**: Use static routing (lower latency)

## Build & Installation

### Requirements

- CUDA 12.3+ (SM90) or CUDA 11.0+ (SM80)
- PyTorch 2.1+
- NVLink (intranode) + RDMA (internode)
- NVSHMEM

### Installation

```bash
# Install NVSHMEM first (see third-party/README.md)

# Build
NVSHMEM_DIR=/path/to/nvshmem python setup.py build

# Create symlink
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.*.so .

# Install
NVSHMEM_DIR=/path/to/nvshmem python setup.py install
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NVSHMEM_DIR` | Path to NVSHMEM installation |
| `DISABLE_SM90_FEATURES` | Disable Hopper features |
| `TORCH_CUDA_ARCH_LIST` | Target architectures |
| `DISABLE_AGGRESSIVE_PTX_INSTRS` | Disable undefined-behavior PTX |

## Implementation Notes

### Undefined-Behavior PTX Usage

For extreme performance, DeepEP uses:
```
ld.global.nc.L1::no_allocate.L2::256B
```

- `.nc` = non-coherent cache (normally for read-only)
- Used for volatile data (technically undefined)
- Works on Hopper due to unified L1/NC cache
- Disable with `DISABLE_AGGRESSIVE_PTX_INSTRS=1` if issues occur

### Buffer Design

Current implementation uses queues (memory efficient but complex). Simpler alternative:
- Fixed-size buffers at maximum capacity
- Better performance, easier to reason about
- See GitHub issue #39 for discussion

## Experimental Features

| Branch | Feature |
|--------|---------|
| `zero-copy` | Remove PyTorch↔buffer copies |
| `eager` | Low-latency protocol without RDMA atomics |
| `hybrid-ep` | TMA instructions, PCIe support, NVFP4 |
| `mori-ep` | AMD ROCm support |

## Key Files

| File | Purpose |
|------|---------|
| `deep_ep/buffer.py` | Main Buffer class |
| `csrc/kernels/` | CUDA kernel implementations |
| `tests/test_*.py` | Unit tests |
| `third-party/README.md` | NVSHMEM setup guide |
