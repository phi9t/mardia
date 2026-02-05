# 3FS (Fire-Flyer File System): Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/3FS.git  
> Commit: `e7ef789d94f9b427ab55847f9aec51a19903363b` (2026-02-04)

## Overview

3FS is a high-performance distributed file system designed for AI training and inference workloads, leveraging modern SSDs and RDMA networks to provide shared storage with strong consistency.

## Performance

### Peak Throughput

| Metric | Value |
|--------|-------|
| Aggregate Read | 6.6 TiB/s |
| Cluster Size | 180 storage nodes |
| NICs per Node | 2× 200Gbps InfiniBand |
| SSDs per Node | 16× 14TiB NVMe |

### GraySort Benchmark

| Metric | Value |
|--------|-------|
| Data Size | 110.5 TiB |
| Partitions | 8,192 |
| Time | 30 min 14 sec |
| Throughput | **3.66 TiB/min** |

### KVCache Performance

| Metric | Value |
|--------|-------|
| Peak Read | 40 GiB/s per client |
| NIC Config | 1× 400Gbps per node |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      3FS Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Clients                                                    │
│   ┌─────┐ ┌─────┐ ┌─────┐                                   │
│   │GPU 0│ │GPU 1│ │GPU n│  (Training/Inference nodes)       │
│   └──┬──┘ └──┬──┘ └──┬──┘                                   │
│      │       │       │                                       │
│      └───────┼───────┘                                       │
│              │ RDMA                                          │
│              ▼                                               │
│   ┌─────────────────────────────────────────────┐           │
│   │            Metadata Service                  │           │
│   │         (Stateless, FoundationDB)           │           │
│   └─────────────────────────────────────────────┘           │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────┐           │
│   │           Storage Nodes                      │           │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │           │
│   │  │SSD 0│ │SSD 1│ │SSD 2│ │SSD n│           │           │
│   │  └─────┘ └─────┘ └─────┘ └─────┘           │           │
│   │        (Chain Replication - CRAQ)           │           │
│   └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Disaggregated Architecture

```
Traditional:  Compute ←──→ Local SSD
              (Locality required)

3FS:          Compute ←──RDMA──→ Distributed SSDs
              (Locality oblivious)
```

Benefits:
- Access any SSD from any compute node
- Combined bandwidth of all SSDs
- No data locality constraints

### 2. Strong Consistency (CRAQ)

Chain Replication with Apportioned Queries:

```
Write Path:
Client → Head → Node 2 → ... → Tail → ACK
         (sequential replication)

Read Path (Clean):
Client → Any Node → Response
         (parallel reads)

Read Path (Dirty):
Client → Tail → Response
         (always consistent)
```

Benefits:
- Strong consistency guarantees
- High read throughput (any replica)
- Simple application code

### 3. POSIX-like File Interface

```c
// Standard file operations
int fd = open("/3fs/data/train.bin", O_RDONLY);
read(fd, buffer, size);
close(fd);

// No new API to learn!
```

## AI Workload Support

### Data Preparation

```python
# Organize data pipelines
/3fs/datasets/
├── raw/
│   └── crawl_20240101/
├── processed/
│   └── tokenized/
└── final/
    └── train_shards/
```

### Dataloaders

```python
# Random access without prefetching
class ThreeFSDataset(Dataset):
    def __init__(self, path):
        self.files = glob(f"{path}/*.bin")
    
    def __getitem__(self, idx):
        # Direct random access - no prefetch needed!
        with open(self.files[idx], 'rb') as f:
            return torch.load(f)
```

### Checkpointing

```python
# High-throughput parallel checkpoint
def save_checkpoint(model, path):
    # Each rank saves its shard
    shard_path = f"{path}/shard_{rank}.pt"
    torch.save(model.state_dict(), shard_path)
    
    # 3FS handles parallel writes efficiently
    # No need for manual coordination
```

### KVCache for Inference

```python
# Offload KV cache to 3FS instead of DRAM
class ThreeFSKVCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir  # 3FS path
    
    def store(self, key, kv_tensor):
        path = f"{self.cache_dir}/{key}.bin"
        with open(path, 'wb') as f:
            kv_tensor.numpy().tofile(f)
    
    def load(self, key):
        path = f"{self.cache_dir}/{key}.bin"
        with open(path, 'rb') as f:
            return torch.from_numpy(np.fromfile(f))
```

Benefits over DRAM:
- Much larger capacity (TBs vs GBs)
- Cost effective
- Persistent across restarts

## USRBIO API

High-performance user-space I/O:

```c
#include "UsrbIo.h"

// Initialize
UsrbIoContext ctx;
usrbio_init(&ctx, config);

// Async read
UsrbIoRequest req;
req.fd = fd;
req.offset = offset;
req.size = size;
req.buffer = buffer;

usrbio_submit(&ctx, &req);
usrbio_wait(&ctx, &req);
```

### FIO Engine

```bash
# Benchmark with custom fio engine
fio --ioengine=usrbio \
    --filename=/3fs/testfile \
    --rw=randread \
    --bs=4k \
    --numjobs=16 \
    --runtime=60
```

## Build & Installation

### Dependencies

```bash
# Ubuntu 22.04
apt install cmake libuv1-dev liblz4-dev liblzma-dev \
    libdouble-conversion-dev libdwarf-dev libunwind-dev \
    libaio-dev libgflags-dev libgoogle-glog-dev \
    clang-14 lld-14 libboost-all-dev

# Additional requirements
# - libfuse 3.16.1+
# - FoundationDB 7.1+
# - Rust 1.75+
```

### Build

```bash
git clone https://github.com/deepseek-ai/3fs
cd 3fs
git submodule update --init --recursive
./patches/apply.sh

cmake -S . -B build \
    -DCMAKE_CXX_COMPILER=clang++-14 \
    -DCMAKE_C_COMPILER=clang-14 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSHUFFLE_METHOD=g++11

cmake --build build -j 32
```

### Cluster Setup

See `deploy/README.md` for:
- Storage node configuration
- Metadata service setup
- Client mounting
- Network configuration

## Integration with smallpond

3FS pairs with smallpond for data processing:

```python
import smallpond

# Process data on 3FS
df = smallpond.read_parquet("/3fs/data/*.parquet")
result = df.filter(...).groupby(...).agg(...)
result.write_parquet("/3fs/output/")
```

## Key Files

| File | Purpose |
|------|---------|
| `src/lib/api/UsrbIo.md` | USRBIO API reference |
| `docs/design_notes.md` | Architecture design |
| `deploy/README.md` | Cluster setup guide |
| `benchmarks/fio_usrbio/` | Benchmark tools |
| `specs/README.md` | P specifications |
