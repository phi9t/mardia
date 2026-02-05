# smallpond: Technical Deep Dive

> Vendored from: https://github.com/deepseek-ai/smallpond.git  
> Commit: `52ecc5e45535c7448f848bcb45b0da00d9484f81` (2025-03-05)

## Overview

smallpond is a lightweight data processing framework optimized for 3FS, providing DuckDB-based analytics with distributed execution for AI data pipelines.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      smallpond                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   User Code (Python)                                         │
│   ┌─────────────────────────────────────────────┐           │
│   │ df = sp.read_parquet("/3fs/data/*.parquet") │           │
│   │ df.filter(...).groupby(...).write(...)      │           │
│   └─────────────────────────────────────────────┘           │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────┐           │
│   │         Query Planner                        │           │
│   │    (Logical → Physical Plan)                │           │
│   └─────────────────────────────────────────────┘           │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────┐           │
│   │       Distributed Executor                   │           │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           │           │
│   │  │DuckDB│ │DuckDB│ │DuckDB│ │DuckDB│         │           │
│   │  │Worker│ │Worker│ │Worker│ │Worker│         │           │
│   │  └─────┘ └─────┘ └─────┘ └─────┘           │           │
│   └─────────────────────────────────────────────┘           │
│              │                                               │
│              ▼                                               │
│   ┌─────────────────────────────────────────────┐           │
│   │              3FS Storage                     │           │
│   └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. DuckDB Backend

```python
import smallpond as sp

# SQL-like operations powered by DuckDB
df = sp.read_parquet("/3fs/data/")

result = df.sql("""
    SELECT category, COUNT(*) as cnt, AVG(value) as avg_val
    FROM df
    WHERE timestamp > '2024-01-01'
    GROUP BY category
    HAVING cnt > 100
    ORDER BY avg_val DESC
""")
```

### 2. Lazy Evaluation

```python
# Operations are lazy - nothing executes yet
df = sp.read_parquet("/3fs/data/")
filtered = df.filter(df['quality'] > 0.8)
grouped = filtered.groupby('category').agg({'value': 'sum'})

# Execution happens on collect/write
result = grouped.collect()  # Now it runs!
```

### 3. Distributed Execution

```python
# Automatically parallelizes across workers
sp.configure(
    num_workers=50,
    worker_memory="64GB",
    storage_path="/3fs/scratch/"
)

# Large-scale processing
df = sp.read_parquet("/3fs/100TB_dataset/")
result = df.process(...)  # Runs on 50 workers
```

## API Reference

### Reading Data

```python
# Parquet files
df = sp.read_parquet("/3fs/data/*.parquet")
df = sp.read_parquet(["/path1/", "/path2/"])

# CSV files
df = sp.read_csv("/3fs/data/*.csv", header=True)

# JSON files
df = sp.read_json("/3fs/data/*.jsonl")
```

### Transformations

```python
# Filter
df = df.filter(df['score'] > 0.5)
df = df.filter("score > 0.5 AND category = 'A'")

# Select
df = df.select(['col1', 'col2', 'col3'])
df = df.select(df['col1'], df['col2'].alias('new_name'))

# Add columns
df = df.with_column('new_col', df['a'] + df['b'])

# Group and aggregate
df = df.groupby('category').agg({
    'value': ['sum', 'mean', 'count'],
    'score': 'max'
})

# Join
df = df1.join(df2, on='key', how='inner')

# Sort
df = df.sort('timestamp', descending=True)

# Distinct
df = df.distinct()

# Limit
df = df.limit(1000)
```

### Writing Data

```python
# Parquet
df.write_parquet("/3fs/output/", partition_by=['date'])

# CSV
df.write_csv("/3fs/output/")

# Single file
df.write_parquet("/3fs/output.parquet", single_file=True)
```

### SQL Interface

```python
# Register table
sp.register("my_table", df)

# Run SQL
result = sp.sql("""
    SELECT * FROM my_table
    WHERE condition
""")

# Complex queries
result = sp.sql("""
    WITH ranked AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC) as rn
        FROM my_table
    )
    SELECT * FROM ranked WHERE rn <= 10
""")
```

## GraySort Implementation

smallpond achieves 3.66 TiB/min on GraySort:

### Phase 1: Partition by Key Prefix

```python
def graysort_partition(input_path, output_path, num_partitions):
    df = sp.read_binary(input_path)
    
    # Extract key prefix (first few bytes)
    df = df.with_column('partition', 
        sp.func.hash(df['key'][:4]) % num_partitions
    )
    
    # Shuffle to partitions
    df.write_partitioned(output_path, partition_by='partition')
```

### Phase 2: In-Partition Sort

```python
def graysort_sort(partition_path, output_path):
    df = sp.read_binary(partition_path)
    
    # Sort within partition
    df = df.sort('key')
    
    df.write_binary(output_path)
```

### Configuration

```python
# GraySort cluster config
sp.configure(
    num_workers=50,              # 50 compute nodes
    worker_memory="2.2TB",       # Per-node RAM
    worker_cores=192,            # Per-node cores
    storage_path="/3fs/scratch/",
    network_bandwidth="200Gbps"
)
```

## AI Data Pipeline Examples

### Tokenization Pipeline

```python
import smallpond as sp
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-v3")

def tokenize_batch(texts):
    return tokenizer(texts, truncation=True, max_length=4096)

# Process large text corpus
df = sp.read_parquet("/3fs/raw_text/")
df = df.map_batches(tokenize_batch, batch_size=1000)
df.write_parquet("/3fs/tokenized/")
```

### Deduplication Pipeline

```python
# MinHash-based deduplication
df = sp.read_parquet("/3fs/data/")

# Compute MinHash signatures
df = df.with_column('minhash', 
    sp.func.minhash(df['text'], num_perm=128)
)

# LSH bucketing
df = df.with_column('bucket',
    sp.func.lsh_bucket(df['minhash'], num_bands=32)
)

# Find duplicates
duplicates = df.groupby('bucket').agg({
    'id': 'collect_list'
}).filter("len(id) > 1")

# Remove duplicates (keep first)
unique_ids = sp.sql("""
    SELECT DISTINCT FIRST_VALUE(id) OVER (PARTITION BY bucket ORDER BY id) as id
    FROM df
""")
df = df.join(unique_ids, on='id', how='inner')
```

### Quality Filtering

```python
# Filter training data by quality score
df = sp.read_parquet("/3fs/raw/")

df = df.filter(
    (df['perplexity'] < 100) &
    (df['length'] > 50) &
    (df['language'] == 'en') &
    (df['quality_score'] > 0.7)
)

df.write_parquet("/3fs/filtered/")
```

## Integration with 3FS

### Direct 3FS Access

```python
# smallpond uses 3FS USRBIO for high performance
sp.configure(
    storage_backend='3fs',
    storage_options={
        'read_ahead': True,
        'io_depth': 32,
    }
)
```

### Scratch Space

```python
# Use 3FS for intermediate data
sp.configure(
    scratch_path="/3fs/scratch/",  # Fast intermediate storage
    cleanup_scratch=True           # Auto-cleanup after job
)
```

## Key Files

| File | Purpose |
|------|---------|
| `smallpond/__init__.py` | Main API |
| `smallpond/dataframe.py` | DataFrame implementation |
| `smallpond/executor.py` | Distributed execution |
| `examples/` | Usage examples |
| `docs/` | Documentation |
