# Sequence Parallelism for Wan2.1-T2V-1.3B DiT

## Overview

Apply Sequence Parallelism (SP) alongside existing TP=4 to reduce per-rank compute by ~3.8x. Each rank processes seqlen/4 tokens for Linear/Norm ops, then AllGathers K/V before attention.

## Current State (TP=4 only)

- Input per rank: `[1, 12870, 1536]` (full sequence, all ranks identical)
- QKV projection: ColumnParallel splits output dim (1536→384 per rank)
- Attention: 3 heads per rank, full sequence length
- O/FFN: RowParallel with all-reduce
- RMSNorm: TPRMSNorm (all-reduces sum-of-squares for global RMS)

## Proposed: TP=4 + SP=4

- Input per rank: `[1, 3217, 1536]` (sequence split across 4 ranks)
- QKV projection: same ColumnParallel, but on 1/4 the tokens
- Before attention: AllGather K and V to full sequence
- Attention: Q is local (3217 tokens), K/V are global (12870 tokens)
- After attention: ReduceScatter output back to SP shards

## Compute/Communication Tradeoff

| Metric | TP only | TP + SP=4 |
|--------|---------|-----------|
| Tokens per rank (Linear/Norm) | 12,870 | 3,217 |
| FLOPs per rank per block | ~106B | ~28B |
| AllGather per block | 0 | 2 (K + V, ~10MB each) |
| ReduceScatter per block | 0 | 2 (O + FFN, ~5MB each) |
| Total extra comms per block | 0 | ~30MB |
| Est. comm time (500 GB/s) | 0 | ~60µs |
| Compute:Comm ratio | — | 236:1 |

## Implementation Changes

### 1. Split input sequence before block loop

```python
# In _forward_inference, after patch_embedding:
x = x.flatten(2).transpose(1, 2)  # [1, seqlen, dim]
# SP: split sequence across ranks
sp_rank = get_tp_rank()  # reuse TP group for SP
chunk_size = x.shape[1] // tp_degree
x = x[:, sp_rank * chunk_size:(sp_rank + 1) * chunk_size]
```

### 2. AllGather K/V in self-attention

```python
# After QKV projection (ColumnParallel, local):
q = self.q(x).view(b, s_local, n, d)  # [1, 3217, 3, 128]
k = self.k(x).view(b, s_local, n, d)
v = self.v(x).view(b, s_local, n, d)

# AllGather K and V
k = all_gather_along_seq(k, dim=1)  # [1, 12870, 3, 128]
v = all_gather_along_seq(v, dim=1)  # [1, 12870, 3, 128]

# Q stays local: [1, 3217, 3, 128]
```

### 3. ReduceScatter after RowParallel

```python
# Current RowParallelLinear does all-reduce
# Replace with ReduceScatter: combines reduce + scatter along seq dim
out = linear(x, weight)  # local matmul
out = reduce_scatter(out, dim=1)  # [1, 3217, 1536] per rank
```

### 4. RoPE position adjustment

Each rank applies RoPE with offset positions:
```python
# Rank 0: positions [0, 3217)
# Rank 1: positions [3217, 6434)
# etc.
start_pos = sp_rank * chunk_size
rope_positions = start_pos + torch.arange(chunk_size)
```

### 5. KV cache

Two options:
- **Gather before cache write**: AllGather K/V, write full sequence to cache (current cache structure unchanged)
- **Shard the cache**: Each rank stores only its SP shard. Requires AllGather at cache read time.

## Complications

- KV cache management assumes full sequence on each rank
- The rolling forcing pipeline passes full-sequence tensors to the generator
- Cross-attention context (T5 embeddings) is replicated — no SP needed there
- First-frame anchor block in cache needs special handling

## Reference

SDE team achieved 1.7s → 1.25s (1.36x) on 8-core with SP on QKV + RMSNorm for a different model. Expected similar gains here.

## Status

Parked. Requires architectural refactoring of cache management. Pursue after NEFF count reduction is resolved with SDK team.
