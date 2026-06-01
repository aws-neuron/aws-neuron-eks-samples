# Rolling Forcing Optimization Summary

## Parallelism

### TP=4 (Tensor Parallelism)

12 attention heads split across 4 ranks (3 heads/rank). Q/K/V projections are ColumnParallel (each rank computes its head shard, no communication). O projection and FFN fc2 are RowParallel (local matmul + all_reduce to combine partial sums across ranks). Result: each rank does 1/4 of the attention and FFN compute.

### SP=2 (Sequence Parallelism)

The sequence of tokens (e.g., 12870 for 15 frames) is split across 2 SP ranks. Each rank processes half the tokens through QKV projection. Before attention, K/V are AllGathered across SP so each rank has full-sequence K/V for correct attention scoring. Q stays local (half-sequence) — this halves the attention compute (Q×K^T is O(seq_q × seq_k), reducing seq_q by 2x). After attention, output goes through O projection (RowParallel handles the TP communication).

### Combined: TP=4 × SP=2 = 8 ranks

Runs on a single Neuron Device (s-lnc1-trn2, LNC=1, 8 physical NeuronCores).

## Compilation Strategy

### Sub-module `fullgraph=True`

Individual Linear layers (Q/K/V, FFN fc1, GELU) compiled with `torch.compile(backend='neuron', fullgraph=True)`. Each compiles into a single optimized NEFF with no graph breaks. RowParallel layers (O, FFN fc2) stay eager because `dist.all_reduce` in subgroups crashes the Neuron compiler. NKI kernels (attention, RoPE, KV cache copy) run in eager between compiled sub-modules.

### `cache_size_limit=128`

Allows dynamo to cache multiple compiled variants (different input shapes from 15-frame main forward vs 3-frame cache-update) without falling back to eager.

## Attention Module (`dit_attention.py`)

Ported from the Neuron Science Team. Key features over the original:

- **SP-aware QKV:** `_local_qkv_norm` projects on local SP shard, `_gather_qkv` AllGathers across SP group to reconstruct full sequence for K/V
- **RoPE with grid caching:** `build_rope_grids` constructs 3D position embeddings, cached per (grid_size, start_frame) to avoid recomputation
- **`forward_merged`:** Combines cache-update (3 frames) and denoising (15 frames) into one call. Splits internally into cu/dn portions, runs separate cache writes and attention passes, but shares the QKV projection and output projection overhead
- **`_attend`:** Calls the science team kernel with `actual_seqlen_k` (no mask tensor needed) and `use_dynamic_loop=True` for batch processing

## Self-Attention Kernel (`self_attention_nst.py`)

Science team's 792-line optimized NKI kernel. Key differences from the original:

- **Flash attention with sectioning:** For sequences >10K tokens, splits K/V into 8K sections with online softmax (running max + running sum). Reduces SBUF pressure.
- **`actual_seqlen_k` masking:** No explicit mask tensor passed — kernel handles valid positions internally. Eliminates mask creation overhead in Python.
- **`ModularAllocator` for SBUF management:** Precise tile placement with modular addressing for circular buffering. Multiple tiles share physical SBUF addresses via `block_dim`/`num_free_tiles`.
- **`use_dynamic_loop` batch processing:** Iterates over attention heads using NKI `dynamic_range` (hardware loop) instead of Python loop.
- **No identity matrix hack:** Direct PV computation without the transpose-via-matmul workaround.
- **Requires LNC=1 (full NeuronCores):** Compiler crashes (`NCC_IXGM002`) on LNC=2 half-cores.

## KV Cache (`kv_cache_copy.py`)

Uses `nki_op` with `mutates_args` — enables in-place DMA writes to cache tensors via NKI kernel. Same DMA hardware as `tensor.copy_()` but integrated into the NKI dispatch path.

## Results

| Config | FPS | DiT/block | Hardware |
|--------|-----|-----------|----------|
| Original baseline | 0.43 | 18.0s | s-lnc2-trn2 (4 half-NCs) |
| + sub-module fullgraph | 0.88 | 7.5s | s-lnc2-trn2 (4 half-NCs) |
| + SP=2 + science team kernel + dit_attention | **1.20** | **4.0s** | s-lnc1-trn2 (8 full-NCs) |

## Branches

- `rolling-forcing-tp4`: 0.88 FPS, TP=4, s-lnc2-trn2 (4 half-NCs), stable
- `rolling-forcing-tp8-sp2`: 1.20 FPS, TP=4×SP=2, s-lnc1-trn2 (8 full-NCs), science team kernel

## Known Limitations

- Science team kernel requires LNC=1 — crashes with `NCC_IXGM002` on LNC=2
- `dist.all_reduce` with TP subgroups cannot be compiled (`replica id #0 not seen`) — O/FFN stay eager on 8-rank setup
- `cache_size_limit > 8` crashes `neuronx-cc` with the original pipeline code (filed as SDK bug)
