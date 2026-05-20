# Rolling Forcing — Neuron Profiling Results

**Date:** 2026-05-20  
**Instance:** trn2.48xlarge  
**Model:** Wan2.1-T2V-1.3B (DiT + T5 + VAE)  
**Config:** TP=4, 81 latent frames, 5 denoising steps, `rolling_forcing_dmd_f81_b1.yaml`  
**Profile:** Session-level device trace via `NEURON_RT_INSPECT_DEVICE_PROFILE=session`

---

## Benchmark Results (from profiling run)

| Metric | Value |
|--------|-------|
| Pixel frames generated | 243 |
| Latent frames | 81 |
| Blocks processed | 27 |
| Total time | 1455.1s |
| Compilation time (block 0) | 1033.1s |
| T5 encode time | 0.14s |
| Steady-state DiT/block | 8.3s |
| Steady-state VAE/block | 2.8s |
| Steady-state block E2E | 11.1s |
| **Overall FPS** | **0.17** |
| **Streaming FPS** | **0.26** |
| **Steady-state FPS** | **0.77** |
| VAE decode FPS | 1.09 |
| Real-time ratio (vs 16fps) | 0.048x |
| **Speedup needed for real-time** | **20.7x** |

---

## NEFF Profile Analysis (Rank 0, NC 0)

### Summary

| Metric | Value |
|--------|-------|
| **Total NEFFs** | **437** (per rank) |
| **Total NEFF size** | 66.43 MB |
| **NTFF trace size** | 4.5 GB |
| **Total profile time** | 1535.2s |
| **Model FLOPS** | 54.1 TFLOPS |
| **MFU** | **0.03%** |
| **Total active time** | 0.417s |
| **Total kernel launches** | 425,789 |
| **Executions profiled** | 16,121 (before dropped notifications) |

### NEFF Size Distribution

| Bucket | Count | % | Interpretation |
|--------|-------|---|----------------|
| <10KB (tiny) | 1 | 0.2% | Single scalar op |
| **10-100KB (small)** | **351** | **80.3%** | **Individually compiled sub-modules** |
| 100KB-1MB (medium) | 73 | 16.7% | NKI kernels (attn, rope) |
| >1MB (large fused) | 12 | 2.7% | T5/VAE full models |

### Top 12 Largest NEFFs (main compute kernels)

| Size | Likely Component |
|------|-----------------|
| 13 MB | T5 encoder (full model, torch.compile) |
| 4.1 MB | VAE decoder (full model, torch.compile) |
| 3.7 MB | NKI self_attention (initial window: q=[3,128,12928]) |
| 2.7 MB | NKI self_attention (streaming window: q=[3,128,2688]) |
| 2.6 MB | NKI cross_attention (initial window) |
| 2.4 MB | NKI cross_attention (streaming window) |
| 1.9 MB | NKI rope (initial window) |
| 1.8 MB | NKI rope (streaming window) |
| 1.6 MB | VAE conv2d_k1 or self_attention |
| 1.1 MB | VAE conv2d_k3 |

### 351 Small NEFFs Breakdown (the problem)

These 351 NEFFs (10-100KB each) are the **individually compiled sub-modules**:

| Sub-module | Count (estimated) | NEFFs per |
|-----------|-------------------|-----------|
| `block.ffn` × 30 layers × 2 shapes | 60 | 21-52KB each |
| `patch_embedding` × 2 shapes | 2 | ~30KB |
| `text_embedding` | 1 | ~25KB |
| `time_embedding` | 1 | ~25KB |
| `time_projection` | 1 | ~20KB |
| `head` × 2 shapes | 2 | ~30KB |
| Intermediate ops (norms, reshapes, etc.) | ~284 | ~21KB each |

Each of these triggers a **separate kernel launch** with Python-level scheduling.

---

## Critical Finding: 99.97% of Time is Overhead

```
Total profile time:     1535.2s
NeuronCore active time:    0.417s  (0.03%)
Python + scheduling:    1534.8s  (99.97%)
```

The NeuronCores are idle 99.97% of the time. The bottleneck is:

1. **437 NEFFs launched 425,789 times** = massive kernel scheduling overhead
2. **Python control flow** between every DiT block (KV cache indexing, eviction, shape management)
3. **Sequential execution**: each small NEFF must complete before the next starts

### MFU Analysis

```
Model FLOPS:          54.1 TFLOPS (total across all blocks)
Peak FLOPS (trn2):    ~380 TFLOPS/core (bf16)
Active time:          0.417s
Achieved TFLOPS:      54.1 / 0.417 = 130 TFLOPS/s during active time
MFU during compute:   130 / 380 = 34% (reasonable!)
```

**The compute itself is fine (34% MFU when active).** The problem is that NeuronCores are active for only 0.4s out of 1535s.

---

## Optimization: Aggressive Fusion Strategy

### Current Compilation (437 NEFFs per rank)

```python
# Each sub-module compiled separately:
dit_model.patch_embedding = torch.compile(dit_model.patch_embedding, backend='neuron')
dit_model.text_embedding = torch.compile(dit_model.text_embedding, backend='neuron')
dit_model.time_embedding = torch.compile(dit_model.time_embedding, backend='neuron')
dit_model.time_projection = torch.compile(dit_model.time_projection, backend='neuron')
dit_model.head = torch.compile(dit_model.head, backend='neuron')
for block in dit_model.blocks:
    block.ffn = torch.compile(block.ffn, backend='neuron')
# NKI kernels: self_attn, cross_attn, rope (separate kernel launches)
```

### Target: Fused Compilation (~30 NEFFs per rank)

Compile **entire transformer blocks** as single units:

```python
# Fuse the whole block forward pass (minus KV cache I/O):
#   norm1 → self_attn → residual → norm2 → cross_attn → residual → norm3 → FFN → residual
# 
# This collapses: norm + QKV_proj + rope + self_attn + O_proj + 
#                 norm + cross_QKV_proj + cross_attn + cross_O_proj +
#                 norm + FFN (fc1 + GELU + fc2) + residuals
# Into a SINGLE NEFF per block per shape.
#
# 30 blocks × 2 shapes = 60 NEFFs (vs 351 small NEFFs today)
# Plus: T5(1) + VAE(1) + embeddings(~5) = ~67 total

for i, block in enumerate(dit_model.blocks):
    block = torch.compile(block, backend='neuron', dynamic=False)
```

### Expected Improvement

| Metric | Current | After Fusion | Improvement |
|--------|---------|-------------|-------------|
| NEFFs per rank | 437 | ~67 | **6.5x fewer** |
| Kernel launches per block | ~15 | ~2 | **7.5x fewer** |
| Python scheduling overhead | 1534.8s | ~200s (est.) | **7.7x** |
| Streaming FPS | 0.77 | ~5-6 (est.) | **7-8x** |

### Implementation Challenges

1. **NKI kernels inside the block**: `wrap_nki` HOP (Higher Order Primitive) calls for self_attn, cross_attn, rope must be compatible with `torch.compile` wrapping the outer block. This works if the NKI kernel is registered as a custom op.

2. **KV cache is dynamic state**: The block's forward pass reads/writes KV cache indexed by frame position. Options:
   - Pass cache slices as function arguments (makes the compiled graph static)
   - Use `torch.compiler.allow_in_graph` for cache ops
   - Keep cache I/O outside the compiled block boundary

3. **Two input shapes**: Initial window (15 frames → 12928 tokens) and streaming window (3 frames → 2688 tokens). Need 2 compiled variants per block.

4. **all_reduce inside blocks**: TP communication (all_reduce for RowParallel outputs) must be preserved inside the compiled graph. Neuron backend handles this.

---

## Files & Artifacts

| Artifact | Location |
|----------|----------|
| Profile artifacts | `/var/mdl/rolling_forcing/profiles/dit_vae_20260520_190632/` |
| NTFF trace (rank 0) | `i-0c2d160cb9d13cf60_pid_823/profile_nc_0_session_0.ntff` |
| Summary text | `/var/mdl/rolling_forcing/profile_json_output/rolling_forcing_summary.txt` |
| JSON profile | `/var/mdl/rolling_forcing/profile_json_output/rolling_forcing_profile.json` |
| Full JSON (1.4 GB) | `/var/mdl/rolling_forcing/profile_json_output/i-0c2d160cb9d13cf60_pid_823_nc_0_session_0.json` |

---

## Next Steps

1. ✅ Profile captured (this doc)
2. → **Implement whole-block fusion** (`torch.compile(block, ...)`)
3. → Re-profile and compare NEFF count + MFU
4. → If still overhead-bound, consider fusing multiple blocks together (e.g., 5 blocks as one graph)
5. → Validate video quality is unchanged after fusion
