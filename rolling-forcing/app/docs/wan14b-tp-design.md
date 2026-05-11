# Wan2.1-T2V-14B with Tensor Parallelism (TP=8) on Trainium

## Overview

This document describes the implementation of Wan2.1-T2V-14B video generation
with tensor parallelism across 8 NeuronCores on 4 NeuronDevices (trn2).

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  4 NeuronDevices (ND0-ND3), 2 cores per device sharing HBM                  │
│                                                                               │
│  ND0 (HBM Bank 0)       ND1 (HBM Bank 1)       ND2 (HBM Bank 2)       ND3 (HBM Bank 3)│
│  ┌────────┬────────┐   ┌────────┬────────┐   ┌────────┬────────┐   ┌────────┬────────┐│
│  │ Rank 0 │ Rank 1 │   │ Rank 2 │ Rank 3 │   │ Rank 4 │ Rank 5 │   │ Rank 6 │ Rank 7 ││
│  │ DiT/8  │ DiT/8  │   │ DiT/8  │ DiT/8  │   │ DiT/8  │ DiT/8  │   │ DiT/8  │ DiT/8  ││
│  │ VAE    │        │   │        │        │   │ T5     │        │   │        │        ││
│  │ 5 hd   │ 5 hd   │   │ 5 hd   │ 5 hd   │   │ 5 hd   │ 5 hd   │   │ 5 hd   │ 5 hd   ││
│  └────────┴────────┘   └────────┴────────┘   └────────┴────────┘   └────────┴────────┘│
│  ≈19 GB                 ≈15 GB                 ≈21 GB                 ≈15 GB            │
│                                                                                          │
│              ← All-Reduce after O-proj & FFN-down across all 8 ranks →                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Layout Per HBM Bank

| Bank | Contents | Estimated Usage |
|------|----------|-----------------|
| Bank 0 (ND0) | Rank 0: DiT/8 (3.75GB) + VAE (0.66GB) + KV cache (1.79GB) + NEFFs<br>Rank 1: DiT/8 (3.75GB) + KV cache (1.79GB) + NEFFs | ≈ 19 GB |
| Bank 1 (ND1) | Rank 2: DiT/8 + KV cache + NEFFs<br>Rank 3: DiT/8 + KV cache + NEFFs | ≈ 15 GB |
| Bank 2 (ND2) | Rank 4: DiT/8 (3.75GB) + T5 (9.6GB) + KV cache (1.79GB) + NEFFs<br>Rank 5: DiT/8 (3.75GB) + KV cache (1.79GB) + NEFFs | ≈ 21 GB |
| Bank 3 (ND3) | Rank 6: DiT/8 + KV cache + NEFFs<br>Rank 7: DiT/8 + KV cache + NEFFs | ≈ 15 GB |

**Key insight**: T5 (rank 4, ND2) and VAE (rank 0, ND0) are on separate HBM banks.
This ensures VAE decode never OOMs due to T5 weights consuming shared HBM.

### Design Rationale

Rather than dedicating ranks to individual model components (e.g., 1 rank for TE,
1 for VAE, remaining for DiT), we assign **all 8 cores to DiT** via tensor parallelism.
T5 and VAE are additionally co-located on specific ranks. This is critical because:

1. **DiT is the bottleneck**: ~90%+ of inference time is in the 40 transformer blocks
2. **High utilization**: All 8 cores participate in every DiT forward pass
3. **Memory planning**: T5 and VAE on different HBM banks avoids OOM
4. **No idle ranks**: T5 encodes once at start; VAE decodes once at end; DiT uses all ranks throughout

### Inference Coordination Protocol

```
Step 1: Rank 0 tokenizes prompt (CPU, fast) → broadcasts token IDs to all ranks
Step 2: Rank 4 runs T5 (Neuron, compiled) → broadcasts embeddings to all ranks  
Step 3: All 8 ranks run DiT rolling forcing (TP=8, Neuron, NKI kernels)
Step 4: Rank 0 decodes latents with VAE (Neuron, compiled) → output frames
```

All models run on Neuron, compiled with `torch.compile(backend='neuron')`.

## Model Specifications (14B vs 1.3B)

| Parameter | 1.3B | 14B | 14B per-rank (TP=8) |
|-----------|------|-----|---------------------|
| dim | 2048 | 5120 | 5120 (replicated) |
| num_heads | 16 | 40 | 5 |
| ffn_dim | 8192 | 13824 | 1728 |
| num_layers | 30 | 40 | 40 |
| head_dim | 128 | 128 | 128 |
| KV cache | [B, S, 16, 128] | [B, S, 40, 128] | [B, S, 5, 128] |
| Params (total) | 1.3B | 14B | 1.99B |

## TP Sharding Strategy

| Layer | Strategy | Communication |
|-------|----------|---------------|
| Patch embedding | Replicated | None |
| Text embedding | Replicated | None |
| Time embedding/projection | Replicated | None |
| Self-Attention Q/K/V | Column-parallel | None |
| Self-Attention O | Row-parallel | **All-reduce** |
| Cross-Attention Q/K/V | Column-parallel | None |
| Cross-Attention O | Row-parallel | **All-reduce** |
| FFN fc1 (up) | Column-parallel | None |
| FFN fc2 (down) | Row-parallel | **All-reduce** |
| Norms, modulation | Replicated | None |
| Head | Replicated | None |

**Total communication**: 3 all-reduces per block × 40 blocks = **120 all-reduces** per forward pass.

## Compilation Strategy

| Component | Method | Reason |
|-----------|--------|--------|
| T5 (full model) | `torch.compile(backend='neuron')` | Static shape, no state |
| VAE (full model) | `torch.compile(backend='neuron')` | Static shape, no state |
| DiT patch_embedding | `torch.compile(backend='neuron')` | Pure, static |
| DiT text_embedding | `torch.compile(backend='neuron')` | Pure, static |
| DiT time_embedding | `torch.compile(backend='neuron')` | Pure, static |
| DiT time_projection | `torch.compile(backend='neuron')` | Pure, static |
| DiT head | `torch.compile(backend='neuron')` | Pure, static |
| DiT FFN (×40 blocks) | `torch.compile(backend='neuron')` | Pure: Linear→GELU→Linear |
| Self-attention | NKI kernel (wrap_nki HOP) | Custom flash attention with KV cache |
| Cross-attention | NKI kernel (wrap_nki HOP) | Custom flash cross-attention |
| RoPE | NKI kernel (wrap_nki HOP) | Custom rope rotation |
| DiT top-level forward | Python (not compiled) | Dynamic KV cache control flow |

## File Structure

```
models/
├── tp_utils.py                    # TP primitives (Column/RowParallelLinear, shard_model_tp)
├── causal_model_tp.py             # CausalWanModelTP (TP-aware model definition)
├── causal_model_wrapper_tp.py     # WanDiffusionWrapperTP (loads + shards weights)
├── causal_inference_pipeline_tp.py # CausalInferencePipelineTP (rolling-forcing with TP)
├── layers.py                      # NKI kernel loading and layer definitions
kernels/
├── self_attention.py              # NKI flash self-attention with KV cache
├── cross_attention.py             # NKI flash cross-attention
├── rope.py                        # NKI RoPE rotation
├── kv_cache_copy.py              # KV cache update (tensor.copy_() DMA)
configs/
├── rolling_forcing_dmd_14b_tp8.yaml # Config for 14B with TP=8
├── rolling_forcing_dmd_14b_tp4.yaml # Config for 14B with TP=4
inference_neuron_tp.py              # Entry point (torchrun compatible, FastAPI server)
run_inference_neuron_tp.sh          # Launch script
```

## Usage

### 1. Download Model Weights

```bash
# Cache Wan2.1-T2V-14B weights from HuggingFace
aws s3 cp s3://your-bucket/wan_models/Wan2.1-T2V-14B/ wan_models/Wan2.1-T2V-14B/ --recursive
```

### 2. Run Inference Server

```bash
# Launch with TP=8 across 8 NeuronCores
torchrun --nproc_per_node=8 inference_neuron_tp.py

# Or use the launch script
./run_inference_neuron_tp.sh
```

### 3. Generate Video

```bash
# Health check
curl http://localhost:8000/health

# Generate video
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on a sunny beach", "seed": 42}'

# Stream frames
curl -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on a sunny beach", "seed": 42}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TP_DEGREE` | 8 | Number of TP ranks |
| `T5_RANK` | 4 | Which rank hosts T5 encoder |
| `CONFIG_PATH` | `configs/rolling_forcing_dmd_14b_tp8.yaml` | Config file |
| `MODEL_PATH` | `wan_models/Wan2.1-T2V-14B` | Model weights directory |
| `VAE_PATH` | `wan_models/Wan2.1-T2V-14B/Wan2.1_VAE.pth` | VAE weights |
| `DEFAULT_NUM_FRAMES` | 161 | Frames per video (10s at 16fps) |
| `DEFAULT_FPS` | 16 | Output video FPS |

## Memory Budget (per rank, bf16, TP=8)

| Component | Size |
|-----------|------|
| DiT weights (sharded, 1.99B params) | ~3.75 GB |
| KV cache (40 layers × [1, 37440, 5, 128]) | ~1.79 GB |
| Shared attention buffers | ~0.08 GB |
| NEFFs (compiled code + constants) | ~0.4 GB |
| Collectives buffers | ~0.38 GB |
| Scratchpad | ~0.5 GB |
| **DiT-only rank total** | **~6.9 GB** |
| + T5 (rank 4 only) | +9.6 GB |
| + VAE (rank 0 only) | +0.66 GB |

## Key Implementation Details

### Lockstep Execution
All 8 ranks execute identical Python control flow. They receive the same noise
(seeded identically), same embeddings, and same scheduling decisions. The only
difference is the weight/KV shards each rank holds.

### NKI Kernel Compatibility
The existing NKI kernels (self-attention, cross-attention, RoPE) work with
fewer heads per rank. They operate per-head so naturally adapt to 5 heads
instead of 40.

### KV Cache
Sized for local heads only: `[1, 37440, 5, 128]` per layer per rank.
The cache eviction, anchor block, and working cache logic are identical to
the single-rank version. After DiT completes, `release_device_memory()` frees
the KV cache for reuse on the next request.

### Weight Loading
Loads full HuggingFace weights on each rank, then `shard_model_tp()` splits
attention/FFN weights in-place, discarding non-local shards.

### Rolling Forcing Windows
With `num_frame_per_block=3` and 21 latent frames:
- Window 0: frames 0-14 (initial, 23400 tokens)
- Windows 1-10: 3 new frames each (4680 tokens)
- Only 2 unique compiled shapes needed

## References

- [RollingForcing](https://github.com/TencentARC/RollingForcing) — Causal video generation
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — Base model architecture
- [CausVid](https://arxiv.org/abs/2412.07772) — Few-step video distillation
