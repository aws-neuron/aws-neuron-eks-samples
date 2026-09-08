# Video Quality Improvement — Rolling Forcing DMD on AWS Neuron

## Setup

We have a **Wan2.1-T2V-1.3B** video generation model running on **AWS Neuron** (trn1/inf2 instances) using **Rolling Forcing** autoregressive inference with **DMD (Distribution Matching Distillation)** for few-step generation. The system streams generated video frames via a FastAPI server.

### Hardware & Architecture
- **Instance:** trn1.32xlarge node
- **Pod allocation:** 2 Neuron Devices (4 NeuronCores with lnc=2)
- **DiT model:** Wan2.1-T2V-1.3B on `neuron:0` (ND0, NC0+NC1)
- **T5 text encoder:** on `neuron:2` (ND1, NC0+NC1)
- **VAE decoder:** on `neuron:3` (ND1, NC2+NC3)
- **Precision:** bfloat16 throughout
- Custom NKI kernels for self-attention, cross-attention, RoPE, and KV-cache copy

### Current Config (`rolling_forcing_dmd_f21_b1_med.yaml`)
```yaml
model_name: Wan2.1-T2V-1.3B
generator_ckpt: checkpoints/ode_init.pt       # DMD distilled checkpoint
denoising_step_list: [1000, 800, 600, 400, 200]  # 5 denoising steps
num_frame_per_block: 1
image_or_video_shape: [1, 21, 16, 44, 78]     # 21 frames, 44×78 latent (~352×624 video)
timestep_shift: 5.0
warp_denoising_step: true
distribution_loss: dmd
mixed_precision: true
```

### Inference Flow
1. T5 encodes text prompt → prompt embeddings
2. DiT runs rolling forcing loop: slides a denoising window across frame blocks; each block passes through all 5 denoising steps progressively
3. Finalized latent blocks are decoded by VAE one frame at a time
4. Frames are JPEG-encoded and streamed via HTTP (Server-Sent Events)

## Problem

The generated videos have **poor visual quality** — blurry, low detail, poor temporal coherence. The system works functionally (frames are generated and streamed correctly for 17 and 21 frames), but output quality is bad.

## What We Ruled Out

### NKI Kernels — NOT a quality factor
The custom NKI kernels (`self_attention.py`, `cross_attention.py`, `rope.py`, `kv_cache_copy.py`) are **numerically equivalent** to CPU/GPU reference implementations. This was verified at **three levels**:

#### Level 1: Kernel-level unit tests (`tests/wan_kernels/`)
- **`test_attention_kernel.py`** — Tests `wan_flash_self_attn` and `wan_cross_attn` NKI kernels directly against a manual `ref_attention()` implementation (QK^T → softmax → PV). Tests 6 self-attention shapes (batch=1/12, seqlen_q up to 23400, seqlen_k up to 18720) and 2 cross-attention shapes. Asserts `torch.allclose(rtol=1e-2, atol=1e-3)`.
- **`test_rope_kernel.py`** — Tests `causal_rope_rotation` and `build_rope_grids` NKI kernels against CPU rotate_half reference AND end-to-end against `causal_rope_apply`. Tests 4 grid configurations (full block, anchor block, varying start_frame). Asserts `torch.allclose(rtol=1e-2, atol=1e-3)`.
- **`test_kv_cache_copy.py`** — Tests `cache_copy` and `kv_cache_copy` NKI kernels for exact bitwise correctness (`rtol=0, atol=0`) across 4 production seqlen shapes.

#### Level 2: Module-level integration tests (`tests/wan_modules/`)
- **`test_wan_self_attn.py`** — Full end-to-end test: `RefCausalSelfAttention` (CPU, uses `F.scaled_dot_product_attention`) vs `CausalWanSelfAttention` (Neuron, uses NKI kernels). Same weights, same inputs. Simulates **46 rolling forcing windows** (W0–W12 + W42–W45) covering all code paths: eviction vs no-eviction, anchor vs non-anchor writes, cache-update vs normal denoising, varying `k_len_int` and `valid_tokens`, ramp-down with decreasing `num_valid_frames`. Each window asserts `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`.
- **`test_wan_cross_attn.py`** — Tests `WanT2VCrossAttention` (Neuron) vs `RefCrossAttention` (CPU) end-to-end with shared weights. Covers first call (KV projection + cache init) and second call (cached KV path). Also includes a pure-compute test chaining JIT'd projections + NKI cross-attention kernel.
- 13 additional module tests covering every sub-component: QKV projections, RMSNorm, FFN, patch embed/unpatchify, modulated norm, sinusoidal embedding, flow prediction conversion, attention block, causal head, causal model, and the full inference pipeline.

#### Level 3: Diagnostic scripts (root directory)
- **`self_attn_diag.py`** — Phase 1: Validates PyTorch SDPA reference at all 9 production shapes (anchor, 2/5 blocks, full cache, edge cases). Cross-validates manual implementation against `F.scaled_dot_product_attention`. Phase 2: Tests NKI kernel against the reference on Neuron device with real padding/masking.
- **`self_attn_wiring_diag.py`** — Replicates the **EXACT code path** from `layers.py` `CausalWanSelfAttention.forward()` Phase 4: Q padding to 128-multiple, mask construction, `num_sections` calculation, kernel call, output slicing. Tests 19 production shapes including all 17-frame and 21-frame configurations plus edge cases. Reports max_diff and mean_diff for each.

**Conclusion:** The NKI kernels affect inference speed, not output quality. Optimizing them further will not change the visual output.

### `guidance_scale` — NOT USED
**Validated:** The config contains `guidance_scale: 3.0` but this parameter has **zero references in any Python file** — neither the Neuron inference code nor the GPU reference code reads it. It is a dead training-config artifact. Changing it does nothing during inference.

### Memory/Pod issues — RESOLVED
Earlier OOM/pod-kill issues were fixed by properly sizing memory limits (220Gi) and extending liveness probe timeouts.

## Validated Quality Levers

### 1. Resolution (Latent Spatial Dimensions)

**Status: Validated — Can be changed at inference time**

Two resolution tiers exist and are proven to work:

| Config | Latent Size | Video Resolution | `frame_seq_length` |
|--------|-------------|------------------|--------------------|
| Small  | 30×52       | ~240×416         | 390                |
| Medium | 44×78       | ~352×624         | 858                |

The model architecture is resolution-agnostic (patch-based DiT). Higher latent dimensions produce more patches per frame, giving better spatial detail. The constraint is **HBM memory per rolling forcing window** — larger latents need more memory for KV cache, noise buffers, and intermediate activations.

**How to go higher:** Create a new config with larger latent dims (e.g., `60×106` → ~480×848). This requires:
- Ensuring HBM fits on the allocated NeuronCores
- Updating `frame_seq_length = (H × W) // 4`
- Updating `image_or_video_shape` accordingly

### 2. `num_frame_per_block` (Temporal Coherence)

**Status: Validated — Can be changed at inference time**

Currently set to `1` — each rolling forcing block contains 1 frame. The model processes frames individually within the denoising window, which can cause temporal flickering.

Increasing to `3` or `5` means the model jointly denoises multiple frames per block, improving temporal coherence. Constraint: more frames per block = larger window = more HBM memory.

**Coupling:** The checkpoint `ode_init.pt` was likely trained with `num_frame_per_block: 1`. Changing this at inference time may work (the architecture supports it), but the quality gain depends on whether the training used the same block size. **Needs validation.**

### 3. Denoising Steps — CANNOT be changed independently

**Status: Validated — Coupled to DMD training**

The `denoising_step_list: [1000, 800, 600, 400, 200]` defines the 5 noise levels the DMD-distilled model was **trained** to denoise from. The rolling forcing window size equals `len(denoising_step_list)`.

**Why you can't just add more steps:** DMD distillation teaches the model to predict clean output from specific noise levels. The model has NOT learned transitions for intermediate levels (e.g., 900, 700, 500). Adding them would produce garbage — the model doesn't know what to do at those noise levels.

**To get more denoising steps:** Must **retrain** the DMD checkpoint with the desired step schedule (e.g., 8 or 10 steps with different noise levels). This is a training-side change.

## Questions for Rolling Forcing / DMD Experts

1. Was `ode_init.pt` trained with `num_frame_per_block: 1`? If so, can we safely change to 3 at inference time, or does this require retraining?

2. What is the maximum resolution (latent H×W) that the Wan2.1-T2V-1.3B model supports for quality generation? Is there an upper limit beyond which the model produces artifacts regardless of memory?

3. How many DMD denoising steps are needed for acceptable quality? Does going from 5 to 8 or 10 steps significantly improve output, and does this require retraining from scratch or fine-tuning?

4. Is there a non-DMD inference mode (e.g., full DDPM/DDIM with 50+ steps) supported by this architecture that could be used for quality validation, even if slow?

5. Does `context_noise` (currently `0.0`) affect output quality? The GPU reference code has commented-out context noise injection during the cache update phase.

## Summary Table

| Lever | Effect | Can Change at Inference? | Cost |
|-------|--------|--------------------------|------|
| Resolution (latent H×W) | Spatial detail | ✅ Yes | More HBM memory |
| `num_frame_per_block` | Temporal coherence | ⚠️ Maybe (needs validation) | More HBM memory |
| More denoising steps | Overall denoising quality | ❌ No — requires retraining | Slower + retraining |
| `guidance_scale` | N/A | ❌ Not used in code | N/A |
| Better checkpoint | Overall quality | ❌ Requires training | Training compute |
| NKI kernel optimization | N/A (speed only) | N/A | N/A |
