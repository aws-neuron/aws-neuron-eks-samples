# Rolling Forcing Inference: Optimization Notes

This document describes all changes between the original (baseline) inference files and the
optimized (`_opt`) versions. The optimized code produces **bitwise-identical output** while
making all tensor shapes through the DiT model static, enabling future use of CUDA graphs
and `torch.compile`.

## File Mapping

| Original | Optimized |
|---|---|
| `pipeline/rolling_forcing_inference.py` | `pipeline/rolling_forcing_inference_opt.py` |
| `utils/wan_wrapper.py` | `utils/wan_wrapper_opt.py` |
| `wan/modules/causal_model.py` | `wan/modules/causal_model_opt.py` |
| `wan/modules/attention.py` | `wan/modules/attention_opt.py` |
| `wan/modules/model.py` | `wan/modules/model_opt.py` |
| `inference.py` | `inference_opt.py` |
| `run.sh` | `run_opt.sh` |

---

## 1. Pipeline: `rolling_forcing_inference_opt.py`

### 1.1 Static-shape padding for generator input

**Original:** The pipeline passes variable-sized tensors to the generator. The rolling window
spans 1–5 blocks (3–15 frames), so `noisy_input` changes shape every iteration during ramp-up
and ramp-down.

**Optimized:** Pre-allocate `padded_input` and `padded_timestep` at the fixed maximum window
size (`max_frames = 15`). A padded `noisy_cache` holds partially-denoised data; each iteration
fills `padded_input` with a single static `.copy_()` of `max_frames` from `noisy_cache`
(OOB-safe due to padding), overwrites `nfpb` frames with fresh noise when a new block enters,
and copies the timestep from a pre-computed pattern (see section 1.6). The full padded tensors
are passed to the generator with `num_valid_frames`. The full `max_frames` output is copied
to a padded `output` tensor (see section 1.8).

```python
# Static copy: always max_frames from noisy_cache (OOB-safe due to padding)
padded_input.copy_(noisy_cache[:, current_start_frame : current_start_frame + max_frames])

# Overwrite nfpb frames with fresh noise (ramp-up/steady only)
if current_num_frames == max_frames or current_start_frame == 0:
    noise_offset = current_num_frames - nfpb
    padded_input[:, noise_offset : noise_offset + nfpb].copy_(
        noise[:, current_end_frame - nfpb : current_end_frame])

# Static timestep copy from pre-computed patterns
padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]

_, denoised_pred = self.generator(
    noisy_image_or_video=padded_input,       # always [B, 15, C, H, W]
    timestep=padded_timestep,                # always [B, 15]
    num_valid_frames=num_valid_frames, ...)

# Static-length copy to output (always max_frames, see section 1.8)
output[:, current_start_frame:current_start_frame + max_frames].copy_(denoised_pred)
```

### 1.2 Dedicated 3-frame tensors for cache-update call

The cache-update call (second generator call per window) always processes `nfpb=3` frames.
Dedicated 3-frame tensors are pre-allocated once — no padding, no stale data:

```python
cache_input = torch.zeros([B, nfpb, C, H, W], ...)       # overwritten each window
cache_timestep = torch.full([B, nfpb], context_noise)     # constant
cache_sigma = torch.full([B, nfpb], context_sigma)        # constant

cache_input.copy_(denoised_pred[:, :nfpb])
self.generator(noisy_image_or_video=cache_input, timestep=cache_timestep,
               num_valid_frames=nfpb, updating_cache=True, sigma=cache_sigma, ...)
```

The model receives exactly 3 frames (no 12-frame padding waste).

### 1.3 Python int KV cache indices

**Original:** `global_end_index` and `local_end_index` are GPU scalar tensors requiring
`.item()` and `.fill_()` calls that cause GPU-CPU synchronization.

**Optimized:** Plain Python `int`s. Direct assignment: `kv_cache["global_end_index"] = cache_end`.

### 1.4 Shared KV buffers across all layers

**Original:** Cache eviction uses `.clone()` to copy tokens during the left-shift, allocating
a temporary tensor on every eviction. Each of the 30 transformer layers has its own
`buffer_k`/`buffer_v` pair stored in the per-layer KV cache dict.

**Optimized:** A single `shared_buffer_k`/`shared_buffer_v` pair is allocated once at the
pipeline level (size: 32,768 tokens = `max_attention_size` aligned up to 8,192). All 30
layers reuse the same pair since they execute sequentially. The buffers are passed as
`shared_buffers=(buffer_k, buffer_v)` through wrapper → model → self-attention.

These buffers serve dual purpose:
- **Scratch space** during cache eviction left-shift (replaces `.clone()`)
- **KV input** to the single attention call (see section 3.2)

Memory savings: ~8.8 GB (eliminates 29 redundant buffer pairs, each `[B, 32768, 12, 128]`
in bf16).

### 1.5 KV cache allocation

KV cache tensors are allocated at 37,440 tokens (`24 × 1560`), matching the logical eviction
boundary. Phase 3 copies use dynamic lengths, so no padding beyond the logical size is needed.

```python
kv_cache_alloc_size = 1560 * 24  # 37440
```

### 1.6 Pre-computed timestep patterns

**Original:** Each iteration builds a timestep tensor dynamically via `torch.cat` and per-block
scalar assignments.

**Optimized:** `_build_timestep_patterns()` pre-computes all 9 unique timestep patterns
(`2*nds - 1` where `nds=5`) on the CPU in `__init__` using plain Python lists:

- **Pattern 0:** Steady-state — denoising steps in reverse, each repeated `nfpb` times
- **Patterns 1..4:** Ramp-up — tail of steady pattern (growing window from start)
- **Patterns 5..8:** Ramp-down — head of steady pattern (shrinking window from end)

`pattern_indices` is a plain Python list computed in the same loop as `window_start/end_blocks`:

```python
if num_blks == nds:
    pattern_indices.append(0)                  # steady-state
elif start_block == 0:
    pattern_indices.append(num_blks)            # ramp-up
else:
    pattern_indices.append(nds - 1 + num_blks)  # ramp-down
```

The patterns tensor is lazily moved to GPU on first inference call.

### 1.7 Static copy lengths in the denoising loop

All `.copy_()` operations in the denoising loop use static (constant) lengths:

| Operation | Copy length | Value |
|---|---|---|
| `noisy_cache` → `padded_input` | `max_frames` | 15 frames |
| `noise` → `padded_input` (fresh noise) | `nfpb` | 3 frames |
| `timestep_patterns` → `padded_timestep` | `max_frames` | 15 values |
| `denoised_pred` → `output` | `max_frames` | 15 frames |
| `denoised_pred` → `cache_input` (cache-update) | `nfpb` | 3 frames |

The `noisy_cache` is allocated with `num_output_frames + max_frames` to prevent OOB when
the static `max_frames` copy reads past the last valid frame during ramp-down windows.

### 1.8 Static-length output copy

**Original:** `denoised_pred` is sliced to `num_valid_frames` (dynamic, 3–15) and copied
into `output` with a dynamic-length assignment.

**Optimized:** The full `denoised_pred` (always `max_frames = 15`) is copied to `output`
with a static-length `.copy_()`. The `output` tensor is padded to
`num_output_frames + max_frames - nfpb` to accommodate the overshoot.

```python
output[:, current_start_frame:current_start_frame + max_frames].copy_(denoised_pred)
```

Garbage frames in the padding zone are harmless:
- **Ramp-up:** Later windows overwrite the same positions with correct values.
- **Ramp-down:** `end_block` is always the last block, so garbage starts at
  `current_end_frame = num_output_frames`, landing entirely in the padding zone.

The output is trimmed before VAE decode: `output = output[:, :num_output_frames]`.

The re-noising loop still needs `num_valid_frames` for noise generation (to preserve RNG
draw count for bitwise match). A `noise_template` tensor with the valid shape is constructed
for `torch.randn_like`:

```python
noise_template = torch.empty(batch_size * num_valid_frames, *denoised_pred.shape[2:],
                              device=denoised_pred.device, dtype=denoised_pred.dtype)
full_noise = torch.randn_like(noise_template)
```

### 1.9 Simplified re-noising loop

**Original:** Each block's denoising step index is found by reading the timestep from
`padded_timestep`, then searching `denoising_step_list` via `torch.abs` + `torch.nonzero`
(GPU operations). `block_t` is constructed each iteration via `next_timestep * torch.ones(...)`.

**Optimized:** Two simplifications:

1. **Step index from window structure (pure Python):** The timestep pattern assigns step
   index `step_base - local_offset` where `step_base` depends on window type:
   - Ramp-up (`start_block == 0` and `num_blks < nds`): `step_base = num_blks - 1`
   - Steady / ramp-down: `step_base = nds - 1`

   No GPU reads, no `.item()`, no `torch.nonzero`.

2. **Pre-allocated `block_t` tensors:** One tensor per denoising step, created once before
   the denoising loop. Reused by index: `block_t = block_t_list[step_index + 1]`.

---

## 2. Wrapper: `wan_wrapper_opt.py`

### 2.1 `num_valid_frames` propagation

The wrapper accepts a new `num_valid_frames` parameter and passes it through to the model.

### 2.2 `shared_buffers` propagation

The wrapper accepts a new `shared_buffers` parameter and passes it through to the model.

### 2.3 Fully static x0 conversion

**Original:** Slices `noisy_image_or_video` and `timestep` to `n_valid` frames before
`_convert_flow_pred_to_x0`, producing variable-size output.

**Optimized:** Passes the full padded tensors. Padding frames produce garbage x0 values that
are harmless — they land in the `output` padding zone (see section 1.8) and are trimmed
before VAE decode.

```python
pred_x0 = self._convert_flow_pred_to_x0(
    flow_pred=flow_pred.flatten(0, 1),
    xt=noisy_image_or_video.flatten(0, 1),       # full static shape
    timestep=timestep.flatten(0, 1)               # full static shape
).unflatten(0, flow_pred.shape[:2])
```

---

## 3. Model: `causal_model_opt.py`

### 3.1 `num_valid_frames` instead of `grid_sizes` override

**Original:** Overrides `grid_sizes[:, 0] = num_valid_frames` before the transformer blocks,
making `unpatchify` extract only valid tokens — producing variable-size output.

**Optimized:** `grid_sizes` is never modified. `num_valid_frames` is passed through `kwargs`
to each `CausalWanAttentionBlock` → `CausalWanSelfAttention`, where it computes
`valid_tokens = num_valid_frames * frame_seqlen`. This Python int is passed as `valid_q`/`valid_k`
to `flash_attn_varlen_b1` for masking via the fake-batch-2 trick (see section 4). The model
output keeps the full padded frame count.

### 3.2 Decoupled 5-phase self-attention with static KV buffers

**Original:** `CausalWanSelfAttention.forward` interleaves cache management and attention
computation across 3 separate `attention()` call sites, each with different K/V shapes.

**Optimized:** Restructured into 5 clean phases:

```
Phase 1: QKV projection + RoPE
Phase 2: Cache management (write + eviction) — no attention calls
Phase 3: Assemble KV into shared buffer_k/buffer_v [B, 32768, N, D]
Phase 4: Single flash_attn_varlen_b1(roped_query, buffer_k, buffer_v, valid_q, valid_k) call
Phase 5: Output projection
```

**Phase 2** uses a unified index computation for both eviction and no-eviction paths.
All copy operations use static lengths:

```python
num_evicted = 0
if cache_overflows:
    num_evicted = ...
    evict_rolled = kv_cache_size - 2 * sink_tokens  # static: 28080
    # left-shift using shared buffer_k/buffer_v as scratch

local_end_index = local_end_index_current + num_new_tokens - num_evicted
local_start_index = local_end_index - block_length
# unified write
```

**Phase 3** has 2 branches (down from 3 in the original), using dynamic copy lengths:

| Branch | Condition | KV content |
|--------|-----------|------------|
| Cache-update | `updating_cache=True` | Full cache, `cache_len` tokens (dynamic, ≤ 32,760) |
| Normal / first block | `updating_cache=False` | Anchor + working cache + current, each copied at exact valid length |

The first-block case (no cache history) naturally falls out of the normal path when
`local_start_index == 0` — the anchor/working-cache block is skipped and only current
tokens are copied.

In the normal branch, working cache and current tokens are copied at their exact valid
lengths (`wc_len` and `valid_tokens`). No garbage beyond valid data enters the buffer.

```python
# Working cache: copy exact valid length
buffer_k[0, offset:offset + wc_len].copy_(kv_cache["k"][0, wc_start:wc_start + wc_len])
offset += wc_len
# Current tokens: copy exact valid length
buffer_k[0, offset:offset + valid_tokens].copy_(roped_key[0, :valid_tokens])
k_len_int = offset + valid_tokens
```

**Phase 4** — single `flash_attn_varlen_b1` call:
- Q: `[1, 23400, N, D]` (static — padded input, 15 frames × 1560 tokens/frame)
- K: `[1, 32768, N, D]` (static — shared buffer, valid data up to `k_len_int`)
- V: `[1, 32768, N, D]` (static — shared buffer)
- `valid_q`/`valid_k` Python ints drive fake-batch-2 cu_seqlens split (see section 4)

### 3.3 Working cache budget uses `valid_tokens`

**Original:** `query_length = roped_query.shape[1]` uses the full padded sequence length.

**Optimized:** `query_length = valid_tokens`. This matters for end-of-video windows where
padding is significant (e.g., 12 valid frames padded to 15).

### 3.4 `x.flatten(1, 2)` before `unpatchify`

`CausalHead` returns `[B, F, seq_per_frame, out_C]`. Without the `grid_sizes` override, the
head output must be flattened to `[B, F*seq, out_C]` before `unpatchify`.

### 3.5 Dead code removal and batch_size=1 simplification

Three asserts guard the invariants at the top of `_forward_inference`:

```python
assert self.model_type == 't2v'
assert x.shape[0] == 1
assert not torch.is_grad_enabled()
```

**Removed dead code** (unreachable given the asserts above):

| Removed code | Reason |
|---|---|
| `clip_fea` and `y` parameters | Only used by `i2v` model type; caller never passes them |
| `if self.model_type == 'i2v'` branch | Model is always `t2v` (from config.json) |
| `if y is not None: x = [torch.cat(...)]` | `y` was only passed for `i2v` |
| `if clip_fea is not None: context_clip = ...` | `clip_fea` was only passed for `i2v` |
| Gradient checkpointing branch + `create_custom_forward` | Grad is always disabled during inference |

**Batch_size=1 simplifications** — list comprehension / loop-over-batch patterns
replaced with direct tensor operations:

| Original pattern | Replacement |
|---|---|
| `[self.patch_embedding(u.unsqueeze(0)) for u in x]` → `torch.cat(x)` | `self.patch_embedding(x)` directly |
| `[u.flatten(2).transpose(1, 2) for u in x]` → `torch.cat(x)` | `x.flatten(2).transpose(1, 2)` directly |
| `torch.tensor([u.size(1) for u in x])` | `torch.tensor([x.size(1)])` |
| `torch.stack([torch.cat([u, pad]) for u in context])` → `self.text_embedding(...)` | `assert context.size(1) == self.text_len` + `self.text_embedding(context)` (tokenizer always pads to `text_len=512`) |
| `for u in x` loop in `unpatchify` accumulating list → return list | `x[0].view(...)` directly → return single tensor |
| `for i in range(x.shape[0])` loop in `causal_rope_apply` → `torch.stack(output)` | `x[0]` directly → `.unsqueeze(0)` |
| `torch.stack(x)` return in `_forward_inference` | `self.unpatchify(x, grid_sizes).unsqueeze(0)` |

Also removed the redundant `[:total]` slice in `unpatchify` — `total = f*h*w` always
equals `x.shape[1]` since `grid_sizes` is derived from the same `patch_embedding` output.

### 3.6 Copy lengths summary

| Operation | Copy length | Type |
|---|---|---|
| Eviction left-shift | `evict_rolled` = 28,080 | Static (`kv_cache_size - 2 × sink_tokens`) |
| Cache-update read | `cache_len` ≤ 32,760 | Dynamic |
| Working cache assembly | `wc_len` | Dynamic |
| Current tokens assembly | `valid_tokens` | Dynamic |

Dynamic copies keep all reads within the KV cache logical size (37,440), so no
over-allocation is needed. The shared buffer (32,768) is sized to `max_attention_size`
(32,760) aligned up to 8,192.

---

## 4. Attention: `attention_opt.py`

### 4.1 `flash_attn_varlen_b1` — batch_size=1, static shapes

The function `flash_attn_varlen_b1` assumes `batch_size=1` (asserted at entry) and passes
full padded `[1, L, N, D]` tensors directly to `flash_attn_varlen_func`. No packing,
no unpacking, no output padding loops. Input and output have identical static shapes.

- Squeeze: `[1, L, N, D]` → `[L, N, D]` (just a view, no copy)
- Call `flash_attn_varlen_func` with cu_seqlens
- Unsqueeze: `[L, N, D]` → `[1, L, N, D]`

### 4.2 K-side masking via fake batch=2

When `valid_k < Lk`, garbage K tokens must not participate in attention for valid Q tokens.
This is achieved by pretending `batch_size=2` in the cu_seqlens:

```python
cu_seqlens_q = [0, valid_q, Lq]   # seq 0: valid Q, seq 1: garbage Q
cu_seqlens_k = [0, valid_k, Lk]   # seq 0: valid K, seq 1: garbage K
```

Sequence 0 (valid Q → valid K) produces **bitwise-identical** results to the old packed
approach since `flash_attn_varlen_func` processes each sequence independently.
Sequence 1 (garbage Q → garbage K) wastes computation but is harmless.

When all K tokens are valid (cross-attention), a simple `batch_size=1` call is used:

```python
cu_seqlens_q = [0, Lq]
cu_seqlens_k = [0, Lk]
```

All Q tokens (valid + padding) attend to all K tokens. Valid Q tokens get correct output;
padding Q tokens get non-zero garbage (harmless — never contaminates valid tokens since
all operations are per-token, and `unpatchify` slices only valid tokens).

### 4.3 Parameters

`valid_q` and `valid_k` are Python ints passed directly from the caller — no GPU tensor
construction, no `.item()` calls, no GPU-CPU syncs. The only GPU tensors created are the
tiny cu_seqlens (2–3 element int32 tensors).

---

## 5. Cross-attention: `model_opt.py`

`WanT2VCrossAttention` calls `flash_attn_varlen_b1(q, k, v)` with no masking arguments.
All 512 text K tokens are always valid (`context_lens` is `None` in `_forward_inference`),
so the batch=1 path is used. Padding Q tokens attend to the full text context and produce
irrelevant output that is discarded downstream.

All other classes (`WanRMSNorm`, `WanLayerNorm`, `rope_params`, etc.) are imported from the
original `model.py` unchanged.

---

## Shape Flow Summary

```
Pipeline (rolling_forcing_inference_opt.py):
  padded_input:    [B, 15, C, H, W]     (always static)
  padded_timestep: [B, 15]              (always static)
      |
      v
Wrapper (wan_wrapper_opt.py):
  model input:     [B, C, 15, H, W]     (permuted, static)
  model output:    [B, C, 15, H, W]     (static, padding = garbage)
  pred_x0:         [B, 15, C, H, W]     (static, padding = garbage)
      |
      v
Self-attention (causal_model_opt.py) → flash_attn_varlen_b1:
  Q:               [1, 23400, N, D]      (static, 15 frames × 1560 tokens)
  K (buffer_k):    [1, 32768, N, D]      (static, shared across 30 layers)
  V (buffer_v):    [1, 32768, N, D]      (static, shared across 30 layers)
  valid_q, valid_k: Python ints          (drive fake-batch-2 cu_seqlens split)
      |
      v
Pipeline output copy:
  output[:, start:start+max_frames].copy_(denoised_pred)  (static max_frames=15 copy)
  output trimmed to [:, :num_output_frames] before VAE decode

KV cache allocation:
  kv_cache["k/v"]:  [B, 37440, N, D]    (= logical size, no padding needed)
  shared_buffer_k:   [B, 32768, N, D]    (1 pair shared by all 30 layers)
  shared_buffer_v:   [B, 32768, N, D]
```

---

## Key Constants

| Constant | Value | Derivation |
|---|---|---|
| `frame_seqlen` | 1,560 | `60 × 26` (spatial tokens per frame) |
| `block_length` | 4,680 | `3 × 1560` (3 frames per block) |
| `max_attention_size` | 32,760 | `21 × 1560` (full attention window) |
| `kv_cache_alloc_size` | 37,440 | `24 × 1560` (= logical eviction boundary) |
| `max_buffer_size` | 32,768 | `max_attention_size` aligned up to 8,192 |
| `evict_rolled` | 28,080 | `37440 - 2×4680` (cache − 2×sink) |

---

## 6. `causal_rope_apply` rewrite

### 6.1 Complex → real arithmetic

`rope_params` returns a single complex tensor `freqs`. The old `causal_rope_apply` used
`view_as_complex`, complex multiplication (`x_0 * freqs_i`), and `view_as_real` — ops not
available on all accelerators.

**Optimized:** Split `self.freqs` (complex) into `self.freqs_cos` (real) and `self.freqs_sin`
(real), extracted via `.real.clone()` / `.imag.clone()` from the original complex computation
(guarantees bitwise match). `causal_rope_apply` now takes `(freqs_cos, freqs_sin)` and uses
explicit real arithmetic:

```python
out_re = x_re * cos - x_im * sin
out_im = x_re * sin + x_im * cos
```

All call sites (5 total) and both function signatures (`CausalWanSelfAttention.forward`,
`CausalWanAttentionBlock.forward`) updated from `freqs` → `freqs_cos, freqs_sin`.

### 6.2 Neuron-friendly ops

Additional changes to avoid ops unsupported by the Neuron JIT tracer:

| Original op | Problem | Replacement |
|---|---|---|
| `freqs.split(sizes, dim=1)` | Tuple return breaks tracer | Explicit slicing: `freqs_cos[:, :s0]`, `[:, s0:s0+s1]`, `[:, s0+s1:]` |
| `x[0, :seq_len]` (select) | Select op unsupported | `x[:, :seq_len]` (slice) |
| `x_pairs[..., 0]` (select last dim) | Select op unsupported | `x_pairs[:,:,:,:, 0:1].reshape(...)` |
| `torch.stack([re, im], dim=-1)` | Stack unsupported | `unsqueeze(-1)` + `torch.cat(dim=-1)` |
| `torch.cat([x_0, x[0, seq_len:]])` | Dead padding code | Removed (output is `[1, seq_len, N, D]`, no padding) |

### 6.3 Tensor `start_frame` for IR reuse

`start_frame` was changed from a Python int to a **scalar tensor** (shape `[]`). When
`start_frame` is an int, different values produce different compiled IRs (and thus separate
NEFFs). Using a tensor keeps the value out of the IR via `torch.index_select`:

```python
frame_idx = start_frame + torch.arange(f, device=start_frame.device)
torch.index_select(freqs_cos[:, :s0], 0, frame_idx)  # replaces freqs_cos[start_frame:start_frame+f]
```

This reduces the number of NEFFs from one per unique `(grid_sizes, start_frame)` pair to
one per unique `grid_sizes`. The trade-off is longer compilation time and larger NEFF size
due to the extra `index_select` indirection.

All 5 call sites updated to pass `torch.tensor(start_frame_int, device=...)`.

The upcast was also changed from `torch.float64` to `torch.float32` (Neuron does not support
float64). On GPU, `freqs_cos`/`freqs_sin` remain float64 (from complex decomposition), so
`float32 × float64` broadcasts to float64 — computation precision is unchanged.

---

## 7. Explicit B=1 indexing in `CausalWanSelfAttention.forward`

Batch size is always 1 in the rolling forcing pipeline. All `.copy_()` and slice assignment
ops in `forward()` now use `[0, ...]` (select) instead of `[:, ...]` (slice) for the batch
dimension:

```python
# before
buffer_k[:, :evict_rolled].copy_(kv_cache["k"][:, src_start:src_start + evict_rolled])
# after
buffer_k[0, :evict_rolled].copy_(kv_cache["k"][0, src_start:src_start + evict_rolled])
```

An `assert b == 1` guard is added at the start of `forward()`.

`causal_rope_apply` is unchanged — it accepts and returns 4D `[B, L, N, D]` tensors.
Where its output (`anchor_roped`) is used as a `.copy_()` source, `anchor_roped[0]` selects
the single batch element to match the 3D LHS.

---

## Correctness Verification

Both versions run with identical inputs (same seed, same prompts, same config). Output latent
tensors are compared with `torch.equal()` — **all 126 frames match bitwise**.
Baseline latents are saved to `output_baseline.pt` for future fast comparisons.

`causal_rope_apply` was rewritten from complex to real arithmetic. Output verified with
`torch.equal()` — **bitwise-identical** to baseline across all 126 frames (46 windows).

---

## 8. `unpatchify`: `einsum` → `permute`

In `causal_model_opt.py`, replaced `torch.einsum('fhwpqrc->cfphqwr', u)` with
`u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()` — the einsum is a pure 7D permutation
with no contraction. Also changed `x[0]` to `x.squeeze(0)`.

```python
# before
u = x[0].view(f, h, w, *self.patch_size, c)
u = torch.einsum('fhwpqrc->cfphqwr', u)

# after
u = x.squeeze(0).view(f, h, w, *self.patch_size, c)
u = u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
```

**Bitwise identical** (max diff 0.0) against `output_baseline.pt` over the full 46-window
pipeline run.

---

## 9. `_convert_flow_pred_to_x0`: precompute sigma, remove `argmin`

`_convert_flow_pred_to_x0` in `wan_wrapper_opt.py` performed a `torch.argmin` over a
`[B*F, 1000]` tensor every forward call to map timestep floats back to sigma values.
Since the timestep values originate from the scheduler's own table (fixed at init), the
mapping is deterministic and can be precomputed.

**Pipeline (`rolling_forcing_inference_opt.py`):**
- Added `_timestep_to_sigma()`: maps a single timestep value to its sigma via the
  scheduler's paired `timesteps`/`sigmas` tables (argmin done once at init).
- Added `_build_sigma_patterns()`: precomputes a `[2*nds-1, max_frames]` sigma pattern
  tensor mirroring `timestep_patterns`.
- Added `context_sigma`: precomputed sigma for `context_noise=0` (cache-update calls).
- `padded_sigma` is filled alongside `padded_timestep` and passed to `self.generator()`.

**Wrapper (`wan_wrapper_opt.py`):**
- `_convert_flow_pred_to_x0` now takes `sigma_t` directly instead of `timestep`.
  The argmin lookup is gone; it simply does `x0 = xt - sigma_t * flow_pred` in float64.
- `forward()` accepts a `sigma` parameter and passes it through.

```python
# before (per-step, in _convert_flow_pred_to_x0)
timestep_id = torch.argmin(
    (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)

# after (at init, in pipeline)
sigma_patterns = _build_sigma_patterns()  # precomputed once
# at runtime, just index and pass through:
padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]
```

**Bitwise identical** (max diff 0.0) against `output_baseline.pt` over the full 46-window
pipeline run.

---

## 10. `add_noise`: precompute sigma, remove `argmin`

`scheduler.add_noise()` in the re-noising loop performed the same `torch.argmin` over
`[B*nfpb, 1000]` to map timestep floats to sigma values. Since the timestep values come
from `self.denoising_step_list` (fixed at init), sigma can be precomputed.

Added `block_sigma_list` alongside `block_t_list` at init, and a standalone `add_noise()`
function that takes precomputed sigma directly:

```python
# before
block_t = block_t_list[step_index + 1]
self.scheduler.add_noise(block_pred, block_noise, block_t)
# internally: argmin over 1000 timesteps → sigma → (1-sigma)*clean + sigma*noise

# after
block_sigma = block_sigma_list[step_index + 1]
add_noise(block_pred, block_noise, block_sigma)
# directly: (1-sigma)*clean + sigma*noise
```

**Bitwise identical** (max diff 0.0) against `output_baseline.pt` over the full 46-window
pipeline run.
