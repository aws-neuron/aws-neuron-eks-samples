# neuronx-cc crashes (exit code 70) when torch._dynamo.config.cache_size_limit > 8

## Summary

`torch.compile(backend='neuron')` with `cache_size_limit > 8` causes neuronx-cc to terminate abnormally (exit code 70, `[F139] neuronx-cc terminated abnormally`). This happens because dynamo guards on Python int values inside compiled transformer blocks (KV cache indices, sequence positions), triggering recompilation for each unique value. At the default limit of 8, dynamo falls back to eager after 8 variants — but this limits performance. Any attempt to raise the limit (tested 12, 16, 64) crashes the compiler.

## Impact

Our DiT model produces **438 NEFFs** because dynamo recompiles for each unique value of `current_start`, `kv_cache['global_end_index']`, `kv_cache['local_end_index']`, `updating_cache`, and `num_valid_frames`. With limit=8, it compiles 8 variants then falls back to eager for the rest. We're stuck at **0.44 FPS** (target: 16 FPS). Raising the limit would allow more variants to be compiled (reducing eager fallback overhead), but crashes the compiler.

## Reproduction

```python
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 12  # any value > 8 crashes

# Compile a transformer block that has Python int arguments
# used in conditional logic (KV cache management)
for i, block in enumerate(dit_model.blocks):
    dit_model.blocks[i] = torch.compile(block, backend='neuron', dynamic=False)

# Run inference with varying current_start values per window
# After 8+ unique values of current_start, the 9th+ compilation crashes neuronx-cc
```

## Guard failures that trigger recompilation

```
- current_start == 2574          # layers.py:688 — current_start_frame = current_start // frame_seqlen
- kv_cache['global_end_index'] == 0    # layers.py:706 — num_new_tokens = cache_end - global_end_index
- kv_cache['local_end_index'] == 5148  # layers.py:729 — local_end_index = local_end_index_current + ...
- updating_cache == True               # layers.py:748 — if updating_cache:
- num_valid_frames == 6                # layers.py:697 — if num_valid_frames is not None:
```

## What we need

1. **Fix the neuronx-cc crash** at cache_size_limit > 8 so we can compile more guard variants
2. **Or** provide guidance on how to prevent dynamo from guarding on Python int function arguments that are used in control flow (KV cache index management) inside compiled blocks — without using `@torch.compiler.disable` (which increases NEFF count)

## Environment

- **Image**: `concourse-release-0461d3b`
- **Python**: 3.12.13, **PyTorch**: 2.6, **torch-neuronx**: private (editable `/opt/torch-neuronx/`)
- **Instance**: trn2.48xlarge, TP=4, NEURON_LOGICAL_NC_CONFIG=2
- **Model**: Wan2.1-T2V-1.3B DiT, 30 blocks, dim=1536, 12 heads

## Test results

| cache_size_limit | Pipeline | Result |
|-----------------|----------|--------|
| 8 (default) | Original 15-frame | 0.44 FPS, 438 NEFFs, correct quality |
| 12 | Original 15-frame | neuronx-cc exit code 70 crash |
| 16 | Original 15-frame | neuronx-cc exit code 70 crash |
| 64 | Original 15-frame | neuronx-cc exit code 70 crash |

## Repo

https://github.com/aws-neuron/aws-neuron-eks-samples/tree/rolling-forcing/rolling-forcing

- `app/models/layers.py` — `CausalWanSelfAttention.forward()` (lines 688-748, the guarded code)
- `app/inference_neuron_tp.py` — compilation setup
