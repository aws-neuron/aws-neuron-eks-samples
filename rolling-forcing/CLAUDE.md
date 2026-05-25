# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rolling Forcing video generation on AWS Trainium2 (trn2) via EKS. The core ML pipeline implements Distribution Matching Distillation (DMD) on the Wan2.1-T2V diffusion model, reducing denoising from 50 steps to 5 for near-real-time text-to-video streaming. Runs on NeuronCores with tensor parallelism (TP=4) and custom NKI kernels for attention/RoPE.

## Architecture

The system has two main layers:

**Infrastructure** (`cluster/`, `dra/`, `deploy/`): EKS cluster with trn2.48xlarge nodes, Neuron DRA driver for NeuronCore allocation via ResourceClaimTemplates, and Kubernetes manifests.

**Application** (`app/`): Python inference pipeline running on Neuron:
- `inference_neuron_tp.py` — Main entry point. FastAPI server launched via `torchrun --nproc_per_node=4`. All 4 ranks run DiT (TP-sharded), rank 2 hosts T5, rank 0 hosts VAE.
- `models/layers.py` — All Neuron-optimized diffusion layers (attention blocks, RoPE, norms, FFN, patch embedding). Loads NKI kernels at import time based on `USE_NKI_KERNELS` env var.
- `models/causal_model_tp.py` / `causal_inference_pipeline_tp.py` — TP-sharded model and pipeline orchestration.
- `kernels/` — NKI custom kernels (`cross_attention.py`, `rope.py`, `self_attention.py`, `kv_cache_copy.py`). Uses bundled `neuronxcc.nki` API with `@nki.jit` decorator, integrated via `torch_neuronx.nki_hop.wrap_nki()`.
- `configs/` — YAML configs controlling resolution, frame count, block size. Naming: `rolling_forcing_dmd_f{frames}_b{block_size}[_med].yaml`.

## Running Tests

Tests must run on a Neuron instance. They compile NEFFs on first run (slow), then cache.

```bash
cd app
source .venv/bin/activate

# Kernel tests (run separately — different compilation profile)
python -m pytest tests/wan_kernels -v

# Module tests (exclude heavy attention block test first)
python -m pytest tests/wan_modules -n auto -vs --ignore tests/wan_modules/test_wan_attn_block.py

# Attention block test (reuses cached NEFFs from above)
python -m pytest tests/wan_modules/test_wan_attn_block.py -n auto -vs

# Single test
python -m pytest tests/wan_kernels/test_attention_kernel.py -v
python -m pytest tests/wan_modules/test_wan_ffn.py::test_wan_ffn -v
```

Quick kernel validation (all three kernels, production shapes):
```bash
NEURON_RT_NUM_CORES=4 python test_all_kernels.py
```

## Running the Pipeline

Sequential (single-core steps):
```bash
cd app
./run_pipeline.sh --prompt "A cat on a beach" --output out.mp4
```

TP=4 server mode (production):
```bash
torchrun --nproc_per_node=4 --master_port=29500 inference_neuron_tp.py
```

## Deployment

Pods do `git clone` at startup from GitHub. To deploy changes:
1. Commit and push to the `rolling-forcing` branch
2. Restart the pod (`kubectl rollout restart deployment rf`)
3. Verify code presence with grep on the pod
4. Clear NEFF cache (`/tmp/neff_cache/`) if kernel source changed

Key env vars: `USE_NKI_KERNELS`, `RF_DEVICE_BACKEND=neuron`, `TP_DEGREE=4`, `NEURON_LOGICAL_NC_CONFIG=2`, `USE_NEFF_CACHE`.

## NKI Kernel Development Rules

**Critical**: Always use `nl.sequential_range` for loops with HBM loads/stores. `nl.affine_range` corrupts silently at >8 tile iterations due to SBUF overwrite from software pipelining.

Key constraints:
- Input parameters are immutable — allocate output with `nl.shared_hbm` and return
- Never branch on LoopVar — use branchless algorithms with identity-element initialization
- Never index Python lists with LoopVar — pass data as tensors
- Return-style ops: `result = nisa.tensor_tensor(a, b, nl.multiply)` not dst-style
- Seq_len must be padded to multiple of 128 (tile size) at the call site

Integration pattern:
```python
from torch_neuronx.nki_hop import wrap_nki
from kernels.my_kernel import my_kernel
kernel = wrap_nki(my_kernel)
output = kernel(q, k, v, ...)  # compiles on first call
```

## NKI Compiler Constraints

### DMA with dynamic (runtime) offsets
- A `LoadRegister` result (from `nisa.load_register`) can be used directly in `nb.ds(reg, size)` for DMA offsets.
- Arithmetic expressions on registers (`reg + constant`, `reg1 + reg2`) CANNOT be used as DMA offsets — MLIR pass fails with "failed to find register name for dynamic access".
- Loop induction variables from `nb.fori_loop` / `nb.fori_range_loop` CAN be used with affine constant arithmetic (e.g., `i * 128`).
- **Workaround**: Preload all needed rows in a single indirect DMA using the raw register, then index the preloaded SBUF tile with static offsets.

### Matmul `moving` operand — partition constraint
- The `moving` operand must start at partition 0. A view like `tile[nb.ds(f, 1), :]` where `f > 0` fails BIR verification.
- **Workaround**: Use `nisa.dma_copy` to copy the row to partition 0, then pass to matmul.

### Matmul tile size limits
- `moving` operand free dimension must be ≤ 512. If data exceeds 512, pad to a multiple and loop in chunks of 512.

### `nb.ndarray` deduplication
- The MLIR tracer identifies allocations by source location, not by `name`. An `nb.ndarray(...)` inside a Python loop must produce the same shape every iteration — otherwise raises `ValueError: ndarray() called with different properties`.
- **Fix**: Pad data so every iteration uses uniform chunk size, or manually unroll.

### Avoiding recompilation with runtime values
- `int` hyperparameters are part of the compilation cache key — different values trigger recompilation.
- To pass runtime-varying values without recompilation, make them `Tensor` parameters. Load into SBUF via `nisa.dma_copy`, then into a register via `nisa.load_register`.

### Tile view operations (zero-copy)
- `.rearrange("p (a b) -> p a b", b=128)` — einops-style reshape/permute of free dimensions
- `.repeat("p x -> p x c", c=2)` — broadcast along a new free dimension (zero stride)
- `.view(new_shape)` — reinterpret memory layout (total bytes must match, full views only)
- Partition dimension (axis 0) cannot change in any view operation.

### Optimization guidelines
- Coalesce small DMAs into larger ones — assemble in SBUF first, then one `dma_copy`
- Minimize instruction count — compiler scheduling scales quadratically
- Use `.repeat()` zero-copy broadcast instead of `nisa.gather` with pre-built index

## Model Layer Conventions

- Use slicing (`x[:, :, 0:1]`) not indexing (`x[:, :, 0]`) — Neuron tracing doesn't support `select` ops
- Internal math upcasts to float32 (Neuron has no float64)
- `nn.Conv3d` not supported — use reshape + matmul
- GPU reference models are in `gpu/RollingForcing/wan/modules/*_opt.py` — these are ground truth for shapes/dtypes

## Test Tolerances

- Simple layers: `rtol=5e-3, atol=5e-3`
- Multi-step ops (attention, norms): `rtol=1e-2, atol=1e-2`
- NKI kernels: `rtol=1e-2, atol=1e-3`
- Pure data movement: `rtol=0, atol=0`
- `conftest.py` sets `torch.manual_seed(42)` and `NEURON_FALLBACK_ENABLED=0`
