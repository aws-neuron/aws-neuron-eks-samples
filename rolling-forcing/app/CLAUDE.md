# Video Streaming on AWS Neuron

Diffusion-based video streaming on AWS Neuron (Trainium/Inferentia). The proxy model is **Rolling Forcing** based on **Wan2.1-T2V** (text-to-video). This repo contains Neuron-optimized model layers, NKI attention kernels, GPU reference code, and unit tests.

## Code structure

```
models/
  layers.py          Neuron diffusion model layers: GELU, SiLU, WanFFN, WanRMSNorm,
                     WanLayerNorm, WanPatchEmbed, CausalHead, CausalWanSelfAttention,
                     WanT2VCrossAttention, CausalWanAttentionBlock, unpatchify,
                     convert_flow_pred_to_x0, sinusoidal_embedding_1d, rope_params,
                     causal_rope_apply, modulation helpers

kernels/
  self_attention.py  NKI flash self-attention kernel (bundled neuronxcc.nki API, validated 8/9 shapes)
  cross_attention.py NKI cross-attention kernel (bundled neuronxcc.nki API, single-pass for seqlen_k=512)
  rope.py            NKI RoPE rotation kernel (bundled neuronxcc.nki API)
  kv_cache_copy.py   NKI KV cache copy kernel (ported but not wired — uses tensor.copy_() instead)
  kernel_builder/    Original kernel_builder API versions (reference only, not used at runtime)

docs/
  kernel-builder-to-nki-conversion.md   API tables, tracer constraints, conversion patterns
  nki-kernel-migration-agent-workflow.md 7-stage autonomous agent workflow for kernel migration

gpu/RollingForcing/
  wan/modules/       GPU reference models. Files with `_opt` suffix avoid dynamic shapes
                     and CPU-GPU sync (used as basis for Neuron ports).
  inference.py       GPU inference entry point (_opt.py = optimized variant)
  pipeline/          Rolling forcing inference pipeline
  utils/             Dataset loading, scheduler, WAN wrapper

tests/
  conftest.py        Shared fixtures: torch.manual_seed(42), NEURON_FALLBACK_ENABLED=0
  wan_modules/       Unit tests for Neuron model layers (test against CPU reference)
  wan_kernels/       Unit tests for NKI kernels (compiled via torch_neuronx)
```

## Environment

- Python 3.11, venv at `.venv/`
- Key deps: `torch_neuronx` (editable), NKI compiler (`kernel_builder` syntax), `neuronx-cc`, `pytest`, `pytest-xdist`

## Running tests

Do **not** run `python -m pytest tests -n auto -v` directly — mixing kernel and module tests in one parallel run is very slow. Run them separately:

```bash
source .venv/bin/activate

# 1. Kernel tests (separate run)
python -m pytest tests/wan_kernels -v

# 2. Module tests — first run everything EXCEPT test_wan_attn_block.py
#    This lets all simpler tests build and cache their NEFFs in parallel.
python -m pytest tests/wan_modules -n auto -vs --ignore tests/wan_modules/test_wan_attn_block.py

# 3. Then run test_wan_attn_block.py alone (it reuses cached NEFFs from step 2)
python -m pytest tests/wan_modules/test_wan_attn_block.py -n auto -vs
```

Run a specific test file or test case:
```bash
python -m pytest tests/wan_kernels/test_attention_kernel.py -v
python -m pytest tests/wan_modules/test_wan_ffn.py::test_wan_ffn -v
```

## Development conventions

### General principle

Use the **same shapes, dtypes, and tensor layouts as the GPU reference model** (`gpu/RollingForcing/wan/modules/*_opt.py`) when developing Neuron model layers and kernels. The `_opt` variants are the ground truth — they avoid dynamic shapes and CPU-GPU sync, making them directly portable to Neuron.

### Model layers (`models/layers.py`)

- Modules are wrapped with `@jit` decorator (from `torch_neuronx.jit`) for Neuron tracing. Sub-modules and standalone functions use `jit(...)` inline.
- Internal math (norms, RoPE, flow conversion) upcasts to **float32** since Neuron does not support float64.
- Use **slicing** (`x[:, :, 0:1]`) instead of **indexing** (`x[:, :, 0]`) — Neuron tracing doesn't support tensor `select` ops or `.chunk()` returns.
- `nn.Conv3d` is not supported on Neuron; `WanPatchEmbed` implements it via reshape + matmul.

### NKI kernels (`kernels/`)

**Two API generations exist in this repo:**

- **Bundled `neuronxcc.nki` API** (current, in `kernels/*.py`) — the active kernels used at runtime. Decorated with `@nki.jit`, wrapped with `torch_neuronx.nki_hop.wrap_nki()` for PyTorch integration.
- **`kernel_builder` API** (legacy, in `kernels/kernel_builder/`) — reference only. Uses `nki.compiler.kernel_builder` namespace with `@jit(is_nki_kb=True)`.

**Active kernel status:**

| Kernel | File | Wired into layers.py? |
|--------|------|-----------------------|
| cross_attention | `kernels/cross_attention.py` | ✅ YES |
| rope (rotation) | `kernels/rope.py` | ✅ YES |
| self_attention | `kernels/self_attention.py` | ❌ Not yet (validated, ready to wire) |
| kv_cache_copy | `kernels/kv_cache_copy.py` | ❌ Uses `tensor.copy_()` instead (immutable input constraint) |

**Key differences from kernel_builder:**
- Return-style ops: `result = nisa.tensor_tensor(a, b, nl.multiply)` (not `dst=` style)
- `nl.sequential_range()` replaces `nb.fori_loop()` — **ALWAYS use for HBM IO loops** (`affine_range` corrupts at >8 tiles)
- No branching on LoopVar — use branchless algorithms
- Input parameters are immutable — allocate output with `nl.shared_hbm` and return
- Self-attention uses a tensor mask (0/-inf) passed by caller instead of compile-time masking

**PyTorch integration pattern:**
```python
from torch_neuronx.nki_hop import wrap_nki
from kernels.cross_attention import wan_cross_attn
kernel = wrap_nki(wan_cross_attn)  # wraps @nki.jit into PyTorch-callable
output = kernel(q, k, v, identity, softmax_scale=scale)  # compiles on first call
```

### Testing patterns

- **wan_modules tests**: Create module on CPU, compute reference output, move to `"neuron"`, compare with `torch.testing.assert_close`.
- **wan_kernels tests**: Generate random inputs on CPU, compute expected output with PyTorch, move inputs to `"neuron"`, call the `@jit`-decorated kernel directly (compilation happens automatically), compare result with `torch.allclose`. See `tests/wan_kernels/test_attention_kernel.py` for the canonical pattern:
  ```python
  # 1. Generate inputs and compute reference on CPU
  q, k, v, identity, expected = gen_test_inputs(batch_size, seqlen_q, seqlen_k, d_head, dtype)
  # 2. Move to Neuron — this is what triggers kernel compilation on first call
  q, k, v, identity = q.to("neuron"), k.to("neuron"), v.to("neuron"), identity.to("neuron")
  # 3. Call the @jit-decorated kernel directly
  out = wan_cross_attn(q, k, v, identity, softmax_scale=scale).cpu()
  # 4. Compare
  assert torch.allclose(out, expected.to(out.dtype), rtol=1e-2, atol=1e-3)
  ```
- Typical tolerances: `rtol=5e-3, atol=5e-3` for simple layers; `rtol=1e-2, atol=1e-2` for multi-step ops (attention, norms); `rtol=0, atol=0` for pure data movement (copy/slice).
- Kernel tests use `rtol=1e-2, atol=1e-3`.
- `torch.manual_seed(42)` is set per-test via `conftest.py` fixture.
- `NEURON_FALLBACK_ENABLED=0` ensures tests fail if ops can't run on device.

### Adding new modules

1. Add the layer to `models/layers.py`, wrap with `@jit` or `jit(...)`.
2. Create `tests/wan_modules/test_wan_<name>.py` — test against CPU reference with production shapes.

### Adding new kernels (bundled neuronxcc.nki)

Follow the **7-stage migration workflow** in `docs/nki-kernel-migration-agent-workflow.md`:

1. **Write kernel** in `kernels/<name>.py` using bundled `neuronxcc.nki` API with `@nki.jit` decorator.
2. **Create diagnostic script** `<name>_diag.py` — standalone accuracy test at all production shapes vs PyTorch SDPA.
3. **Validate on Neuron** — run diagnostic, iterate until ALL shapes pass.
4. **Wire into `layers.py`** — `wrap_nki()` at module level, call in forward(), keep SDPA fallback.
5. **Create test** `tests/wan_kernels/test_<name>.py`:
   - Write a pure-PyTorch reference implementation for expected output.
   - Move inputs to `"neuron"` via `.to("neuron")`.
   - Call the `wrap_nki()`-wrapped kernel — compilation is automatic on first call.
   - `.cpu()` the output and compare against expected with `torch.allclose(rtol=1e-2, atol=1e-3)`.

**If migrating from kernel_builder (nkipy):** See `docs/kernel-builder-to-nki-conversion.md` for API translation tables, tracer constraints, and conversion patterns.

### Adding new kernels (legacy kernel_builder — reference only)

The legacy `kernel_builder` API kernels in `kernels/kernel_builder/` use:
- `@jit(is_nki_kb=True)` decorator from `torch_neuronx.jit`
- `nki.compiler.kernel_builder` namespace (`nb.*`, `nisa.*` with dst-style ops)
- Dynamic loops via `nb.fori_loop` with `CompileOptions`

These are kept as reference for understanding the original algorithm. New kernels should use bundled `neuronxcc.nki`.

## NKI compiler knowledge and constraints

### DMA with dynamic (runtime) offsets — indirect DMA

- A `LoadRegister` result (from `nisa.load_register`) can be used directly in `nb.ds(reg, size)` for DMA offsets. The compiler recognizes it as a named register.
- **Arithmetic expressions on registers (`reg + constant`, `reg1 + reg2`) CANNOT be used as DMA offsets.** The MLIR pass fails with "failed to find register name for dynamic access" because `arith.addi`/`arith.muli` results are unnamed intermediates.
- Loop induction variables from `nb.fori_loop` / `nb.fori_range_loop` CAN be used with affine constant arithmetic (e.g., `i * 128`) for DMA offsets — the compiler special-cases `scf.for` induction variables.
- **Workaround for runtime-offset DMA:** preload all needed rows in a single indirect DMA using the raw register (e.g., `freqs[nb.ds(sf_reg, F), :]`), then index the preloaded SBUF tile with static offsets inside a `nb.range` loop.

### Matmul `moving` operand — partition constraint

- The `moving` operand of `nisa.matmul` must start at partition 0. You cannot pass a view like `tile[nb.ds(f, 1), :]` where `f > 0` as the moving operand — BIR verification fails with "Invalid access of 1 partitions starting at partition N".
- **Workaround:** use SBUF-to-SBUF DMA (`nisa.dma_copy`) to copy the needed row from partition `f` to a fresh `(1, free_dim)` tile at partition 0, then pass that to the matmul. DMA can copy across partitions; `tensor_copy` cannot.

### Matmul tile size limits

- The `moving` operand free dimension must be **≤ 512** (error `NCC_IBIR039`: "Matmult moving input tile size must be <= 128x512"). If the data exceeds 512, pad to a multiple of 512 and broadcast in uniform chunks:
  ```python
  MATMUL_FREE_MAX = 512
  total = W * s1                                          # e.g. 1092
  num_chunks = (total + MATMUL_FREE_MAX - 1) // MATMUL_FREE_MAX
  padded = num_chunks * MATMUL_FREE_MAX                   # 1536
  # pad source to [1, padded], then loop:
  for off in range(0, padded, MATMUL_FREE_MAX):
      nisa.matmul(dst=psum, stationary=ones, moving=src[:, nb.ds(off, MATMUL_FREE_MAX)], ...)
      nisa.tensor_copy(dst=dst[:, nb.ds(off, MATMUL_FREE_MAX)], src=psum)
  ```

### NKI tracer `nb.ndarray` deduplication

- The MLIR tracer identifies `nb.ndarray` allocations by **source location** (file:line), not by the `name` parameter. An `nb.ndarray(...)` call inside a Python `for`/`range` loop must produce the **same shape on every iteration** — otherwise the tracer raises `ValueError: ndarray() called with different properties`.
- **Fix:** pad data so every iteration uses a uniform chunk size (see matmul example above), or manually unroll the loop so each allocation is on a distinct source line.

### Avoiding recompilation with runtime values

- NKI kernel `int` hyperparameters are part of the compilation cache key — different values trigger recompilation.
- To pass a value that changes at runtime without recompilation, make it a `Tensor` parameter (e.g., `[1, 1]` int32). Load into SBUF via `nisa.dma_copy`, then into a register via `nisa.load_register`. Use the register for indirect DMA offsets (subject to constraints above).

### Tile view operations — reshape and broadcast

All view operations are zero-copy (no data movement); they create new `TileView` objects with different index expressions. The partition dimension (axis 0) cannot change.

- **`.rearrange(pattern, **dims)`** — einops-style reshape/permute of free dimensions:
  ```python
  tile.rearrange("p (a b) -> p a b", b=128)   # split free dim
  tile.rearrange("p a b -> p (a b)")            # flatten free dims
  tile.rearrange("p a b -> p b a")              # permute free dims
  ```
- **`.repeat(pattern, **dims)`** — broadcast along a new free dimension (zero stride):
  ```python
  # (P, c) -> (P, c, 2): each element duplicated along new trailing axis
  tile.repeat("p x -> p x c", c=2)
  # (P, D) -> (P, N, D): broadcast across num_heads
  tile.repeat("p x -> p c x", c=N)
  ```
  Equivalent to `repeat_interleave` when the new axis is innermost and then flattened. Only works for free dimensions — use matmul broadcast for partition dim.
- **`.view(new_shape, dtype=None)`** — reinterpret memory layout (total bytes must match). Only works on full views (not slices).

**Composing views:** slice, rearrange, and repeat can be chained. Example — copy a `(P, c, 2)` broadcast view into a `(P, D)` slice of a larger buffer:
```python
combined = nb.ndarray((P, 2 * D), nb.float32)
# rearrange the (P, D) slice to (P, c, 2) so shapes match the broadcast view
nisa.tensor_copy(
    dst=combined[:, nb.ds(0, D)].rearrange("p (x c) -> p x c", c=2),
    src=tile.repeat("p x -> p x c", c=2),
)
```

### Kernel optimization guidelines

- **Coalesce small DMAs into larger ones.** Multiple narrow DMA copies to adjacent HBM regions should be merged into a single wide transfer. Assemble the data in one SBUF buffer first, then issue one `dma_copy`. Each DMA has fixed overhead; fewer, larger transfers improve throughput.
- **Minimize instruction count to reduce compile time.** The NKI compiler's scheduling pass scales quadratically with instruction count.
- **Zero-copy with broadcasting** Replace per-element copy operations (e.g., `nisa.gather` with a pre-built index) with zero-copy views (e.g., `.repeat()`) wherever possible — this eliminates instructions entirely rather than just reducing latency.
