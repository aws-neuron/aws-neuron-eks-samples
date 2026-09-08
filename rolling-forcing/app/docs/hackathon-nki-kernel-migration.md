# Hackathon Proposal: NKI Kernel Migration Skill — From `kernel_builder` to Standard NKI

## Problem Statement

AWS Neuron's [NKI (Neuron Kernel Interface)](https://github.com/aws-neuron/nkipy) allows developers to write custom high-performance kernels for Trainium/Inferentia hardware. However, NKI has two distinct API surfaces:

1. **`nki.compiler.kernel_builder`** — A lower-level, ISA-oriented API used in early NKI development and the standalone `nki` pip package. Uses destination-style ops (`nisa.tensor_tensor_arith(dst=, lhs=, rhs=, op=)`), explicit buffer management (`num_buffers=`), layout transforms (`.rearrange()`, `.repeat()`), and custom loop constructs (`nb.fori_loop`, `nb.range`).

2. **`neuronxcc.nki`** (standard NKI) — The stable, portable API bundled with the Neuron compiler (`neuronxcc`). Uses return-style ops (`result = nisa.tensor_tensor(a, b, op)`), simpler buffer semantics (`buffer=nl.sbuf`), and Python-native control flow.

Kernels written for `kernel_builder` **do not work** with the bundled compiler in production Neuron containers. The import paths differ, the function signatures differ, the buffer management model differs, and several constructs (`.rearrange()`, `nb.fori_loop`, `nisa.arith_op.*` enums) simply don't exist in standard NKI. Today, converting between these APIs is a manual, error-prone process that requires deep understanding of both surfaces and Neuron's hardware memory model (SBUF, PSUM, HBM, tile partitioning).

## What We're Building

An **AI-assisted migration skill** — a structured knowledge base and toolchain that enables an LLM (or a human developer) to systematically convert NKI kernels from `kernel_builder` to standard `neuronxcc.nki`. Think of it as a translation guide with executable validation.

### Core Deliverables

1. **API Mapping Reference** — A comprehensive, verified mapping between the two API surfaces:

   | `kernel_builder` | Standard `neuronxcc.nki` | Notes |
   |---|---|---|
   | `nb.ndarray(shape, dtype, num_buffers=2)` | `nl.ndarray(shape, dtype, buffer=nl.sbuf)` | No `num_buffers` concept |
   | `nisa.tensor_tensor_arith(dst=, lhs=, rhs=, op=nisa.arith_op.Add)` | `result = nisa.tensor_tensor(a, b, nl.add)` | Return-style; `nl.*` ops |
   | `nisa.memset(dst=buf, value=0)` | `buf = nisa.memset(shape, value, dtype)` | Returns new tile |
   | `nisa.dma_copy(dst=, src=)` (positional) | `nisa.dma_copy(dst=, src=)` (keyword-only) | Same semantics, different syntax |
   | `nisa.matmul(dst=, stationary=, moving=, accum=)` | `result = nisa.nc_matmul(stationary, moving)` | `stationary.T @ moving`; always returns PSUM |
   | `.rearrange("p n (c two) -> p n c two", two=2)` | Manual slicing with `nl.ds()` | No einops-style transforms |
   | `.repeat("p x -> p w x", w=W)` | Manual tiling loops | No broadcast repeat |
   | `nb.fori_loop(bound, body)` | Python `for` loop | Standard control flow |
   | `nb.range(N)` | `range(N)` | — |
   | `nisa.activation_function.exp` | `nl.exp` | Different enum namespace |
   | `nisa.arith_op.Multiply` | `nl.multiply` | — |
   | `nisa.tensor_scalar_cache_reduce(...)` | Separate `nisa.tensor_scalar()` + `nisa.tensor_reduce()` | Fused op doesn't exist in standard |
   | `nisa.scalar_tensor_tensor_arith(...)` | Sequence of `nisa.tensor_scalar()` + `nisa.tensor_tensor()` | Fused op doesn't exist in standard |
   | `nisa.activation(dst=, src=, bias=, scale=, op=)` | `result = nisa.activation(op, src, bias=, scale=)` | Return-style; positional `op` |
   | `nisa.load_register(src)` | Not available — use `nl.load()` or indirect indexing | Architecture-specific |
   | `nb.compiler.perfetto_group(...)` | Not available | Profiling annotation only |
   | `get_target_info().name == "trn1"` | Not available — assume single target | |

2. **Conversion Rules & Gotchas** — Documented patterns for the non-obvious transformations:
   - **PSUM accumulation**: `kernel_builder` allows writing to PSUM directly; standard NKI PSUM is write-only by `nc_matmul`. Accumulation must happen in SBUF.
   - **NKI scoping**: Reassigning a tile variable inside a loop shadows it. Use `tile[...] = expr` for in-place updates when the tile is needed after the loop.
   - **Variable-size slices**: NKI requires compile-time-constant slice sizes. Pad inputs at the PyTorch call site; assert divisibility inside the kernel.
   - **`nc_matmul` semantics**: Computes `stationary.T @ moving`. When migrating from `nisa.matmul(dst, stationary, moving)`, the transpose is implicit — do NOT flip operands.
   - **Keyword-only `dma_copy`**: Unlike all other standard NKI ops (which return tiles), `dma_copy` takes `dst=` and `src=` as keyword-only args and returns `None`.
   - **`wrap_nki()` integration**: Standard NKI kernels decorated with `@nki.jit` must be wrapped with `torch_neuronx.nki_hop.wrap_nki()` to be callable from PyTorch eager mode on Neuron devices.

3. **Validation Harness** — A test framework that:
   - Runs both `kernel_builder` and standard NKI versions against a CPU reference
   - Compares outputs within bf16 tolerance
   - Reports which ops traced successfully vs. failed
   - Provides a `hasattr` signature checker to verify available ops in the target `neuronxcc` build

4. **Example Conversions** — Real-world kernel migrations at increasing complexity:
   - **Level 1**: KV Cache Copy (DMA only, ~50 lines)
   - **Level 2**: Cross-attention with softmax (ISA compute + DMA, ~120 lines)
   - **Level 3**: RoPE with layout transforms (`.rearrange`/`.repeat` elimination, ~200 lines)
   - **Level 4**: Multi-section flash self-attention with online softmax (full pipeline, ~400 lines)

## Why This Matters

- **Real production blocker**: Kernel incompatibility between `nkipy` and bundled `neuronxcc.nki` is the #1 reason NKI kernels fail in production Neuron containers today. Developers write kernels using the standalone package's tutorials, then discover they don't work at deploy time.
- **No existing tooling**: There is no automated converter, no migration guide, and no systematic mapping between the two APIs. Developers currently debug this through trial-and-error against opaque compiler errors.
- **AI-assistable**: The conversion is largely mechanical (API mapping) with a few algorithmic hot spots (PSUM→SBUF accumulation, layout transform elimination). This is ideal for an AI skill — the pattern is learnable, the rules are finite, and the validation is objective (does it compile? do the numbers match?).

## Target Audience

- Neuron SDK users writing custom NKI kernels
- Teams migrating from `nkipy` standalone to production Neuron containers
- AI coding assistants (Amazon Q, Cline, etc.) that need to help developers debug NKI kernel failures

## Success Criteria

1. An AI assistant equipped with this skill can convert a `kernel_builder` kernel to standard NKI in a single conversation, with ≤2 compile-fix iterations
2. All 4 example kernels pass the CPU-reference numerical validation
3. The API mapping covers ≥95% of ops in the `nkipy` repository's example kernels

## Known Limitations Discovered During Migration

### In-place / output-parameter kernels cannot be ported

**Standard `neuronxcc.nki` treats all kernel input parameters as immutable.** You cannot `dma_copy` into an input tensor — the tracer raises `TypeError: Cannot update immutable parameter`.

This means **any kernel whose purpose is to write into caller-provided buffers** (like `kv_cache_copy`, which copies K/V cache tensors in-place) **cannot work** in standard NKI. The `kernel_builder` API allowed this pattern; standard NKI does not.

**Workaround options:**
1. Allocate output inside the kernel (`nl.ndarray(..., buffer=nl.shared_hbm)`) and return it, then `copy_()` from the returned tensor. But this doubles the DMA — worse than just calling `tensor.copy_()` directly.
2. Use `tensor.copy_()` (PyTorch) which already uses optimal DMA on Neuron hardware.
3. Wait for standard NKI to support mutable output parameters (if ever).

**Impact:** Only affects DMA-only kernels (copy, scatter, gather). Compute kernels (attention, RoPE, etc.) naturally return new tensors, so this limitation doesn't affect them.

**Error signature:**
```
TypeError: Cannot update immutable parameter `k_dst`.
Info on how to fix: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.errors.html#err-cannot-update-immutable-parameter
```

### RoPE rotation kernel: RESOLVED ✅ — `affine_range` → `sequential_range` fix

**Status:** Kernel compiles, loads, runs on Neuron, and produces **correct output** at all production shapes (max abs diff = 0.000000 vs PyTorch reference).

**Root cause: `nl.affine_range` corrupts SBUF at >8 tile iterations.**

The outer seq_len tile loop originally used `nl.affine_range(num_tiles)`, which enables software pipelining — the compiler overlaps load/compute/store across iterations. When `num_tiles > 8`, the pipeline depth exceeds hardware capacity, and SBUF buffers from iteration N get overwritten by iteration N+k before their stores complete. The fix: change to `nl.sequential_range(num_tiles)`.

**Timeline:**
1. NKI `causal_rope_rotation` kernel ported and validated on CPU. Max abs diff: 0.000000. ✅
2. First deployment: wired via `_nki_rope_apply` → **severe quality regression**.
3. Immediately reverted to PyTorch fallback.
4. Initial investigation: suspected convention mismatch (swap-pair vs complex rotation, sign pattern, cos/sin expansion, CPU vs device tensor construction). All theories **wrong** — diagnostics proved kernel + cos_sin construction both correct for small inputs.
5. **Breakthrough:** Systematic tile-count sweep revealed the real pattern:
   - ≤8 tiles (seq_len ≤ 1024): diff=0.000000 ✅
   - ≥9 tiles (seq_len ≥ 1152): diff=~22 ❌
   - Production shapes (858→7 tiles ✅, 2574→21 tiles ❌, 4290→34 tiles ❌)
6. **Root cause:** `nl.affine_range` in the outer HBM-load/store loop. Software pipelining overlaps too many iterations when trip count > 8, corrupting SBUF.
7. **Fix:** One-word change: `affine_range` → `sequential_range` in outer loop. Inner head loop (N=12, no HBM IO) safely stays `affine_range`.
8. **Verified:** All production shapes pass at diff=0.000000 with `sequential_range`.

**Architecture (hybrid approach):**
- **Grid-building** (cos/sin expansion from freqs tables) done in **PyTorch** — uses `index_select`, `expand`, `repeat_interleave`, sign pattern
- **Rotation** (`x*cos + swap(x)*sin`) done in **NKI kernel** — the compute-heavy part
- `build_rope_grids` NKI kernel port is a future optimization (not needed for correctness)

**Key lessons for the migration skill:**

1. **`affine_range` vs `sequential_range` is critical for correctness, not just performance.** Use `sequential_range` for any outer loop that does HBM loads/stores. Only use `affine_range` for inner loops operating entirely within SBUF.

2. **Small-input tests are insufficient.** A kernel can pass all tests at small sizes and fail catastrophically at production sizes. Always test at the exact shapes the model uses, including padded sizes.

3. **Systematic bisection > theory-driven debugging.** Instead of guessing which convention is wrong, sweep one variable (tile count) while holding everything else constant. The tile-count sweep immediately pinpointed the threshold at 8→9 tiles, which pointed directly to `affine_range` pipelining.

4. **Diagnostic methodology for NKI kernel bugs:**
   ```
   Step 1: Verify kernel + inputs match PyTorch reference (small aligned shape)
   Step 2: Test with exact production shapes (may require padding)
   Step 3: If Step 1 passes but Step 2 fails, sweep tile count to find threshold
   Step 4: Threshold at 8 tiles → affine_range bug. Fix: sequential_range.
   Step 5: Verify fix at ALL production shapes before deploying
   ```

### HBM-to-HBM DMA may not be supported

Standard NKI `dma_copy` may only support HBM↔SBUF transfers, not HBM→HBM. All working examples use HBM→SBUF (load) or SBUF→HBM (store). A kernel that attempts direct HBM-to-HBM copy without going through SBUF may fail at trace time. This needs verification.

## Team Size & Timeline

- 2-3 engineers
- 2-day hackathon sprint
- Day 1: API mapping + validation harness + Level 1-2 conversions
- Day 2: Level 3-4 conversions + skill packaging + demo
