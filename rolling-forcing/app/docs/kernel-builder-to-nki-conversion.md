# Kernel Builder → Bundled neuronxcc.nki Conversion Guide

**Purpose:** Claude skill document for converting NKI kernels from the `nki.compiler.kernel_builder` API to the bundled `neuronxcc.nki` API. Based on hard-won lessons from porting 4 kernels in the rolling forcing video streaming pipeline.

**Date:** April 2026  
**Kernels ported:** cross_attention, self_attention, rope (causal_rope_rotation), kv_cache_copy

---

## Table of Contents

1. [Background: Why Two APIs?](#1-background-why-two-apis)
2. [API Translation Table](#2-api-translation-table)
3. [Fundamental Tracer Constraints](#3-fundamental-tracer-constraints)
4. [Conversion Patterns (Recipes)](#4-conversion-patterns-recipes)
5. [Worked Example: self_attention](#5-worked-example-self_attention)
6. [Kernel Catalogue](#6-kernel-catalogue)
7. [How Kernels Are Invoked](#7-how-kernels-are-invoked)
8. [Diagnostic & Testing Methodology](#8-diagnostic--testing-methodology)

---

## 1. Background: Why Two APIs?

### kernel_builder API (`nki.compiler.kernel_builder`)

The **original** NKI API used during initial kernel development. It's a standalone pip package (`nki`) with its own namespace:

```python
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor
from nki.compiler.kernel_builder import isa as nisa
```

Key characteristics:
- **Destination-style operations**: `nisa.tensor_copy(dst=out, src=inp)`
- **Named operations**: every op takes a `name="..."` parameter for profiling
- **Engine selection**: DMA ops take `engine=nisa.engine.Gpsimd` or `nisa.engine.Sync`
- **`nb.fori_loop()`** for dynamic batch loops
- **`nb.ndarray_like()`** to clone buffer shapes
- **`nb.ts()`** timestep addressing in addition to `nb.ds()`
- **`.rearrange()` / `.repeat()`** zero-copy einops-style view transforms
- **`nb.compiler.perfetto_group()`** for profiling instrumentation
- **Branching on loop variables is legal** — `if section_i == 0:` works

### Bundled neuronxcc.nki API

The **current** API bundled with the Neuron SDK. No standalone pip package needed:

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
```

Key characteristics:
- **Return-style operations**: `result = nisa.tensor_copy(src)` — returns a new tile
- **No `name` parameter** on operations
- **No engine selection** on DMA
- **No `fori_loop`** — use `nl.sequential_range()` or `nl.affine_range()`
- **No `ndarray_like`** — use explicit `nl.ndarray()` with same shape
- **No `.rearrange()` / `.repeat()`** — use manual slicing (`[:, 0::2]`)
- **No `perfetto_group()`**
- **LoopVar constraints** — loop variables from `nl.sequential_range()` are symbolic; cannot index Python lists or branch on them
- **Strict scope enforcement** — variables defined in `if/else` blocks cannot escape

### PyTorch Integration

Both APIs use the same invocation pattern from PyTorch:

```python
from torch_neuronx.nki_hop import wrap_nki
from kernels.my_kernel import my_kernel

kernel = wrap_nki(my_kernel)  # wraps @nki.jit into PyTorch-callable
output = kernel(input1, input2, scalar_param=value)  # compiles on first call
```

---

## 2. API Translation Table

### Imports

| kernel_builder | bundled nki |
|----------------|-------------|
| `import nki.compiler.kernel_builder as nb` | `import neuronxcc.nki.language as nl` |
| `from nki.compiler.kernel_builder import Tensor` | (no type hints needed) |
| `from nki.compiler.kernel_builder import isa as nisa` | `import neuronxcc.nki.isa as nisa` |
| `import neuronxcc.nki as nki` | `import neuronxcc.nki as nki` (same) |

### Buffer Allocation

| kernel_builder | bundled nki | Notes |
|----------------|-------------|-------|
| `nb.ndarray(shape, dtype, memspace=nb.shared_hbm)` | `nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm)` | HBM output |
| `nb.ndarray(shape, dtype, memspace=nb.hbm)` | `nl.ndarray(shape, dtype=dtype, buffer=nl.hbm)` | Private HBM |
| `nb.ndarray(shape, dtype, memspace=nb.psum)` | `nl.ndarray(shape, dtype=dtype, buffer=nl.psum)` | PSUM buffer |
| `nb.ndarray(shape, dtype, name="x")` | `nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)` | SBUF (default, explicit) |
| `nb.ndarray(shape, dtype, num_buffers=2)` | `nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)` | No `num_buffers` in bundled |
| `nb.ndarray_like(tensor)` | `nl.ndarray(tensor.shape, dtype=tensor.dtype, buffer=nl.sbuf)` | Explicit shape/dtype |
| `nb.ds(start, size)` | `nl.ds(start, size)` | Dynamic slice — identical semantics |
| `nb.ts(index, size)` | `nl.ds(index * size, size)` | No timestep addressing — manual |

### DMA Operations

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.dma_copy(dst=d, src=s, name="x", engine=nisa.engine.Sync)` | `nisa.dma_copy(dst=d, src=s)` |
| `nisa.dma_copy(dst=d, src=s)` | `nisa.dma_copy(dst=d, src=s)` |
| `nl.load(src)` | `nl.load(src)` (same — sugar for dma_copy to sbuf) |
| `nl.store(dst, src)` | `nl.store(dst, src)` (same) |

### Arithmetic — Destination-Style → Return-Style

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.tensor_copy(dst=d, src=s)` | `d = nisa.tensor_copy(s)` or `d = nisa.tensor_copy(s, dtype=dtype)` |
| `nisa.tensor_tensor_arith(dst=d, lhs=a, rhs=b, op=nisa.arith_op.Multiply)` | `d = nisa.tensor_tensor(a, b, nl.multiply)` |
| `nisa.tensor_tensor_arith(dst=d, lhs=a, rhs=b, op=nisa.arith_op.Add)` | `d = nisa.tensor_tensor(a, b, nl.add)` |
| `nisa.tensor_tensor_arith(dst=d, lhs=a, rhs=b, op=nisa.arith_op.Min)` | `d = nisa.tensor_tensor(a, b, nl.minimum)` |
| `nisa.tensor_tensor_arith(dst=d, lhs=a, rhs=b, op=nisa.arith_op.Max)` | `d = nisa.tensor_tensor(a, b, nl.maximum)` |

**Operator name mapping:**
| `nisa.arith_op.*` | `nl.*` |
|-------------------|--------|
| `Multiply` | `multiply` |
| `Add` | `add` |
| `Subtract` | `subtract` |
| `Min` | `minimum` |
| `Max` | `maximum` |

### Scalar Operations

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.activation(dst=d, src=s, scale=val, bias=zero, op=nisa.activation_function.copy)` | `d = nisa.tensor_scalar(s, nl.multiply, val)` |
| `nisa.scalar_tensor_tensor_arith(dst=d, src0=a, src1=b, imm0=c, op0=Multiply, op1=Add)` | Split into two ops: `t = nisa.tensor_tensor(a, c, nl.multiply)` then `d = nisa.tensor_tensor(t, b, nl.add)` |

### Reduction Operations

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.tensor_reduce_arith(dst=d, src=s, op=nisa.arith_op.Max, num_r_dim=1)` | `d = nisa.tensor_reduce(nl.maximum, s, axis=1)` |
| `nisa.tensor_reduce_arith(dst=d, src=s, op=nisa.arith_op.Add, num_r_dim=1)` | `d = nisa.tensor_reduce(nl.add, s, axis=1)` |
| `nisa.tensor_scalar_cache_reduce(dst=d, reduce_res=r, src=s, operand0=scale, op0=Multiply, reduce_op=Max)` | Split: `d = nisa.tensor_scalar(nisa.tensor_copy(s), nl.multiply, scale)` then `r = nisa.tensor_reduce(nl.maximum, d, axis=1)` |

### Activation Functions

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.activation(dst=d, src=s, op=nisa.activation_function.exp, bias=b, scale=1.0)` | `shifted = nisa.tensor_tensor(s, b, nl.add)` then `d = nisa.activation(nl.exp, shifted)` |
| `nisa.activation(dst=d, src=s, op=nisa.activation_function.reciprocal, ...)` | `d = nisa.reciprocal(s)` |
| `nisa.activation(dst=d, src=s, op=nisa.activation_function.copy, scale=s2, bias=zero)` | `d = nisa.tensor_tensor(s, s2, nl.multiply)` |

### Memset

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.memset(dst=d, value=0.0)` | `d = nisa.memset(shape, value=0.0, dtype=dtype)` |
| `nisa.memset(dst=d, value=float('-inf'))` | `d = nisa.memset(shape, value=float('-inf'), dtype=dtype)` |

### Matmul

| kernel_builder | bundled nki |
|----------------|-------------|
| `nisa.matmul(dst=psum, stationary=s, moving=m, accum=False)` | `psum = nisa.nc_matmul(s, m)` |
| `nisa.matmul(dst=psum, stationary=s, moving=m, accum=True)` | No direct equivalent — accumulate in SBUF: `psum = nisa.nc_matmul(s, m)` then `sbuf = nisa.tensor_copy(psum)` then `accum = nisa.tensor_tensor(accum, sbuf, nl.add)` |

### Loop Constructs

| kernel_builder | bundled nki | Notes |
|----------------|-------------|-------|
| `for i in range(N):` | `for i in range(N):` | Python range — unrolled at trace time |
| `nb.range(N)` | `nl.sequential_range(N)` | Sequential execution, LoopVar |
| `nb.fori_loop(bound, body)` | `for i in nl.sequential_range(bound):` | No fori_loop in bundled |

### View Operations

| kernel_builder | bundled nki |
|----------------|-------------|
| `tile.rearrange("p (c two) -> p c two", two=2)` | Manual slicing: `tile[:, 0::2]`, `tile[:, 1::2]` |
| `tile.repeat("p x -> p c x", c=N)` | Per-element loop (if N is small) |
| `tile.view(new_shape)` | Not available — reshape data manually |

---

## 3. Fundamental Tracer Constraints

These are the **hard-won lessons** that caused the most debugging time. Each represents a constraint in the bundled NKI tracer that does NOT exist in kernel_builder.

### 3.1 LoopVar Cannot Index Python Lists

**Constraint:** `nl.sequential_range(N)` produces a `LoopVar` — a symbolic trace variable. You **cannot** use it to index Python lists, dicts, or tuples.

```python
# ❌ FAILS — "list indices must be integers or slices, not LoopVar"
mask_info = [[True, False], [False, True]]
for section_i in nl.sequential_range(num_sections):
    entry = mask_info[section_i]  # CRASH

# ✅ WORKS — use tensor indexing instead
mask = nl.ndarray((128, seq_k), dtype=nl.float32, buffer=nl.sbuf)  # 0 or -inf
for section_i in nl.sequential_range(num_sections):
    nisa.dma_copy(dst=mask_sec, src=mask[:, nl.ds(section_i * SECTION, SECTION)])
```

**Key insight:** Even when `num_sections` is a Python int (e.g., `4`), `range()` inside the kernel with that value produces LoopVars if the range variable is used in NKI operations. Only `range()` at the Python level (outside `nl.sequential_range`) produces real Python ints.

### 3.2 Variables Cannot Escape if/else Scope

**Constraint:** The NKI tracer enforces strict lexical scoping. A variable assigned inside an `if`/`else` block **cannot be used outside that block**.

```python
# ❌ FAILS — "local variable 'correction' is referenced outside of its parent scope"
if section_i == 0:
    correction = nisa.memset((P, 1), value=1.0, dtype=nl.float32)
else:
    correction = nisa.activation(nl.exp, diff)
# Outside the if/else:
scaled = nisa.tensor_tensor(old_sum, correction, nl.multiply)  # CRASH

# ✅ WORKS — compute unconditionally (branchless)
# Initialize r_max = -inf before loop. Then:
correction = nisa.activation(nl.exp, nisa.tensor_tensor(old_max, new_max, nl.subtract))
# For section 0: exp(-inf - sec_max) = exp(-inf) = 0, correctly zeroing empty accumulators
```

### 3.3 LoopVar Cannot Be Used for Branching

**Constraint:** You cannot branch (`if section_i == 0`) on a LoopVar from `nl.sequential_range()`. Kernel_builder allows this because it unrolls with concrete values.

```python
# ❌ FAILS in bundled NKI (LoopVar is symbolic)
for section_i in nl.sequential_range(num_sections):
    if section_i == 0:  # comparison with LoopVar
        ...

# ✅ WORKS — make the logic branchless
# Design algorithms so they work identically for all iterations.
# Use initialization values that make the "first iteration" case a no-op.
# Example: r_max = -inf → correction = exp(-inf - x) = 0 → zeroes out empty accumulators
```

### 3.4 affine_range vs sequential_range (8-Tile Corruption)

**THE most important NKI debugging lesson.**

```python
# ❌ CORRUPTS OUTPUT when num_tiles > 8
for tile_i in nl.affine_range(num_tiles):
    x_sb = nl.load(...)   # HBM → SBUF
    # ... compute ...
    nl.store(...)          # SBUF → HBM

# ✅ CORRECT — always use sequential_range for HBM IO
for tile_i in nl.sequential_range(num_tiles):
    x_sb = nl.load(...)
    # ... compute ...
    nl.store(...)
```

**Why:** `affine_range` enables software pipelining — the compiler overlaps load/compute/store across iterations. When trip count > ~8, pipeline depth exceeds SBUF capacity, and buffers from iteration N get overwritten before stores complete. **Silent data corruption** — no errors, just wrong numbers.

**Diagnosis:** Output is correct for seq_len ≤ 1024 (8 tiles), wrong for seq_len > 1024. The corruption is deterministic.

### 3.5 Input Parameters Are Immutable

**Constraint:** In bundled NKI, you cannot `dma_copy` into input tensor parameters. They are read-only.

```python
# ❌ FAILS — cannot write to input parameter
@nki.jit
def my_kernel(dst, src):
    nisa.dma_copy(dst=dst[...], src=src[...])  # dst is an input — immutable!

# ✅ WORKS — allocate output in HBM and return it
@nki.jit
def my_kernel(src):
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[...], src=src[...])
    return out
```

**Impact:** The `kv_cache_copy` kernel cannot be directly ported because it writes into `dst` parameters. layers.py uses `tensor.copy_()` instead, which is already optimal DMA on Neuron.

### 3.6 No Fused Operations

kernel_builder has fused ops like `tensor_scalar_cache_reduce` (scale + partial reduce in one instruction). Bundled NKI doesn't have these — you must split into separate ops:

```python
# kernel_builder: fused scale + max reduce
nisa.tensor_scalar_cache_reduce(
    dst=scores, reduce_res=partial_max,
    src=psum, operand0=scale, op0=Multiply, reduce_op=Max)

# bundled NKI: separate ops
scores = nisa.tensor_copy(psum)
scores = nisa.tensor_scalar(scores, nl.multiply, scale)
partial_max = nisa.tensor_reduce(nl.maximum, scores, axis=1)
```

---

## 4. Conversion Patterns (Recipes)

### Pattern 1: Branching on section_i → Branchless Online Softmax

**Before (kernel_builder):**
```python
for section_i in range(num_sections):
    if section_i == 0:
        nisa.tensor_copy(dst=running_max[:, nb.ds(grp_i, 1)], src=sec_max)
        nisa.tensor_copy(dst=running_sum[:, nb.ds(grp_i, 1)], src=sec_sum)
        current_pv = pv_section
    else:
        old_max = ...
        scaling_factor = exp(old_max - new_max)
        running_sum = running_sum * scaling_factor + sec_sum
        current_pv = prev_output * scaling_factor + pv_section
    
    if section_i == num_sections - 1:
        reciprocal = 1 / running_sum
        output = current_pv * reciprocal
```

**After (bundled NKI):**
```python
# Initialize BEFORE loop
r_max[:, gi] = memset(value=-inf)   # -inf makes first correction = 0
r_sum[:, gi] = memset(value=0.0)
pv_all[:, gi, :] = memset(value=0.0)

for section_i in nl.sequential_range(num_sections):  # LoopVar — no branching!
    ...
    # ALWAYS compute correction (branchless)
    old_max = tensor_copy(r_max[:, gi])
    new_max = tensor_tensor(old_max, sec_max, maximum)
    correction = activation(exp, tensor_tensor(old_max, new_max, subtract))
    # For section 0: exp(-inf - sec_max) = 0 → zeroes empty accumulators ✓
    
    r_max[:, gi] = new_max
    r_sum[:, gi] = tensor_tensor(tensor_tensor(old_sum, correction, multiply), sec_sum, add)
    pv_all[:, gi, :] = tensor_tensor(tensor_tensor(old_pv, correction, multiply), pv_section, add)

# Normalize AFTER loop (no "last section" check)
for gi in range(num_q_grps):
    rcp = reciprocal(r_sum[:, gi])
    output = tensor_tensor(pv_all[:, gi, :], rcp, multiply)
```

### Pattern 2: Python List Masking → Tensor Mask

**Before (kernel_builder):**
```python
# Compute mask at trace time using Python if/else on concrete section_i
if actual_seqlen_k < seqlen_k:
    section_base = section_i * section_len
    for si in range(tiles_per_section):
        global_start = section_base + si * 512
        if global_start >= actual_seqlen_k:
            nisa.memset(dst=scores[:, si], value=float('-inf'))
        elif global_end > actual_seqlen_k:
            valid = actual_seqlen_k - global_start
            nisa.memset(dst=scores[:, nb.ds(valid, 512-valid)], value=float('-inf'))
```

**After (bundled NKI):**
```python
# Caller builds mask tensor: (128, seq_k) bf16, 0 for valid, -inf for invalid
mask = torch.zeros(128, seq_k, dtype=torch.bfloat16)
if actual_seqlen_k < seq_k:
    mask[:, actual_seqlen_k:] = float('-inf')
mask = mask.to(device)

# Kernel: just load and add
for section_i in nl.sequential_range(num_sections):
    # Load mask section
    nisa.dma_copy(dst=mask_sec, src=mask[:, nl.ds(section_i * SECTION, SECTION)])
    # Add to scores (0 = no effect, -inf = masked)
    masked = nisa.tensor_tensor(scores, mask_sec, nl.add)
```

### Pattern 3: dst-Style → Return-Style

```python
# kernel_builder (dst-style)
result = nb.ndarray((128, 128), nb.float32, name="result")
nisa.tensor_tensor_arith(dst=result, lhs=a, rhs=b, op=nisa.arith_op.Multiply)

# bundled NKI (return-style)
result = nisa.tensor_tensor(a, b, nl.multiply)
```

### Pattern 4: In-Place SBUF Update (Scope Safety)

```python
# ❌ RISKY — variable defined in loop body might have scope issues
for vi in range(num_tiles):
    pv_contrib = nisa.tensor_copy(nisa.nc_matmul(attn_T, v_tile))
    pv_acc = nisa.tensor_tensor(pv_acc, pv_contrib, nl.add)  # pv_acc reassigned

# ✅ SAFE — use [...] indexing for in-place update
pv_acc = nisa.memset((P, d), value=0.0, dtype=nl.float32)
for vi in range(num_tiles):
    pv_contrib = nisa.tensor_copy(nisa.nc_matmul(attn_T, v_tile))
    pv_acc[...] = nisa.tensor_tensor(pv_acc, pv_contrib, nl.add)  # in-place
```

### Pattern 5: nb.fori_loop → nl.sequential_range

```python
# kernel_builder
def body(batch_id):
    # ... kernel body using batch_id ...
nb.fori_loop(batch_size, body)

# bundled NKI
for batch_id in nl.sequential_range(batch_size):
    # ... kernel body using batch_id ...
```

### Pattern 6: Matmul Accumulation

```python
# kernel_builder — hardware accumulation
for pi in range(4):
    nisa.matmul(dst=psum, stationary=q, moving=k[:, pi], accum=(pi > 0))

# bundled NKI — software accumulation in SBUF
acc = nisa.memset((P, d), value=0.0, dtype=nl.float32)
for pi in range(4):
    psum = nisa.nc_matmul(q, k[:, nl.ds(pi * 512, 512)])
    sbuf = nisa.tensor_copy(psum)
    acc[...] = nisa.tensor_tensor(acc, sbuf, nl.add)
```

---

## 5. Worked Example: self_attention

### Original (kernel_builder) — 370 lines

The kernel_builder self-attention kernel used:
- `nb.fori_loop(batch_size, body)` for the batch dimension
- `range(num_sections)` with concrete Python ints for section loop
- `if section_i == 0` / `if section_i > 0` / `if section_i == num_sections - 1` branching
- Python list-based masking computed at trace time
- `nisa.tensor_scalar_cache_reduce()` fused ops
- Intermediate PV results stored to HBM between sections
- `nisa.activation(dst=, src=, bias=, scale=, op=)` with explicit bias tensors
- `nb.ndarray_like()` for temp buffers
- `.repeat()` / `.rearrange()` view transforms (in RoPE, not self-attn)

### Ported (bundled NKI) — 160 lines

Key design decisions:
1. **Mask tensor** — caller builds `(128, seq_k)` bf16 tensor (0 or -inf)
2. **Branchless online softmax** — `r_max = -inf`, `r_sum = 0`, `pv_all = 0` before loop
3. **Always compute correction** — `exp(old_max - new_max)` works for all sections
4. **Normalize after loop** — no "last section" check needed
5. **`nl.sequential_range`** for section loop (LoopVar-safe, no indexing issues)
6. **`range()`** for inner loops (unrolled at trace time, concrete Python ints)

### Conversion Highlights

| Aspect | kernel_builder | bundled NKI |
|--------|---------------|-------------|
| Lines of code | 370 | 160 |
| Section loop | `range(N)` + branching | `nl.sequential_range(N)` branchless |
| Masking | Python if/else at trace time | Tensor (0/-inf) from caller |
| Online softmax | 3 branches per section | 1 branchless path |
| Intermediate storage | HBM (load/store between sections) | SBUF (running state in SBUF) |
| Batch loop | `nb.fori_loop()` | `nl.sequential_range()` |
| Fused ops | `tensor_scalar_cache_reduce` | Separate scale + reduce |

### Validation Results (8/9 shapes pass)

| Shape | max_diff | NKI time | Status |
|-------|----------|----------|--------|
| anchor_block (1 section, masked) | 0.000977 | 119ms | ✅ |
| 2_blocks (2 sections, masked) | 0.000488 | 119ms | ✅ |
| 5_blocks (3 sections, masked) | 0.000244 | 119ms | ✅ |
| full_cache (4 sections, near-full) | 0.000244 | 119ms | ✅ |
| cache_update_full (cached compile) | 0.000244 | 72ms | ✅ |
| 5frame_denoise (large seq_q) | 0.000244 | 120ms | ✅ |
| minimal_1section (no masking) | 0.000488 | 73ms | ✅ |
| exact_2sections (no masking) | 0.000488 | 72ms | ✅ |
| past_section_edge | — | — | OOM (pod memory, not kernel bug) |

---

## 6. Kernel Catalogue

### Ported Kernels (bundled neuronxcc.nki)

| File | Kernel | Purpose | Wired? |
|------|--------|---------|--------|
| `kernels/cross_attention.py` | `wan_cross_attn` | Single-pass flash cross-attention for T5 text context (seq_k=512) | ✅ YES |
| `kernels/rope.py` | `causal_rope_rotation` | RoPE rotation: x*cos + swap(x)*sin | ✅ YES |
| `kernels/self_attention.py` | `wan_flash_self_attn` | Multi-section flash self-attention with online softmax | ❌ Not yet |
| `kernels/kv_cache_copy.py` | `cache_copy`, `kv_cache_copy` | HBM-to-HBM DMA copy for KV cache | ❌ Uses tensor.copy_() instead |

### Original Kernels (kernel_builder — reference only)

| File | Kernel | Status |
|------|--------|--------|
| `kernels/kernel_builder/self_attention.py` | `wan_flash_self_attn` | Reference — replaced by bundled port |
| `kernels/kernel_builder/rope.py` | `causal_rope_rotation`, `build_rope_grids` | Reference — `causal_rope_rotation` ported, `build_rope_grids` not yet |
| `kernels/kernel_builder/kv_cache_copy.py` | `cache_copy`, `kv_cache_copy` | Reference — not portable (immutable inputs) |

---

## 7. How Kernels Are Invoked

### Module-Level Wrapping (in layers.py)

```python
# At import time — wraps @nki.jit into PyTorch-callable
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "true").lower() == "true"

if USE_NKI_KERNELS:
    from torch_neuronx.nki_hop import wrap_nki
    from kernels.cross_attention import wan_cross_attn
    wan_cross_attn = wrap_nki(wan_cross_attn)
```

### Call-Site Pattern (in forward methods)

```python
# Check device + availability, reshape to kernel layout, call, reshape back
if q.device.type == "neuron" and NKI_AVAILABLE:
    q_nki = q[0].permute(1, 2, 0).contiguous()   # [N, D, seq_q]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [N, D, seq_k]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [N, seq_k, D]
    
    # Pad seq_q to multiple of 128
    pad = (128 - seq_q % 128) % 128
    if pad > 0:
        q_nki = torch.nn.functional.pad(q_nki, (0, pad))
    
    # Call kernel (compilation happens automatically on first call)
    x_nki = wan_cross_attn(q_nki, k_nki, v_nki, self.identity,
                           softmax_scale=self.softmax_scale)
    
    # Reshape back: [seq_q_padded, N, D] → [1, seq_q, C]
    x = x_nki[:seq_q].unsqueeze(0).flatten(2)
else:
    # PyTorch SDPA fallback
    ...
```

### Self-Attention Kernel Call-Site (to be wired)

`CausalWanSelfAttention.forward()` Phase 4 already has the stub:

```python
# Existing code in layers.py:
self._nki_available = False  # ← flip to True
self._self_attn_kernel = None  # ← assign wrapped kernel

# In forward():
if q_kern.device.type == "neuron" and self._nki_available:
    x = self._self_attn_kernel(q_kern, k_kern, v_kern, self.identity,
                                softmax_scale=..., actual_seqlen_k=k_len_int, ...)
```

**Changes needed to wire self_attention:**
1. Import and wrap the kernel at module level
2. Assign to `self._self_attn_kernel`
3. Set `self._nki_available = True`
4. Build mask tensor `(128, seq_k)` in forward()
5. Pad `q_kern` to multiple of 128
6. Pass `mask` and `num_sections` instead of `actual_seqlen_k`
7. Truncate output to original seq_q

---

## 8. Diagnostic & Testing Methodology

### The Diagnostic Script Pattern

Create a standalone script that:
1. **Phase 1:** Validates PyTorch SDPA reference at all production shapes (CPU)
2. **Phase 2:** Tests NKI kernel against reference (Neuron device)

```python
# Generate inputs with fixed seed
torch.manual_seed(42)
q = torch.randn(bs, d, seq_q, dtype=torch.bfloat16)
k = torch.randn(bs, d, seq_k, dtype=torch.bfloat16)
v = torch.randn(bs, seq_k, d, dtype=torch.bfloat16)

# Phase 1: CPU reference
out_ref = sdpa_reference(q, k, v, softmax_scale, actual_seqlen_k)

# Phase 2: NKI kernel
q, k, v = q.to("neuron"), k.to("neuron"), v.to("neuron")
# Build mask, pad Q, etc.
out_nki = kernel(q, k, v, identity, mask, softmax_scale=..., num_sections=...)
out_nki_cpu = out_nki[:seq_q].cpu()

# Compare
diff = (out_nki_cpu.float() - out_ref.float()).abs()
max_diff = diff.max().item()
assert max_diff < 2.0, f"FAIL: {max_diff}"
```

### Production Shapes to Test

From `CausalWanSelfAttention` with `frame_length=1560`, `block_length=4680`, `max_attention_size=32760`:

| Name | seq_q | seq_k | actual_k | Description |
|------|-------|-------|----------|-------------|
| anchor_block | 4680 | 8192 | 4680 | First block, partial section |
| 2_blocks | 4680 | 16384 | 9360 | Two blocks |
| 5_blocks | 4680 | 24576 | 23400 | Five blocks |
| full_cache | 4680 | 32768 | 32760 | Full attention window |
| 5frame_denoise | 7800 | 32768 | 32760 | Large query (5 frames) |
| minimal_1section | 4680 | 8192 | 8192 | Exact single section, no masking |
| exact_2sections | 4680 | 16384 | 16384 | Exact two sections, no masking |
| past_section_edge | 4680 | 16384 | 8193 | Just past section boundary |

### Tolerances

- **bf16 attention:** max_diff < 2.0, typical max_diff < 0.001
- **Simple ops (copy, slice):** rtol=0, atol=0
- **RoPE:** max_diff = 0.000000 (verified)
- **Cross-attention:** rtol=1e-2, atol=1e-3

### Deployment Verification

```bash
# On pod: verify correct code is present
grep "mask_sec" /workspace/video-streaming-develop/kernels/self_attention.py
# Clear NEFF cache if kernel source changed
rm -rf /tmp/neff_cache/
```

---

## Appendix: Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: list indices must be integers or slices, not LoopVar` | Indexing Python list with `nl.sequential_range` variable | Use tensor indexing instead |
| `SyntaxError: local variable 'X' is referenced outside of its parent scope` | Variable defined in if/else used after the block | Make logic branchless |
| `exit code 137` | OOM kill (pod ran out of memory) | Not a kernel bug — reduce test count or add cache clearing |
| Output correct for ≤8 tiles, wrong for >8 | `affine_range` SBUF corruption | Change to `sequential_range` |
| `ENOENT: no such file or directory` for `nki.compiler.kernel_builder` | Standalone `nki` package not installed | Use bundled `neuronxcc.nki` instead |
| `Input parameters are immutable` | Writing to kernel input tensor | Allocate output with `nl.shared_hbm` and return |
