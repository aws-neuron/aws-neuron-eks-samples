# NKI Kernel Migration Agent Workflow

**Purpose:** Autonomous agent workflow for migrating NKI kernels from `kernel_builder` (nkipy) to bundled `neuronxcc.nki`. Derived from real iteration experience porting self_attention, cross_attention, rope, and kv_cache_copy kernels.

**Companion document:** `docs/kernel-builder-to-nki-conversion.md` — API tables, tracer constraints, and conversion patterns.

---

## Overview: 7-Stage Pipeline

```
Stage 1: Extract & Catalog       → Understand the source kernel
Stage 2: Pattern Detection        → Identify structural hazards  
Stage 3: Transpile                → Mechanical + algorithmic conversion
Stage 4: Generate Diagnostic      → Standalone accuracy test script
Stage 5: Iterate Until Accuracy   → Fix-compile-test loop on Neuron
Stage 6: Wire Into Model          → Integration with layers.py
Stage 7: Integration Test         → End-to-end pipeline validation
```

The key insight: **never go to Stage 6 until Stage 5 passes all shapes.** Our biggest time savings came from the standalone diagnostic script — it's 10x faster to iterate on a standalone test than to debug inside the full model pipeline.

---

## Stage 1: Extract & Catalog

### Goal
Build a structured inventory of the source kernel so you know exactly what needs converting.

### Inputs
- Source kernel from `kernels/kernel_builder/` or nkipy repo (https://github.com/aws-neuron/nkipy)
- Optionally: existing accuracy tests from nkipy

### Process

1. **Read the kernel source** — every line, not just the function signature
2. **List every API call** by category:

```markdown
## API Inventory for: [kernel_name]

### Buffer Allocation
- nb.ndarray((128, 512), nb.float32, memspace=nb.shared_hbm)  [line 45]
- nb.ndarray_like(scores)  [line 78]

### DMA Operations  
- nisa.dma_copy(dst=..., src=..., name="load_q", engine=nisa.engine.Sync)  [line 52]

### Arithmetic
- nisa.tensor_tensor_arith(dst=..., lhs=..., rhs=..., op=nisa.arith_op.Multiply)  [line 89]
- nisa.tensor_scalar_cache_reduce(dst=..., reduce_res=..., ...)  [line 95]

### Loop Constructs
- nb.fori_loop(batch_size, body_fn)  [line 30]
- for section_i in range(num_sections): [line 60] — BRANCHES on section_i

### View Operations
- tile.rearrange("p (c two) -> p c two", two=2)  [line 112]

### Branching
- if section_i == 0:  [line 65]  ← HAZARD: LoopVar branching
- if section_i == num_sections - 1:  [line 98]  ← HAZARD
```

3. **Extract the kernel signature**: inputs, outputs, scalar params
4. **Extract production shapes** from the call-site in `layers.py`
5. **Check nkipy for existing test scripts** — these contain ground truth shapes and tolerances

### Output
Structured inventory document (can be a comment block at the top of the new kernel file, or a separate analysis file).

---

## Stage 2: Pattern Detection

### Goal
Identify every structural hazard — patterns that are legal in kernel_builder but will crash or silently corrupt in bundled NKI.

### Checklist (apply to every item in the Stage 1 inventory)

| # | Pattern | Detection | Impact | Fix Pattern |
|---|---------|-----------|--------|-------------|
| 1 | **LoopVar branching** | `if loop_var == N:` where loop_var comes from `fori_loop` or `range()` used with NKI ops | Trace crash or scope escape error | Branchless algorithm (§4 Pattern 1 in conversion guide) |
| 2 | **LoopVar indexing** | `python_list[loop_var]` | `TypeError: list indices must be integers` | Tensor indexing or pass data as kernel input tensor |
| 3 | **Variable scope escape** | Variable assigned inside `if/else`, used outside | `SyntaxError: local variable referenced outside scope` | Compute unconditionally; use -inf/0 init for identity behavior |
| 4 | **fori_loop** | `nb.fori_loop(bound, body)` | Not available in bundled NKI | `nl.sequential_range(bound)` |
| 5 | **ndarray_like** | `nb.ndarray_like(x)` | Not available | `nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)` |
| 6 | **View transforms** | `.rearrange()`, `.repeat()`, `.view()` | Not available | Manual slicing (`[:, 0::2]`, `[:, 1::2]`) |
| 7 | **Fused ops** | `tensor_scalar_cache_reduce` | Not available | Split into separate scale + reduce |
| 8 | **Matmul accumulation** | `nisa.matmul(..., accum=True)` | `nc_matmul` has no `accum` param | Software accumulation in SBUF |
| 9 | **Writing to inputs** | `nisa.dma_copy(dst=input_param, ...)` | Immutable input parameters | Allocate output via `nl.shared_hbm` and return, or use `tensor.copy_()` from PyTorch |
| 10 | **Named ops / engine** | `name="..."`, `engine=nisa.engine.Sync` | Not available — just remove |

### Decision Gate
If Pattern 9 (immutable inputs) makes the kernel fundamentally non-portable (e.g., `kv_cache_copy` which writes to dst params), **stop here** and use PyTorch fallback instead. Document why in the kernel catalogue.

### Output
Annotated hazard list with fix strategy for each.

---

## Stage 3: Transpile

### Goal
Produce a first draft of the bundled NKI kernel.

### Process

**Step 3a: Mechanical translation** — Apply the API Translation Table (§2 of conversion guide):
- Imports: `nb.*` → `nl.*`, `nisa` stays as `nisa` but from different package
- `dst=` style → return style
- Remove `name=`, `engine=` params
- `nb.ds()` → `nl.ds()` (identical), `nb.ts()` → manual `nl.ds()`

**Step 3b: Structural transformation** — Apply patterns from Stage 2:
- Replace `fori_loop` → `nl.sequential_range`
- Replace branching on section_i → branchless online softmax with `-inf` init
- Replace Python list masking → tensor mask passed by caller
- Replace `.rearrange()` → manual slicing
- Replace fused ops → separate ops
- Replace `accum=True` matmul → software accumulation

**Step 3c: Critical rules** (apply always):
- **Use `nl.sequential_range` for any loop that does HBM IO** (load/store/dma_copy). NEVER use `affine_range` for this — it causes silent corruption when trip count > 8.
- **Use `range()` (Python) for inner loops** that only touch SBUF tiles already loaded — this unrolls at trace time with concrete Python ints, which is safe.
- **Allocate output with `nl.shared_hbm`** and return it (never write to input params).
- **Pad dimensions to tile multiples** (128 for partition dim, 512 for free dim in matmul).

### Template Structure

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def my_kernel(q, k, v, identity, mask, softmax_scale=1.0, num_sections=1):
    # ── Constants ──
    P = 128   # partition dimension (NKI tile height)
    ...
    
    # ── Allocate output in shared HBM ──
    out = nl.ndarray((seq_q, bs, d), dtype=q.dtype, buffer=nl.shared_hbm)
    
    # ── Batch loop (sequential — LoopVar) ──
    for batch_id in nl.sequential_range(bs):
        
        # ── Initialize running state (branchless softmax) ──
        r_max = nisa.memset((P, 1), value=float('-inf'), dtype=nl.float32)
        r_sum = nisa.memset((P, 1), value=0.0, dtype=nl.float32)
        pv_acc = nisa.memset((P, d), value=0.0, dtype=nl.float32)
        
        # ── Section loop (sequential — LoopVar) ──
        for section_i in nl.sequential_range(num_sections):
            # Load Q, K tiles
            # Compute QK^T via nc_matmul
            # Scale scores
            # Load and apply mask (add 0 or -inf)
            # Online softmax update (branchless)
            # Compute PV contribution
            pass
        
        # ── Normalize (after loop, not inside) ──
        rcp = nisa.reciprocal(r_sum)
        final = nisa.tensor_tensor(pv_acc, rcp, nl.multiply)
        
        # ── Store output ──
        nl.store(out[...], final)
    
    return out
```

### Output
First-draft kernel file in `kernels/[name].py`.

---

## Stage 4: Generate Diagnostic Script

### Goal
Create a standalone test script that validates the kernel against PyTorch SDPA at all production shapes.

### Process

1. **Extract production shapes** from `layers.py` call-site:
   - `frame_length`, `block_length`, `max_attention_size`
   - Concrete `seq_q`, `seq_k`, `actual_seqlen_k` combinations
   - `num_heads`, `head_dim`

2. **Build test matrix** — at minimum include:
   - Smallest production shape (1 section, heavy masking)
   - Largest production shape (max sections, near-full)
   - Edge cases: exact section boundaries (no masking), just past boundary (1 token masked)
   - Different seq_q sizes if applicable

3. **Script structure:**

```python
#!/usr/bin/env python3
"""Diagnostic: validate [kernel_name] NKI kernel against PyTorch SDPA reference."""

import torch
import torch.nn.functional as F

# ── Reference implementation (CPU, float32) ──
def sdpa_reference(q, k, v, softmax_scale, actual_seqlen_k):
    """PyTorch SDPA with masking — the ground truth."""
    ...
    return out

# ── Test matrix ──
TEST_CASES = [
    {"name": "anchor_block",    "seq_q": 4680, "seq_k": 8192,  "actual_k": 4680},
    {"name": "2_blocks",        "seq_q": 4680, "seq_k": 16384, "actual_k": 9360},
    {"name": "full_cache",      "seq_q": 4680, "seq_k": 32768, "actual_k": 32760},
    {"name": "exact_1section",  "seq_q": 4680, "seq_k": 8192,  "actual_k": 8192},
    ...
]

# ── Phase 1: CPU reference validation ──
print("=== Phase 1: CPU Reference ===")
for tc in TEST_CASES:
    # Generate inputs, run SDPA, verify finite outputs
    ...

# ── Phase 2: NKI kernel validation ──
print("=== Phase 2: NKI Kernel on Neuron ===")
from kernels.my_kernel import my_kernel
from torch_neuronx.nki_hop import wrap_nki
kernel = wrap_nki(my_kernel)

for tc in TEST_CASES:
    torch.manual_seed(42)
    # Generate inputs, move to neuron, build mask, pad
    # Run kernel
    # Compare against CPU reference
    max_diff = (out_nki_cpu.float() - out_ref.float()).abs().max().item()
    status = "✅ PASS" if max_diff < TOLERANCE else "❌ FAIL"
    print(f"  {tc['name']}: max_diff={max_diff:.6f} {status}")
```

4. **Tolerance thresholds** (from empirical validation):
   - bf16 flash attention: `max_diff < 2.0` (typical < 0.001)
   - RoPE rotation: `max_diff = 0.0` (exact match)
   - Cross-attention: `rtol=1e-2, atol=1e-3`
   - Simple copy/slice: `rtol=0, atol=0`

### Output
Standalone `[kernel_name]_diag.py` script.

---

## Stage 5: Iterate Until Accuracy

### Goal
Run the diagnostic on Neuron hardware, fix failures, repeat until ALL shapes pass.

### The Loop

```
while any_test_fails:
    1. Run diagnostic on Neuron pod
    2. Collect results (PASS/FAIL per shape, max_diff, error messages)
    3. Diagnose failure:
       a. Compilation error → consult Common Error Messages (§Appendix)
       b. Wrong output (small shapes OK, large shapes FAIL) → affine_range corruption
       c. Wrong output (all shapes) → algorithmic bug in softmax / masking
       d. Exit code 137 → OOM (not a kernel bug, reduce test count)
    4. Apply fix to kernel
    5. Clear NEFF cache: rm -rf /tmp/neff_cache/
    6. Go to 1
```

### Common Failure Modes (Ranked by Frequency)

1. **`TypeError: list indices must be integers, not LoopVar`** — missed a Python list indexed by loop var. Fix: convert to tensor.
2. **`SyntaxError: local variable referenced outside scope`** — if/else scope escape. Fix: branchless computation.
3. **Correct for ≤8 tiles, wrong for >8** — `affine_range` used for HBM IO. Fix: change to `sequential_range`.
4. **All shapes wrong by large factor** — softmax normalization bug. Fix: check running_sum accumulation.
5. **Shapes with masking wrong, unmasked shapes correct** — mask tensor not applied correctly. Fix: verify mask construction (0 for valid, -inf for invalid) and that mask sections align with K sections.
6. **Exit code 137** — OOM. Not a kernel bug. Add `del` + `gc.collect()` between test cases, or reduce test count.

### Success Criteria
- ALL test cases show ✅ PASS
- max_diff within tolerance for every shape
- No compilation warnings about unsupported operations

### Output
Validated kernel + passing diagnostic results.

---

## Stage 6: Wire Into Model

### Goal
Enable the kernel in `layers.py` so the model pipeline uses it.

### Pre-requisite
Stage 5 ALL PASS.

### Process

**6a. Module-level import and wrapping:**
```python
# In layers.py, at module level:
if USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.self_attention import wan_flash_self_attn
        wan_flash_self_attn_nki = wrap_nki(wan_flash_self_attn)
        SELF_ATTN_NKI_AVAILABLE = True
    except Exception as e:
        SELF_ATTN_NKI_AVAILABLE = False
```

**6b. Constructor assignment:**
```python
# In __init__:
self._nki_available = SELF_ATTN_NKI_AVAILABLE
self._self_attn_kernel = wan_flash_self_attn_nki if SELF_ATTN_NKI_AVAILABLE else None
```

**6c. Forward method:**
```python
# Build mask tensor for NKI kernel
if self._nki_available:
    mask = torch.zeros(128, seq_k_padded, dtype=torch.bfloat16, device=q.device)
    if actual_seqlen_k < seq_k_padded:
        mask[:, actual_seqlen_k:] = float('-inf')
    
    num_sections = seq_k_padded // SECTION_SIZE
    
    x = self._self_attn_kernel(
        q_kern, k_kern, v_kern, self.identity, mask,
        softmax_scale=self.softmax_scale,
        num_sections=num_sections)
    x = x[:seq_q].unsqueeze(0).flatten(2)  # trim padding, reshape
```

**6d. Keep PyTorch fallback:**
```python
else:
    # PyTorch SDPA fallback (CPU or Neuron without NKI)
    attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
    x = attn_out.permute(0, 2, 1, 3).flatten(2)
```

### Output
Updated `layers.py` with kernel enabled + fallback preserved.

---

## Stage 7: Integration Test

### Goal
Verify the kernel works correctly in the full model pipeline.

### Process
1. Run the model serving pipeline end-to-end with `USE_NKI_KERNELS=true`
2. Generate a test video/frame sequence
3. Compare output against baseline (generated with PyTorch SDPA fallback)
4. Check for visual artifacts, NaN outputs, or quality regression
5. Profile: the NKI kernel should match or beat SDPA performance

### Rollback
If integration test fails but diagnostic passes, the issue is in the wiring (Stage 6) — likely a reshape, padding, or mask construction bug. Debug in Stage 6, not Stage 3.

---

## Appendix A: nkipy as Source Material

The nkipy repo (https://github.com/aws-neuron/nkipy) contains kernel_builder kernels that can serve as source material for Stage 1. Key directories:

- `nkipy/kernels/` — kernel implementations
- `nkipy/tests/` — accuracy tests (extract shapes and tolerances from these!)
- `nkipy/benchmarks/` — performance baselines

When using nkipy as source:
1. Clone the repo and locate the kernel
2. Check if there's a corresponding test — this gives you shapes and expected accuracy
3. The test's reference implementation (usually PyTorch SDPA) becomes your diagnostic ground truth
4. The test's tolerance becomes your Stage 5 success criteria

## Appendix B: Files in This Project

| File | Purpose |
|------|---------|
| `docs/kernel-builder-to-nki-conversion.md` | API tables, tracer constraints, conversion patterns |
| `docs/nki-kernel-migration-agent-workflow.md` | This file — 7-stage agent workflow |
| `kernels/kernel_builder/*.py` | Original kernel_builder kernels (reference) |
| `kernels/*.py` | Ported bundled NKI kernels |
| `*_diag.py` | Diagnostic scripts (standalone accuracy tests) |
| `models/layers.py` | Model integration (where kernels are wired) |
| `.clinerules` | Agent rules including kernel migration skill |
