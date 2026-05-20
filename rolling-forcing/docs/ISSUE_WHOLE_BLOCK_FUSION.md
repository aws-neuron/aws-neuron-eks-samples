# `torch.compile(backend='neuron')`: NKI kernels receive non-contiguous tensors when called inside compiled graph

## Summary

When a module containing `@nki.jit` kernel calls is compiled via `torch.compile(module, backend='neuron')`, the NKI kernels fail at **execution time** with:

```
ERROR torch_neuronx.neuron_dynamo_backend.backend: Execution failed: Cannot process non-contiguous tensors
```

The `.contiguous()` calls placed before the NKI kernel invocations appear to be optimized away by the Neuron backend during compilation.

## Minimal Reproduction Pattern

```python
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def my_nki_kernel(q, k, v):
    """NKI kernel that requires contiguous inputs."""
    # ... kernel implementation ...
    pass

class MyModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_q = torch.nn.Linear(dim, dim)
        self.linear_k = torch.nn.Linear(dim, dim)
        self.linear_v = torch.nn.Linear(dim, dim)
        self.linear_o = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        # Projections produce [B, seq, heads, head_dim]
        q = self.linear_q(x).view(1, -1, 12, 128)
        k = self.linear_k(x).view(1, -1, 12, 128)
        v = self.linear_v(x).view(1, -1, 12, 128)
        
        # Reshape for NKI kernel: permute creates non-contiguous view
        # .contiguous() should force a copy but gets optimized away
        q_nki = q[0].permute(1, 2, 0).contiguous()   # [heads, head_dim, seq]
        k_nki = k[0].permute(1, 2, 0).contiguous()   # [heads, head_dim, seq]
        v_nki = v[0].permute(1, 0, 2).contiguous()   # [heads, seq, head_dim]
        
        # NKI kernel receives non-contiguous tensor at runtime
        out = my_nki_kernel(q_nki, k_nki, v_nki)
        
        return self.linear_o(out.unsqueeze(0).flatten(2))

# This works (NKI kernel called separately, outside compiled graph):
model = MyModule(1536).to('neuron')
model.linear_q = torch.compile(model.linear_q, backend='neuron')  # individual sub-modules

# This FAILS (NKI kernel called inside compiled graph):
model = MyModule(1536).to('neuron')
model = torch.compile(model, backend='neuron')  # whole module
```

## Actual Error Output

```
Neuron NKI - Kernel call: wan_flash_self_attn(
    q = Tensor(shape: (3, 128, 2688), dtype: bfloat16),
    k = Tensor(shape: (3, 128, 24576), dtype: bfloat16),
    v = Tensor(shape: (3, 24576, 128), dtype: bfloat16),
    identity = Tensor(shape: (128, 128), dtype: bfloat16),
    mask = Tensor(shape: (128, 24576), dtype: bfloat16),
    softmax_scale = 0.08838834764831843, num_sections = 3)
ERROR torch_neuronx.neuron_dynamo_backend.backend [rank 822]: Execution failed: Cannot process non-contiguous tensors
ERROR torch_neuronx.neuron_dynamo_backend.backend [rank 823]: Execution failed: Cannot process non-contiguous tensors
ERROR torch_neuronx.neuron_dynamo_backend.backend [rank 824]: Execution failed: Cannot process non-contiguous tensors
ERROR torch_neuronx.neuron_dynamo_backend.backend [rank 825]: Execution failed: Cannot process non-contiguous tensors
```

This repeats for every NKI kernel call in the model (`wan_flash_self_attn`, `wan_cross_attn`).

## Key Observations

1. **Compilation succeeds** — no error during tracing/compilation
2. **Error is at execution time** — the NEFF runs but the NKI kernel dispatch receives non-contiguous data
3. **Works fine without `torch.compile`** — calling the same module in eager mode works (`.contiguous()` is respected)
4. **Works when compiling sub-modules individually** — if you compile only the Linear layers and leave the NKI kernel calls in eager Python, it works

## Root Cause Hypothesis

When `torch.compile(backend='neuron')` traces through a module that contains `.permute().contiguous()` followed by an `@nki.jit` kernel call (via HOP):

1. The tracer sees `.permute()` as a view operation that changes strides
2. The subsequent `.contiguous()` may be seen as a no-op during tracing (if the tracer doesn't propagate stride information correctly) and gets eliminated
3. At runtime, `.permute()` produces a non-contiguous view
4. The NKI kernel (dispatched via HOP) receives this non-contiguous view and fails

## Expected Behavior

Either:
- **The Neuron backend should preserve `.contiguous()` after stride-changing operations** (`.permute()`, `.transpose()`, `.view()` with different strides) — it should never optimize away `.contiguous()` when the preceding op changes memory layout
- **OR the NKI HOP dispatch should automatically call `.contiguous()` on inputs** before passing them to the kernel
- **OR this should fail at compile time** with a clear error, not silently at runtime

## Workaround Attempts

| Approach | Result |
|----------|--------|
| `.contiguous()` after `.permute()` | ❌ Optimized away |
| `.clone()` instead of `.contiguous()` | ❓ Not yet tested (may force the copy) |
| Compile sub-modules individually (no NKI in graph) | ✅ Works but defeats fusion purpose |
| `@torch.compiler.disable` on NKI call-sites | ✅ Works but creates graph breaks (defeats purpose) |

## Why This Matters

In our video generation model (30-layer DiT transformer), compiling sub-modules individually produces **437 NEFFs** with **425K kernel launches** — resulting in **0.03% MFU** because NeuronCores are idle 99.97% of the time waiting for Python scheduling between kernel launches.

Compiling whole transformer blocks would reduce this to **~67 NEFFs** with an estimated **7-8x performance improvement**. But this is blocked by the non-contiguous tensor issue when NKI kernels are called inside the compiled graph.

## Environment

- torch-neuronx version: 2.6.0.2.3.x (SDK 2.30.x release)
- PyTorch version: 2.6
- neuronx-cc version: 2.23.4912.0+c6eb7195
- OS: Ubuntu 22.04
- Instance type: trn2.48xlarge
- Python version: 3.12
- Neuron Runtime: 2.30.50 (78eee)
- Neuron Driver: 2.27.0
- Container image: `421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-d1c940d:latest`
