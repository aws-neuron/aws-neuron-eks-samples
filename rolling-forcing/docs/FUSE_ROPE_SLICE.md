# Fuse contiguous_slice + cos/sin Construction into RoPE Kernel

## Current Call Path (layers.py lines 618-661)

```python
# 1. Build cos/sin grids from frequency tables (PyTorch, ~10 ops)
frame_idx = start_frame + torch.arange(f, device=x.device)
cos_half = torch.cat([
    torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(...).expand(...),
    freqs_cos[:h, s0:s0+s1].view(...).expand(...),
    freqs_cos[:w, s0+s1:].view(...).expand(...),
], dim=-1).reshape(seq_len, c)
sin_half = ...  # same pattern

# 2. Expand to interleaved full-D + sign pattern (PyTorch, ~5 ops)
cos_expanded = cos_half.repeat_interleave(2, dim=-1)
sin_expanded = sin_half.repeat_interleave(2, dim=-1)
sign[0::2] = -1.0
sin_signed = sin_expanded * sign

# 3. Pack into [seq_len, 2*D] (PyTorch, 1 cat + 1 contiguous)
cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()

# 4. Pad (PyTorch, 2 pad ops)
cos_sin = F.pad(cos_sin, (0, 0, 0, pad))
x_nki = F.pad(x[0, :seq_len], (0, 0, 0, 0, 0, pad))

# 5. Call kernel
out = self._rope_kernel(x_nki, cos_sin, num_heads=n, head_dim=d)

# 6. Slice output back (PyTorch, 1 slice + unsqueeze + type cast)
return out[:seq_len].unsqueeze(0).type_as(x)
```

Total: ~20 PyTorch ops OUTSIDE the kernel that each become separate NEFFs or graph fragments.

## Proposed: Fused RoPE Kernel

Pass raw frequency tables + frame index directly to the kernel. The kernel does grid-building, interleaving, and rotation in one fused operation.

### New Kernel Signature

```python
@nki.jit
def causal_rope_rotation_fused(x, freqs_cos, freqs_sin, start_frame, 
                                num_frames, h, w, num_heads, head_dim):
    """Fused RoPE: grid-build + interleave + rotate in one kernel.
    
    Args:
        x:          [seq_len_padded, num_heads, head_dim] bfloat16
        freqs_cos:  [max_seq, head_dim//2] float32 — raw frequency table
        freqs_sin:  [max_seq, head_dim//2] float32 — raw frequency table
        start_frame: int — starting frame index for temporal RoPE
        num_frames:  int — F dimension
        h, w:        int — spatial grid dimensions
        num_heads:   int
        head_dim:    int
    
    Returns:
        out: [seq_len_padded, num_heads, head_dim] bfloat16
    """
```

### What the Kernel Does Internally

For each tile of 128 tokens:
1. Compute which (f, h_pos, w_pos) each token maps to
2. Look up cos/sin from frequency tables for each position component
3. Concatenate the 3 RoPE components (temporal, height, width)
4. Apply interleave + sign pattern
5. Multiply: `out = x * cos + swap(x) * sin`

### Benefits

| Aspect | Current | Fused |
|--------|---------|-------|
| PyTorch ops before kernel | ~20 | 0 (just pad x) |
| Intermediate tensors | cos_half, sin_half, cos_expanded, sin_expanded, cos_sin | None |
| HBM traffic | x + cos_sin (2*D extra per token) | x + freq tables (small, reused) |
| NEFFs from prep | 5-10 small NEFFs | 0 |
| Graph breaks | Multiple (from ops between compile boundary and kernel) | None |

### Implementation Complexity

**Medium-high.** The grid-building logic (`index_select` by frame_idx, expand across h/w) needs to be reimplemented in NKI. The main challenges:

1. **3D position mapping**: Each token at position `p` maps to `(f, h_pos, w_pos)` = `(p // (h*w), (p % (h*w)) // w, p % w)`. Integer division in NKI.

2. **Frequency table lookup with dynamic index**: `freqs_cos[frame_idx + start_frame, :]` — need indirect DMA with the frame index. Since `start_frame` changes per call, this must be a runtime parameter (tensor, not int).

3. **Three separate frequency bands**: temporal uses `d - 4*(d//6)` dims, height uses `2*(d//6)`, width uses `2*(d//6)`. Different table rows for each.

4. **Interleave pattern**: `cos[2j] = cos[2j+1] = cos_half[j]` — can be done with `.repeat()` view in NKI (zero-copy).

### Alternative: Partial Fusion (Lower Effort)

Keep the cos_sin construction in PyTorch but eliminate the slice/pad/unslice:

```python
@nki.jit  
def causal_rope_rotation_v2(x, cos_sin, seq_len_valid, num_heads, head_dim):
    """Accept unpadded input, handle padding internally.
    
    x: [batch_seq, num_heads, head_dim] — may not be multiple of 128
    cos_sin: [seq_len_valid, 2*head_dim] — unpadded
    seq_len_valid: actual sequence length (kernel pads to tile boundary)
    """
    # Round up to tile boundary
    P = nl.tile_size.pmax
    num_tiles = (seq_len_valid + P - 1) // P
    
    out = nl.ndarray((num_tiles * P, num_heads, head_dim), ...)
    
    for tile_i in nl.sequential_range(num_tiles):
        # Load tile (last tile may have garbage beyond seq_len_valid — harmless)
        ...
```

This eliminates the pad ops from PyTorch (2 fewer ops/NEFFs) but keeps the cos_sin construction.

### Recommendation

1. **Short term (now)**: Partial fusion — modify kernel to accept unpadded inputs. Saves 2-4 NEFFs per RoPE call (4 calls per block = 8-16 fewer NEFFs across 30 blocks).

2. **Medium term**: Full fusion — pass freq tables directly. Eliminates ~20 PyTorch ops per RoPE call. Requires NKI indirect DMA for table lookup.

## Status

Ready to implement partial fusion. Full fusion requires more NKI development.
