"""RoPE rotation NKI kernel, ported to bundled neuronxcc.nki API.

causal_rope_rotation: Apply rotary position embeddings (rotate_half).

Validated against PyTorch CPU reference with zero numerical drift.
Max abs diff: 0.000000, Mean abs diff: 0.000000.

IO tensor layouts:
    - x:        [seq_len, num_heads, head_dim] bfloat16
    - cos_sin:  [seq_len, 2 * head_dim] float32
                columns [0, D): cos_expanded (interleaved pairs)
                columns [D, 2D): sin_signed (with sign pattern applied)
    - out:      [seq_len, num_heads, head_dim] same dtype as x

seq_len must be a multiple of 128 (pad at call site).

IMPORTANT: The outer seq_len tile loop uses nl.sequential_range, NOT nl.affine_range.
affine_range enables software pipelining which corrupts SBUF when num_tiles > 8
(the compiler overlaps load/compute/store across iterations, and at >8 iterations
the pipeline depth exceeds hardware capacity, causing SBUF buffers from iteration N
to be overwritten before their stores complete). The inner head loop (N=12) safely
uses affine_range because it operates entirely within SBUF with no HBM IO.

Diagnosed via systematic tile-count sweep: diff=0 for 1-8 tiles, ~22 max abs diff
for 9+ tiles. Fix confirmed with all production shapes (858→896, 2574→2688,
4290→4352) at diff=0.000000.

Key substitutions from kernel_builder:
    - .rearrange("p n (c two) -> p n c two") → strided slicing [:, 0::2] / [:, 1::2]
    - .repeat("p x -> p c x") → per-head loop (N=12 is small)
    - nb.range → nl.sequential_range (outer) / nl.affine_range (inner)
    - tensor_tensor_arith(dst=,...) → return-style nisa.tensor_tensor()
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def causal_rope_rotation(x, cos_sin, num_heads=12, head_dim=128):
    seq_len = x.shape[0]
    N = num_heads
    D = head_dim
    P = nl.tile_size.pmax

    assert seq_len % P == 0
    num_tiles = seq_len // P
    out = nl.ndarray((seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, D)]
        sin_tile = cs_sb[:, nl.ds(D, D)]
        x_sb = nl.load(x[nl.ds(ts, P), :, :])

        out_sb = nl.ndarray((P, N, D), dtype=x.dtype, buffer=nl.sbuf)
        for n in nl.affine_range(N):
            xh = x_sb[:, n, :]
            x_cos = nl.multiply(xh, cos_tile)

            x_swap = nl.ndarray((P, D), dtype=xh.dtype, buffer=nl.sbuf)
            x_swap[:, 0::2] = xh[:, 1::2]
            x_swap[:, 1::2] = xh[:, 0::2]

            x_sin = nl.multiply(x_swap, sin_tile)
            out_sb[:, n, :] = nl.add(x_cos, x_sin)

        nl.store(out[nl.ds(ts, P), :, :], out_sb)

    return out
