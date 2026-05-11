import neuronxcc.nki as nki
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor
from nki.compiler.kernel_builder import isa as nisa


@nki.jit
def causal_rope_rotation(
    x: Tensor,
    cos_sin: Tensor,
    num_heads: int = 12,
    head_dim: int = 128,
):
    """NKI kernel for rotary position embedding (RoPE) rotation.

    Implements the rotate_half formulation:
        out = x * cos_expanded + swap_pairs(x) * sin_signed

    where swap_pairs exchanges adjacent element pairs via rearrange views.
    cos/sin are broadcast across the num_heads dimension via tensor_tensor_arith.

    IO tensor layouts:
        - x:        [seq_len, num_heads, head_dim] bfloat16
        - cos_sin:  [seq_len, 2 * head_dim] float32
                    columns [0, D): cos_expanded
                    columns [D, 2D): sin_signed
        - out:      [seq_len, num_heads, head_dim] same dtype as x
    """
    seq_len = x.shape[0]
    N = num_heads
    D = head_dim
    P = 128

    out = nb.ndarray((seq_len, N, D), x.dtype, memspace=nb.shared_hbm)

    num_tiles = (seq_len + P - 1) // P

    for tile_i in range(num_tiles):
        tile_start = tile_i * P
        tile_size = min(P, seq_len - tile_start)

        # Load cos+sin [P, 2*D], then reshape to [P, 1, D] views for broadcast
        cos_sin_tile = nb.ndarray((P, 2 * D), nb.float32, num_buffers=2, name="cos_sin_tile")
        if tile_size < P:
            nisa.memset(dst=cos_sin_tile, value=0.0)
        nisa.dma_copy(
            dst=cos_sin_tile[:tile_size, :],
            src=cos_sin[nb.ds(tile_start, tile_size), :],
        )
        cos_tile = cos_sin_tile[:, nb.ds(0, D)].repeat("p x -> p c x", c=N)
        sin_tile = cos_sin_tile[:, nb.ds(D, D)].repeat("p x -> p c x", c=N)

        # Load ALL heads in one DMA [P, N, D]
        x_all = nb.ndarray((P, N, D), nb.bfloat16, num_buffers=2, name="x_all")
        if tile_size < P:
            nisa.memset(dst=x_all, value=0.0)
        nisa.dma_copy(
            dst=x_all[:tile_size, :, :],
            src=x[nb.ds(tile_start, tile_size), :, :],
        )

        # Extract even/odd elements via rearrange views (zero-copy)
        # x_shaped[..., j, 0] = x[..., 2j], x_shaped[..., j, 1] = x[..., 2j+1]
        x_shaped = x_all.rearrange("p n (c two) -> p n c two", two=2)
        x_even = x_shaped[:, :, :, nb.ds(0, 1)]
        x_odd = x_shaped[:, :, :, nb.ds(1, 1)]

        # Reshape sin_tile the same way for pair-wise access
        sin_shaped = sin_tile.rearrange("p n (c two) -> p n c two", two=2)
        sin_even = sin_shaped[:, :, :, nb.ds(0, 1)]
        sin_odd = sin_shaped[:, :, :, nb.ds(1, 1)]

        # x * cos — cos_tile [P, 1, D] broadcasts across N heads
        x_cos_all = nb.ndarray((P, N, D), nb.float32, name="x_cos_all")
        nisa.tensor_tensor_arith(
            dst=x_cos_all, lhs=x_all, rhs=cos_tile,
            op=nisa.arith_op.Multiply,
        )

        # swap(x) * sin via reshape views — no gather needed
        # Adjacent pair swap in (P, N, D//2, 2) layout:
        #   x_sin[..., 2j]   = x[..., 2j+1] * sin[..., 2j]   → odd * sin_even
        #   x_sin[..., 2j+1] = x[..., 2j]   * sin[..., 2j+1] → even * sin_odd
        x_sin_all = nb.ndarray((P, N, D), nb.float32, name="x_sin_all")
        x_sin_shaped = x_sin_all.rearrange("p n (c two) -> p n c two", two=2)
        nisa.tensor_tensor_arith(
            dst=x_sin_shaped[:, :, :, nb.ds(0, 1)],
            lhs=x_odd, rhs=sin_even,
            op=nisa.arith_op.Multiply,
        )
        nisa.tensor_tensor_arith(
            dst=x_sin_shaped[:, :, :, nb.ds(1, 1)],
            lhs=x_even, rhs=sin_odd,
            op=nisa.arith_op.Multiply,
        )

        # out = x_cos + x_sin
        out_all = nb.ndarray((P, N, D), x.dtype, num_buffers=2, name="out_all")
        nisa.tensor_tensor_arith(
            dst=out_all, lhs=x_cos_all, rhs=x_sin_all,
            op=nisa.arith_op.Add,
        )

        # Store ALL heads in one DMA [P, N, D]
        nisa.dma_copy(
            dst=out[nb.ds(tile_start, tile_size), :, :],
            src=out_all[:tile_size, :, :],
        )

    return out


@nki.jit
def build_rope_grids(
    freqs_cos: Tensor,
    freqs_sin: Tensor,
    sign_pattern: Tensor,
    start_frame: Tensor,
    F: int = 15,
    H: int = 30,
    W: int = 52,
    head_dim: int = 128,
):
    """NKI kernel to build RoPE cos/sin grids from frequency tables.

    Constructs the 3D positional grid (frame x height x width), gathers
    frequencies for each axis, interleaves to full head_dim, and applies
    the sign pattern for rotate_half.

    Processes all H positions in parallel using H as the partition dimension
    with multi-dimensional broadcasting (no explicit H loop). Requires H <= 128.

    IO tensor layouts:
        - freqs_cos:      [max_seq_len, head_dim // 2] float32
        - freqs_sin:      [max_seq_len, head_dim // 2] float32
        - sign_pattern:   [128, head_dim] float32 — sign[:, 2j] = -1, sign[:, 2j+1] = 1
        - start_frame:    [1, 1] int32 — starting frame index (tensor to avoid recompilation)
        - combined_out:   [F*H, W * 2 * head_dim] float32
                          Physically identical to [F*H*W, 2 * head_dim].
                          Caller should .view(F*H*W, 2*head_dim) if needed.
    """
    P = 128
    c = head_dim // 2
    D = head_dim
    s0 = c - 2 * (c // 3)
    s1 = c // 3

    combined_out = nb.ndarray((F * H, W * 2 * D), nb.float32, memspace=nb.shared_hbm)

    # ── Load sign pattern [H, D] → view as [H, W, c, 2] for broadcast ──
    sign_H = nb.ndarray((H, D), nb.float32, name="sign_H")
    nisa.dma_copy(dst=sign_H, src=sign_pattern[:H, :])
    sign_4d = sign_H.rearrange("p (a b) -> p a b", b=2).repeat("p a b -> p w a b", w=W)

    # ── ones vector [1, H] for partition-dim broadcast via matmul ──
    ones_H = nb.ndarray((1, H), nb.float32, name="ones_H")
    nisa.memset(dst=ones_H, value=1.0)

    # ── Load start_frame into a register for indirect DMA ──
    sf_sbuf = nb.ndarray((1, 1), nb.int32, name="sf")
    nisa.dma_copy(dst=sf_sbuf, src=start_frame)
    sf_reg = nisa.load_register(src=sf_sbuf)

    # ── Preload width freqs as [1, W*s1] for partition broadcast ──
    # Load [W, s1] from HBM, then flatten to [1, W*s1] via SBUF-to-SBUF DMA
    w_cos_raw = nb.ndarray((P, s1), nb.float32, name="w_cos_raw")
    nisa.memset(dst=w_cos_raw, value=0.0)
    nisa.dma_copy(dst=w_cos_raw[:W, :], src=freqs_cos[nb.ds(0, W), nb.ds(s0 + s1, s1)])

    w_cos_flat = nb.ndarray((1, W * s1), nb.float32, name="w_cos_flat")
    for wi in nb.range(W):
        nisa.dma_copy(dst=w_cos_flat[:, nb.ds(wi * s1, s1)],
                      src=w_cos_raw[nb.ds(wi, 1), :])

    w_sin_raw = nb.ndarray((P, s1), nb.float32, name="w_sin_raw")
    nisa.memset(dst=w_sin_raw, value=0.0)
    nisa.dma_copy(dst=w_sin_raw[:W, :], src=freqs_sin[nb.ds(0, W), nb.ds(s0 + s1, s1)])

    w_sin_flat = nb.ndarray((1, W * s1), nb.float32, name="w_sin_flat")
    for wi in nb.range(W):
        nisa.dma_copy(dst=w_sin_flat[:, nb.ds(wi * s1, s1)],
                      src=w_sin_raw[nb.ds(wi, 1), :])

    # ── Broadcast width to [H, W*s1] via chunked matmul outer product ──
    # Matmul moving free dim limited to 512, so chunk and pad W*s1
    MATMUL_FREE_MAX = 512
    ws = W * s1
    num_chunks = (ws + MATMUL_FREE_MAX - 1) // MATMUL_FREE_MAX
    ws_padded = num_chunks * MATMUL_FREE_MAX

    # Pad flat buffers so all chunks are uniform size
    w_cos_pad = nb.ndarray((1, ws_padded), nb.float32, name="w_cos_pad")
    nisa.memset(dst=w_cos_pad, value=0.0)
    nisa.tensor_copy(dst=w_cos_pad[:, nb.ds(0, ws)], src=w_cos_flat)

    w_sin_pad = nb.ndarray((1, ws_padded), nb.float32, name="w_sin_pad")
    nisa.memset(dst=w_sin_pad, value=0.0)
    nisa.tensor_copy(dst=w_sin_pad[:, nb.ds(0, ws)], src=w_sin_flat)

    w_cos_bc_pad = nb.ndarray((H, ws_padded), nb.float32, name="w_cos_bc_pad")
    w_sin_bc_pad = nb.ndarray((H, ws_padded), nb.float32, name="w_sin_bc_pad")
    psum_chunk = nb.ndarray((H, MATMUL_FREE_MAX), nb.float32, memspace=nb.psum,
                            name="psum_wc")
    for chunk_off in range(0, ws_padded, MATMUL_FREE_MAX):
        nisa.matmul(dst=psum_chunk, stationary=ones_H,
                    moving=w_cos_pad[:, nb.ds(chunk_off, MATMUL_FREE_MAX)], accum=False)
        nisa.tensor_copy(dst=w_cos_bc_pad[:, nb.ds(chunk_off, MATMUL_FREE_MAX)],
                         src=psum_chunk)
        nisa.matmul(dst=psum_chunk, stationary=ones_H,
                    moving=w_sin_pad[:, nb.ds(chunk_off, MATMUL_FREE_MAX)], accum=False)
        nisa.tensor_copy(dst=w_sin_bc_pad[:, nb.ds(chunk_off, MATMUL_FREE_MAX)],
                         src=psum_chunk)

    # Take the valid [H, ws] slice for downstream use
    w_cos_bc = w_cos_bc_pad[:, nb.ds(0, ws)]
    w_sin_bc = w_sin_bc_pad[:, nb.ds(0, ws)]

    # ── Load height frequencies [H, s1] ──
    h_cos = nb.ndarray((H, s1), nb.float32, name="h_cos")
    nisa.dma_copy(dst=h_cos, src=freqs_cos[nb.ds(0, H), nb.ds(s0, s1)])

    h_sin = nb.ndarray((H, s1), nb.float32, name="h_sin")
    nisa.dma_copy(dst=h_sin, src=freqs_sin[nb.ds(0, H), nb.ds(s0, s1)])

    # ── Preload frame frequency rows [F, c] via indirect DMA ──
    frame_cos = nb.ndarray((P, c), nb.float32, name="frame_cos")
    nisa.memset(dst=frame_cos, value=0.0)
    nisa.dma_copy(dst=frame_cos[:F, :], src=freqs_cos[nb.ds(sf_reg, F), :])

    frame_sin = nb.ndarray((P, c), nb.float32, name="frame_sin")
    nisa.memset(dst=frame_sin, value=0.0)
    nisa.dma_copy(dst=frame_sin[:F, :], src=freqs_sin[nb.ds(sf_reg, F), :])

    # ── PSUM buffer for frame broadcast ──
    psum_fs = nb.ndarray((H, s0), nb.float32, memspace=nb.psum, name="psum_fs")

    for f in nb.range(F):
        # ── Frame cos: [1, s0] → broadcast to [H, s0] via matmul ──
        fc_row = nb.ndarray((1, s0), nb.float32, name="fc_row")
        nisa.dma_copy(dst=fc_row, src=frame_cos[nb.ds(f, 1), nb.ds(0, s0)])
        nisa.matmul(dst=psum_fs, stationary=ones_H, moving=fc_row, accum=False)
        fc_bc = nb.ndarray((H, s0), nb.float32, name="fc_bc")
        nisa.tensor_copy(dst=fc_bc, src=psum_fs)

        # ── Frame sin: [1, s0] → broadcast to [H, s0] via matmul ──
        fs_row = nb.ndarray((1, s0), nb.float32, name="fs_row")
        nisa.dma_copy(dst=fs_row, src=frame_sin[nb.ds(f, 1), nb.ds(0, s0)])
        nisa.matmul(dst=psum_fs, stationary=ones_H, moving=fs_row, accum=False)
        fs_bc = nb.ndarray((H, s0), nb.float32, name="fs_bc")
        nisa.tensor_copy(dst=fs_bc, src=psum_fs)

        # ── Assemble cos [H, W, c] via rearrange + broadcast views ──
        cos_full = nb.ndarray((H, W * c), nb.float32, name="cos_full")
        cos_3d = cos_full.rearrange("p (w x) -> p w x", w=W)
        nisa.tensor_copy(dst=cos_3d[:, :, nb.ds(0, s0)],
                         src=fc_bc.repeat("p x -> p w x", w=W))
        nisa.tensor_copy(dst=cos_3d[:, :, nb.ds(s0, s1)],
                         src=h_cos.repeat("p x -> p w x", w=W))
        nisa.tensor_copy(dst=cos_3d[:, :, nb.ds(s0 + s1, s1)],
                         src=w_cos_bc.rearrange("p (w s) -> p w s", w=W))

        # ── Assemble sin [H, W, c] ──
        sin_full = nb.ndarray((H, W * c), nb.float32, name="sin_full")
        sin_3d = sin_full.rearrange("p (w x) -> p w x", w=W)
        nisa.tensor_copy(dst=sin_3d[:, :, nb.ds(0, s0)],
                         src=fs_bc.repeat("p x -> p w x", w=W))
        nisa.tensor_copy(dst=sin_3d[:, :, nb.ds(s0, s1)],
                         src=h_sin.repeat("p x -> p w x", w=W))
        nisa.tensor_copy(dst=sin_3d[:, :, nb.ds(s0 + s1, s1)],
                         src=w_sin_bc.rearrange("p (w s) -> p w s", w=W))

        # ── Interleave [H, W, c] → [H, W, c, 2] via zero-copy repeat ──
        cos_e = cos_3d.repeat("p w x -> p w x b", b=2)
        sin_e = sin_3d.repeat("p w x -> p w x b", b=2)

        # ── Apply sign: sin_signed = sin_interleaved * sign ──
        sin_s = nb.ndarray((H, W * D), nb.float32, name="sin_s")
        sin_s_4d = sin_s.rearrange("p (w a b) -> p w a b", w=W, b=2)
        nisa.tensor_tensor_arith(
            dst=sin_s_4d, lhs=sin_e, rhs=sign_4d,
            op=nisa.arith_op.Multiply,
        )

        # ── Assemble combined [H, W*2*D] and store ──
        combined_tile = nb.ndarray((H, W * 2 * D), nb.float32, name="combined_tile")
        combined_3d = combined_tile.rearrange("p (w d) -> p w d", w=W)
        nisa.tensor_copy(
            dst=combined_3d[:, :, nb.ds(0, D)].rearrange("p w (a b) -> p w a b", b=2),
            src=cos_e,
        )
        nisa.tensor_copy(
            dst=combined_3d[:, :, nb.ds(D, D)].rearrange("p w (a b) -> p w a b", b=2),
            src=sin_s_4d,
        )

        nisa.dma_copy(
            dst=combined_out[nb.ds(f * H, H), :],
            src=combined_tile,
        )

    return combined_out
