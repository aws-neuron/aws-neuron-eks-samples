"""Cross-attention NKI kernel using bundled neuronxcc.nki API.

Adapted to the bundled neuronxcc.nki.isa return-style API:
- dma_copy: keyword-only (dst=, src=), in-place, returns None
- nc_matmul: returns PSUM tile (stationary.T @ moving)
- tensor_copy/scalar/tensor/reduce/activation/reciprocal: return new tile
- memset: returns new tile from (shape, value, dtype)

Algorithm: single-pass flash attention for small seq_k (512 from T5).
Uses identity matmul trick for transpose in PV computation.
Accumulation done in SBUF (PSUM is write-only by nc_matmul).
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np


@nki.jit
def wan_cross_attn(q, k, v, identity, softmax_scale=None):
    """Flash cross-attention kernel for Wan T2V DiT blocks.

    IO tensor layouts:
        - q:        (bs, d, seq_q)  bs=num_heads=12, d=head_dim=128
        - k:        (bs, d, seq_k)  seq_k=512 (T5 text tokens)
        - v:        (bs, seq_k, d)
        - identity: (128, 128)      used for transpose trick via nc_matmul
        - out:      (seq_q, bs, d)  output
    """
    batch_size = q.shape[0]       # num_heads (12)
    d = q.shape[1]                # head_dim (128)
    seqlen_q = q.shape[2]         # frame_seq_length * num_frames
    seqlen_k = k.shape[2]         # 512 (T5 output length)

    P = nl.tile_size.pmax          # 128
    assert seqlen_q % P == 0, f"seqlen_q ({seqlen_q}) must be a multiple of P ({P}). Pad at call site."
    num_q_grps = seqlen_q // P
    num_v_tiles = seqlen_k // P    # 512 / 128 = 4

    # Allocate output in HBM
    out = nl.ndarray((seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity matrix into SBUF (used for transpose trick)
    id_sbuf = nl.ndarray((P, P), dtype=identity.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for batch_id in range(batch_size):
        # ── Load K: [d=128, seq_k=512] ────────────────────────
        k_buf = nl.ndarray((d, seqlen_k), dtype=k.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_buf, src=k[batch_id])

        # ── Process Q in groups of P=128 tokens ───────────────
        for gi in range(num_q_grps):
            q_start = gi * P

            # Load Q tile: [d=128, P=128] — seqlen_q guaranteed multiple of P
            q_buf = nl.ndarray((d, P), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_buf, src=q[batch_id, :, nl.ds(q_start, P)])

            # ── Phase 1: QK^T — attention scores ──────────────
            # nc_matmul: stationary.T @ moving
            # stationary=Q[d,P], moving=K[d,seq_k] → Q.T @ K = [P, seq_k]
            qk_psum = nisa.nc_matmul(q_buf, k_buf)  # returns PSUM [P, seq_k]

            # Copy PSUM→SBUF and apply scale
            qk = nisa.tensor_copy(qk_psum)  # PSUM → SBUF
            qk = nisa.tensor_scalar(qk, nl.multiply, softmax_scale)

            # ── Phase 2: Numerically stable softmax ───────────
            # Row max
            row_max = nisa.tensor_reduce(nl.maximum, qk, axis=1)

            # Subtract max: qk_shifted = qk - row_max
            qk_shifted = nisa.tensor_tensor(qk, row_max, nl.subtract)

            # Exp
            exp_qk = nisa.activation(nl.exp, qk_shifted)

            # Row sum
            row_sum = nisa.tensor_reduce(nl.add, exp_qk, axis=1)

            # Reciprocal of row_sum for normalization
            row_sum_recip = nisa.reciprocal(row_sum)

            # ── Phase 3: PV matmul ────────────────────────────
            # Accumulate in SBUF (PSUM is only written by nc_matmul)
            pv_accum = nisa.memset((P, d), value=0.0, dtype=nl.float32)

            for vi in range(num_v_tiles):
                # Load V tile: [P=128, d=128]
                v_tile = nl.ndarray((P, d), dtype=v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=v_tile, src=v[batch_id, nl.ds(vi * P, P), :])

                # Extract attention weights for this chunk
                attn_chunk = nisa.tensor_copy(exp_qk[:, nl.ds(vi * P, P)])

                # Transpose via identity matmul trick:
                # nc_matmul: stationary.T @ moving
                # attn_chunk[P,P].T @ I[P,P] = attn_chunk.T [P,P]
                attn_T_psum = nisa.nc_matmul(attn_chunk, id_sbuf)  # PSUM
                attn_T = nisa.tensor_copy(attn_T_psum)  # PSUM → SBUF

                # nc_matmul: attn_T[P,P].T @ V[P,d] = [P, d]
                pv_contrib_psum = nisa.nc_matmul(attn_T, v_tile)  # PSUM
                pv_contrib = nisa.tensor_copy(pv_contrib_psum)  # PSUM → SBUF

                # Accumulate in SBUF (in-place update to preserve scope)
                pv_accum[...] = nisa.tensor_tensor(pv_accum, pv_contrib, nl.add)

            # ── Phase 4: Normalize and store ──────────────────
            # Multiply by 1/sum (normalize)
            pv_normed = nisa.tensor_tensor(pv_accum, row_sum_recip, nl.multiply)

            # Cast to output dtype
            pv_out = nisa.tensor_copy(pv_normed, dtype=q.dtype)

            nisa.dma_copy(
                dst=out[nl.ds(q_start, P), batch_id, :d],
                src=pv_out
            )

    return out
