"""Cross-attention NKI kernel using nl.* (nki.language) return-style APIs.

Algorithm: single-pass flash attention for small seq_k (512 from T5).
Uses identity matmul trick for transpose in PV computation.
Accumulation done in SBUF.

API note: Uses nl.* return-style APIs exclusively.
nisa.dma_copy and nisa.nc_matmul are the only ISA calls retained
(dma_copy is dst-style with keyword args; nc_matmul returns PSUM).
"""
import nki
import nki.language as nl
import nki.isa as nisa
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

            # Load Q tile: [d=128, P=128]
            q_buf = nl.ndarray((d, P), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_buf, src=q[batch_id, :, nl.ds(q_start, P)])

            # ── Phase 1: QK^T — attention scores ──────────────
            qk_psum = nl.ndarray((P, seqlen_k), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(qk_psum, q_buf, k_buf)

            # Copy PSUM→SBUF and apply scale
            qk = nl.copy(qk_psum, dtype=nl.float32)
            qk = nl.multiply(qk, softmax_scale)

            # ── Phase 2: Numerically stable softmax ───────────
            # Row max
            row_max = nl.max(qk, axis=1, keepdims=True)

            # Subtract max: qk_shifted = qk - row_max
            qk_shifted = nl.subtract(qk, row_max)

            # Exp
            exp_qk = nl.exp(qk_shifted)

            # Row sum
            row_sum = nl.sum(exp_qk, axis=1, keepdims=True)

            # Reciprocal of row_sum for normalization
            row_sum_recip = nl.reciprocal(row_sum)

            # ── Phase 3: PV matmul ────────────────────────────
            pv_accum = nl.zeros((P, d), dtype=nl.float32)

            for vi in range(num_v_tiles):
                # Load V tile: [P=128, d=128]
                v_tile = nl.ndarray((P, d), dtype=v.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=v_tile, src=v[batch_id, nl.ds(vi * P, P), :])

                # Extract attention weights for this chunk (must be SBUF for nc_matmul)
                # 2-step: PSUM(f32) → SBUF(f32) → SBUF(bf16) [gen3 PSUM only supports f32]
                attn_chunk_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                attn_chunk_f32[...] = nl.copy(exp_qk[:, nl.ds(vi * P, P)], dtype=nl.float32)
                attn_chunk = nl.copy(attn_chunk_f32, dtype=nl.bfloat16)

                # Transpose via identity matmul trick:
                attn_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(attn_T_psum, attn_chunk, id_sbuf)
                # 2-step: PSUM(f32) → SBUF(f32) → SBUF(bf16)
                attn_T_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                attn_T_f32[...] = nl.copy(attn_T_psum, dtype=nl.float32)
                attn_T = nl.copy(attn_T_f32, dtype=nl.bfloat16)

                # nc_matmul: attn_T[P,P].T @ V[P,d] = [P, d]
                pv_contrib_psum = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(pv_contrib_psum, attn_T, v_tile)
                pv_contrib = nl.copy(pv_contrib_psum, dtype=nl.float32)

                # Accumulate in SBUF
                pv_accum[...] = nl.add(pv_accum, pv_contrib)

            # ── Phase 4: Normalize and store ──────────────────
            pv_normed = nl.multiply(pv_accum, row_sum_recip)

            # Cast to output dtype
            pv_out = nl.copy(pv_normed, dtype=q.dtype)

            nisa.dma_copy(
                dst=out[nl.ds(q_start, P), batch_id, :d],
                src=pv_out
            )

    return out
