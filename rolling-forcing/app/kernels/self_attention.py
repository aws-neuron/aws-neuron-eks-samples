"""Self-attention NKI kernel — mask-tensor + branchless online softmax.

Key design: NO if/else on section_i, NO Python list indexing with LoopVars.
Masking is done via a tensor passed by the caller.
Online softmax correction is always computed (init r_max=-inf makes it safe).

Call-site responsibilities:
    - Pad seq_q to multiple of 128
    - Build mask tensor: (128, seqlen_k) bf16, 0 for valid, -inf for invalid
    - Pass num_sections = seqlen_k // 8192 as Python int
    - Truncate output[:seq_q] after kernel returns

API note: Uses nl.* (nki.language) return-style APIs exclusively.
nisa.dma_copy and nisa.nc_matmul are the only ISA calls retained
(dma_copy is dst-style with keyword args; nc_matmul returns PSUM).
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def wan_flash_self_attn(q, k, v, identity, mask, softmax_scale=None,
                        num_sections=None, use_dynamic_loop=False):
    """Flash self-attention for Wan T2V DiT blocks.

    Args:
        q:        (bs, d, seq_q) bf16 — query, seq_q must be multiple of 128
        k:        (bs, d, seq_k) bf16 — key, seq_k must be multiple of 8192
        v:        (bs, seq_k, d) bf16 — value
        identity: (128, 128) bf16     — identity matrix for transpose trick
        mask:     (128, seq_k) bf16   — 0 for valid positions, -inf for masked
        softmax_scale: float          — 1/sqrt(head_dim)
        num_sections: int             — seqlen_k // 8192 (Python int)
        use_dynamic_loop: ignored

    Returns:
        out: (seq_q, bs, d) bf16
    """
    batch_size = q.shape[0]
    d = q.shape[1]
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]
    P = nl.tile_size.pmax  # 128

    SECTION = 8192
    tiles_512 = 16       # SECTION // 512
    tiles_128 = 64       # SECTION // P
    tiles_2048 = 4       # SECTION // 2048
    num_q_grps = seqlen_q // P

    # Output in HBM
    out = nl.ndarray((seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Identity matrix in SBUF (for transpose trick)
    id_sbuf = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for batch_id in nl.sequential_range(batch_size):

        # ── Initialize online softmax running state ──
        r_max = nl.ndarray((P, num_q_grps), dtype=nl.float32, buffer=nl.sbuf)
        r_sum = nl.ndarray((P, num_q_grps), dtype=nl.float32, buffer=nl.sbuf)
        pv_all = nl.ndarray((P, num_q_grps, d), dtype=nl.float32, buffer=nl.sbuf)

        for gi in range(num_q_grps):
            r_max[:, nl.ds(gi, 1)] = nl.full(
                (P, 1), fill_value=float('-inf'), dtype=nl.float32)
            r_sum[:, nl.ds(gi, 1)] = nl.zeros(
                (P, 1), dtype=nl.float32)
            pv_all[:, gi, :] = nl.zeros(
                (P, d), dtype=nl.float32)

        # ── Section loop (LoopVar — no Python list indexing!) ──
        for section_i in nl.sequential_range(num_sections):

            # Load K section: [d, 8192]
            k_sec = nl.ndarray((d, SECTION), dtype=k.dtype, buffer=nl.sbuf)
            for ti in range(tiles_512):
                ks = section_i * SECTION + ti * 512
                nisa.dma_copy(dst=k_sec[:, nl.ds(ti * 512, 512)],
                              src=k[batch_id, :, nl.ds(ks, 512)])

            # Load V section: 64 tiles of [128, 128]
            v_sec = nl.ndarray((P, tiles_128, d), dtype=v.dtype, buffer=nl.sbuf)
            for ti in range(tiles_128):
                vs = section_i * SECTION + ti * P
                nisa.dma_copy(dst=v_sec[:, ti, :],
                              src=v[batch_id, nl.ds(vs, P), :])

            # Load mask section: [128, 8192] bf16 → f32
            mask_sec = nl.ndarray((P, SECTION), dtype=nl.float32, buffer=nl.sbuf)
            for ti in range(tiles_512):
                ms = section_i * SECTION + ti * 512
                mask_tile = nl.ndarray((P, 512), dtype=mask.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=mask_tile, src=mask[:, nl.ds(ms, 512)])
                mask_sec[:, nl.ds(ti * 512, 512)] = nl.copy(mask_tile, dtype=nl.float32)

            for grp_i in range(num_q_grps):

                # Load Q tile [d, P]
                q_tile = nl.ndarray((d, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile,
                              src=q[batch_id, :, nl.ds(grp_i * P, P)])

                # ═══ Phase 1: QK^T × scale + mask ═══
                scores = nl.ndarray((P, SECTION), dtype=nl.float32, buffer=nl.sbuf)
                pmaxes = nl.ndarray((P, tiles_512), dtype=nl.float32, buffer=nl.sbuf)

                for ti in range(tiles_512):
                    qk_psum = nl.ndarray((P, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(qk_psum, q_tile, k_sec[:, nl.ds(ti * 512, 512)])
                    qk_sbuf = nl.copy(qk_psum, dtype=nl.float32)
                    qk_scaled = nl.multiply(qk_sbuf, softmax_scale)
                    # Add mask (0 for valid, -inf for invalid)
                    masked = nl.add(
                        qk_scaled, mask_sec[:, nl.ds(ti * 512, 512)])
                    scores[:, nl.ds(ti * 512, 512)] = masked
                    pmaxes[:, nl.ds(ti, 1)] = nl.max(masked, axis=1, keepdims=True)

                sec_max = nl.max(pmaxes, axis=1, keepdims=True)

                # ═══ Phase 2: Online softmax (ALWAYS — no if/else) ═══
                old_max = nl.copy(r_max[:, nl.ds(grp_i, 1)])
                new_max = nl.maximum(old_max, sec_max)
                corr_arg = nl.subtract(old_max, new_max)
                correction = nl.exp(corr_arg)
                r_max[:, nl.ds(grp_i, 1)] = new_max

                neg_max = nl.multiply(new_max, -1.0)

                exp_sc = nl.ndarray((P, SECTION), dtype=nl.bfloat16, buffer=nl.sbuf)
                p_sums = nl.ndarray((P, tiles_2048), dtype=nl.float32, buffer=nl.sbuf)

                for si in range(tiles_2048):
                    chunk = scores[:, nl.ds(si * 2048, 2048)]
                    shifted = nl.add(chunk, neg_max)
                    exp_f32 = nl.exp(shifted)
                    exp_sc[:, nl.ds(si * 2048, 2048)] = nl.copy(exp_f32, dtype=nl.bfloat16)
                    p_sums[:, nl.ds(si, 1)] = nl.sum(exp_f32, axis=1, keepdims=True)

                sec_sum = nl.sum(p_sums, axis=1, keepdims=True)

                # Update running sum: r_sum = r_sum * correction + sec_sum
                old_sum = nl.copy(r_sum[:, nl.ds(grp_i, 1)])
                scaled_sum = nl.multiply(old_sum, correction)
                r_sum[:, nl.ds(grp_i, 1)] = nl.add(
                    scaled_sum, sec_sum)

                # ═══ Phase 3: Transpose + PV matmul ═══
                pv_acc = nl.zeros((P, d), dtype=nl.float32)

                for v_ti in range(tiles_128):
                    col = v_ti * P
                    attn_chunk = nl.copy(exp_sc[:, nl.ds(col, P)])
                    attn_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(attn_T_psum, attn_chunk, id_sbuf)
                    # 2-step: PSUM(f32) → SBUF(f32) → SBUF(bf16) [gen3 PSUM only supports f32]
                    attn_T_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                    attn_T_f32[...] = nl.copy(attn_T_psum, dtype=nl.float32)
                    attn_T = nl.copy(attn_T_f32, dtype=nl.bfloat16)
                    pv_psum = nl.ndarray((P, d), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv_psum, attn_T, v_sec[:, v_ti, :])
                    pv_tile = nl.copy(pv_psum, dtype=nl.float32)
                    pv_acc[...] = nl.add(pv_acc, pv_tile)

                # ═══ Phase 4: Update running PV ═══
                old_pv = nl.copy(pv_all[:, grp_i, :])
                scaled_pv = nl.multiply(old_pv, correction)
                pv_all[:, grp_i, :] = nl.add(
                    scaled_pv, pv_acc)

        # ── After all sections: normalize and store ──
        for grp_i in range(num_q_grps):
            rcp = nl.reciprocal(r_sum[:, nl.ds(grp_i, 1)])
            pv_normed = nl.multiply(pv_all[:, grp_i, :], rcp)
            pv_out = nl.copy(pv_normed, dtype=q.dtype)
            nisa.dma_copy(
                dst=out[nl.ds(grp_i * P, P), batch_id, :d],
                src=pv_out)

    return out
