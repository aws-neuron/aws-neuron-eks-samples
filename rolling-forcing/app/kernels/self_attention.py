"""Self-attention NKI kernel — mask-tensor + branchless online softmax.

Key design: NO if/else on section_i, NO Python list indexing with LoopVars.
Masking is done via a tensor passed by the caller.
Online softmax correction is always computed (init r_max=-inf makes it safe).

Call-site responsibilities:
    - Pad seq_q to multiple of 128
    - Build mask tensor: (128, seqlen_k) bf16, 0 for valid, -inf for invalid
    - Pass num_sections = seqlen_k // 8192 as Python int
    - Truncate output[:seq_q] after kernel returns
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


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
            r_max[:, nl.ds(gi, 1)] = nisa.memset(
                (P, 1), value=float('-inf'), dtype=nl.float32)
            r_sum[:, nl.ds(gi, 1)] = nisa.memset(
                (P, 1), value=0.0, dtype=nl.float32)
            pv_all[:, gi, :] = nisa.memset(
                (P, d), value=0.0, dtype=nl.float32)

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
                mask_sec[:, nl.ds(ti * 512, 512)] = nisa.tensor_copy(
                    mask_tile, dtype=nl.float32)

            for grp_i in range(num_q_grps):

                # Load Q tile [d, P]
                q_tile = nl.ndarray((d, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_tile,
                              src=q[batch_id, :, nl.ds(grp_i * P, P)])

                # ═══ Phase 1: QK^T × scale + mask ═══
                scores = nl.ndarray((P, SECTION), dtype=nl.float32, buffer=nl.sbuf)
                pmaxes = nl.ndarray((P, tiles_512), dtype=nl.float32, buffer=nl.sbuf)

                for ti in range(tiles_512):
                    qk_psum = nisa.nc_matmul(q_tile, k_sec[:, nl.ds(ti * 512, 512)])
                    qk_sbuf = nisa.tensor_copy(qk_psum)
                    qk_scaled = nisa.tensor_scalar(qk_sbuf, nl.multiply, softmax_scale)
                    # Add mask (0 for valid, -inf for invalid)
                    masked = nisa.tensor_tensor(
                        qk_scaled, mask_sec[:, nl.ds(ti * 512, 512)], nl.add)
                    scores[:, nl.ds(ti * 512, 512)] = masked
                    pmaxes[:, nl.ds(ti, 1)] = nisa.tensor_reduce(
                        nl.maximum, masked, axis=1)

                sec_max = nisa.tensor_reduce(nl.maximum, pmaxes, axis=1)

                # ═══ Phase 2: Online softmax (ALWAYS — no if/else) ═══
                old_max = nisa.tensor_copy(r_max[:, nl.ds(grp_i, 1)])
                new_max = nisa.tensor_tensor(old_max, sec_max, nl.maximum)
                corr_arg = nisa.tensor_tensor(old_max, new_max, nl.subtract)
                correction = nisa.activation(nl.exp, corr_arg)
                r_max[:, nl.ds(grp_i, 1)] = new_max

                neg_max = nisa.tensor_scalar(new_max, nl.multiply, -1.0)

                exp_sc = nl.ndarray((P, SECTION), dtype=nl.bfloat16, buffer=nl.sbuf)
                p_sums = nl.ndarray((P, tiles_2048), dtype=nl.float32, buffer=nl.sbuf)

                for si in range(tiles_2048):
                    chunk = scores[:, nl.ds(si * 2048, 2048)]
                    shifted = nisa.tensor_tensor(chunk, neg_max, nl.add)
                    exp_f32 = nisa.activation(nl.exp, shifted)
                    exp_sc[:, nl.ds(si * 2048, 2048)] = nisa.tensor_copy(
                        exp_f32, dtype=nl.bfloat16)
                    p_sums[:, nl.ds(si, 1)] = nisa.tensor_reduce(
                        nl.add, exp_f32, axis=1)

                sec_sum = nisa.tensor_reduce(nl.add, p_sums, axis=1)

                # Update running sum: r_sum = r_sum * correction + sec_sum
                old_sum = nisa.tensor_copy(r_sum[:, nl.ds(grp_i, 1)])
                scaled_sum = nisa.tensor_tensor(old_sum, correction, nl.multiply)
                r_sum[:, nl.ds(grp_i, 1)] = nisa.tensor_tensor(
                    scaled_sum, sec_sum, nl.add)

                # ═══ Phase 3: Transpose + PV matmul ═══
                pv_acc = nisa.memset((P, d), value=0.0, dtype=nl.float32)

                for v_ti in range(tiles_128):
                    col = v_ti * P
                    attn_chunk = nisa.tensor_copy(exp_sc[:, nl.ds(col, P)])
                    attn_T_psum = nisa.nc_matmul(attn_chunk, id_sbuf)
                    attn_T = nisa.tensor_copy(attn_T_psum)
                    pv_psum = nisa.nc_matmul(attn_T, v_sec[:, v_ti, :])
                    pv_tile = nisa.tensor_copy(pv_psum)
                    pv_acc[...] = nisa.tensor_tensor(pv_acc, pv_tile, nl.add)

                # ═══ Phase 4: Update running PV ═══
                old_pv = nisa.tensor_copy(pv_all[:, grp_i, :])
                scaled_pv = nisa.tensor_tensor(old_pv, correction, nl.multiply)
                pv_all[:, grp_i, :] = nisa.tensor_tensor(
                    scaled_pv, pv_acc, nl.add)

        # ── After all sections: normalize and store ──
        for grp_i in range(num_q_grps):
            rcp = nisa.reciprocal(r_sum[:, nl.ds(grp_i, 1)])
            pv_normed = nisa.tensor_tensor(pv_all[:, grp_i, :], rcp, nl.multiply)
            pv_out = nisa.tensor_copy(pv_normed, dtype=q.dtype)
            nisa.dma_copy(
                dst=out[nl.ds(grp_i * P, P), batch_id, :d],
                src=pv_out)

    return out
