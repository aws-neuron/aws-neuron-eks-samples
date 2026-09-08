"""NKI kernel for VAE AttentionBlock — single-head spatial self-attention.

The VAE AttentionBlock does:
  RMSNorm → Conv2d_1x1(QKV) → scaled_dot_product_attention → Conv2d_1x1(proj) + residual

This kernel handles the core SDPA part. The 1x1 convs use vae_conv2d_k1.

Input:  q (1, d, seq), k (1, d, seq), v (1, seq, d)
Output: (seq, 1, d)

IMPORTANT: All seq dims must be padded to multiple of 512 by the caller.
NKI requires compile-time constant sizes in nl.ds() — no variable-size slices.

Architecture constraints:
  - nc_matmul moving operand free dim ≤ 512
  - So QK^T must be tiled along seq_k in 512-token chunks

Production shapes:
  Decoder middle block: d=1024, seq=30*52=1560 → pad to 2048

NKI API: uses only nisa.dma_copy and nisa.nc_matmul from ISA;
all other ops use nl.* (nl.zeros, nl.copy, nl.add, nl.exp, etc.) — new SDK compatible.
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def vae_self_attention(q, k, v, identity, softmax_scale=None):
    """Single-head self-attention for VAE AttentionBlock.

    IO tensor layouts (matching wan_cross_attn convention):
        - q:        (1, d, seq_q)  single head, d=dim (e.g. 1024)
        - k:        (1, d, seq_k)  seq_k == seq_q for self-attention
        - v:        (1, seq_k, d)
        - identity: (128, 128)     used for transpose trick via nc_matmul
        - out:      (seq_q, 1, d)  output

    REQUIREMENT: seq_q and seq_k MUST be multiples of 512. Pad at call site.
    d MUST be a multiple of 128.

    nc_matmul moving operand limit: free dim ≤ 512.
    So we tile QK^T and PV along seq_k in P=128 chunks, loading K in 512-wide pieces.
    """
    batch_size = q.shape[0]  # 1 for VAE
    d = q.shape[1]           # 1024 (or 512, 256)
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]

    P = nl.tile_size.pmax    # 128
    CHUNK = 512              # DMA chunk size for seq loads

    num_q_grps = seqlen_q // P       # seq_q / 128
    num_v_tiles = seqlen_k // P      # seq_k / 128
    num_d_tiles = d // P             # d / 128
    num_sk_chunks = seqlen_k // CHUNK  # seq_k / 512

    # Output in HBM
    out = nl.ndarray((seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    # Load identity matrix
    id_sbuf = nl.ndarray((P, P), dtype=identity.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=id_sbuf, src=identity)

    for batch_id in range(batch_size):  # just 1 iteration for VAE

        for gi in range(num_q_grps):
            q_start = gi * P

            # ── Phase 1: QK^T = sum over d-tiles ──
            qk_acc = nl.zeros((P, seqlen_k), dtype=nl.float32)

            for dt in range(num_d_tiles):
                d_off = dt * P

                # Load Q tile: [P_d, P_seq]
                q_buf = nl.ndarray((P, P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=q_buf, src=q[batch_id, nl.ds(d_off, P), nl.ds(q_start, P)])

                # Tile K along seq_k in 512-wide chunks
                for sk_c in range(num_sk_chunks):
                    sk_off = sk_c * CHUNK

                    # Load K chunk: [P_d, 512]
                    k_chunk = nl.ndarray((P, CHUNK), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=k_chunk,
                                  src=k[batch_id, nl.ds(d_off, P), nl.ds(sk_off, CHUNK)])

                    # nc_matmul: Q[P,P].T @ K_chunk[P,512] → [P, 512]
                    qk_psum = nl.ndarray((P, CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(qk_psum, q_buf, k_chunk)
                    qk_sbuf = nl.copy(qk_psum, dtype=nl.float32)

                    # Accumulate into the right seq_k columns
                    qk_slice = nl.copy(qk_acc[:, nl.ds(sk_off, CHUNK)])
                    qk_updated = nl.add(qk_slice, qk_sbuf)
                    qk_acc[:, nl.ds(sk_off, CHUNK)] = qk_updated

            # Scale (nl.multiply handles tile × float; Python * operator not supported in NKI)
            qk_scaled = nl.multiply(qk_acc, softmax_scale)

            # ── Phase 2: Softmax (matches cross_attention.py pattern) ──
            # nl.max/nl.sum handle large free dims natively (512 limit is nc_matmul only)
            row_max = nl.max(qk_scaled, axis=1, keepdims=True)
            qk_shifted = nl.subtract(qk_scaled, row_max)
            exp_qk = nl.exp(qk_shifted)
            row_sum = nl.sum(exp_qk, axis=1, keepdims=True)
            row_sum_recip = nl.reciprocal(row_sum)

            # Cast exp to bf16 for PV matmul
            exp_bf16 = nl.copy(exp_qk, dtype=nl.bfloat16)

            # ── Phase 3: PV matmul ──
            # attn @ V: [P, seq_k] @ [seq_k, d] → [P, d]
            pv_accum = nl.zeros((P, d), dtype=nl.float32)

            for vi in range(num_v_tiles):
                v_start = vi * P

                # Load V tile: [P, d]
                v_tile = nl.ndarray((P, d), dtype=v.dtype, buffer=nl.sbuf)
                d_chunks = d // CHUNK
                d_rem = d % CHUNK
                for dc in range(d_chunks):
                    dc_start = dc * CHUNK
                    nisa.dma_copy(dst=v_tile[:, nl.ds(dc_start, CHUNK)],
                                  src=v[batch_id, nl.ds(v_start, P), nl.ds(dc_start, CHUNK)])
                if d_rem > 0:
                    dc_start_rem = d_chunks * CHUNK
                    nisa.dma_copy(dst=v_tile[:, nl.ds(dc_start_rem, d_rem)],
                                  src=v[batch_id, nl.ds(v_start, P), nl.ds(dc_start_rem, d_rem)])

                # Extract attn weights: [P, P] from exp_bf16 (must be SBUF for nc_matmul)
                attn_chunk_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                attn_chunk_f32[...] = nl.copy(exp_bf16[:, nl.ds(v_start, P)], dtype=nl.float32)
                attn_chunk = nl.copy(attn_chunk_f32, dtype=nl.bfloat16)

                # Transpose via identity matmul trick
                # Must pin attn_T to SBUF explicitly (nc_matmul stationary requires SBUF)
                attn_T_psum = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(attn_T_psum, attn_chunk, id_sbuf)
                attn_T_f32 = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                attn_T_f32[...] = nl.copy(attn_T_psum, dtype=nl.float32)
                attn_T = nl.copy(attn_T_f32, dtype=nl.bfloat16)

                # nc_matmul: attn_T[P,P].T @ V[P,d]
                # V has free dim = d, which could be 256/512/1024
                # 1024 > 512 limit! Need to tile V along d too.
                d_mat_chunks = d // CHUNK
                d_mat_rem = d % CHUNK
                for dmc in range(d_mat_chunks):
                    dmc_start = dmc * CHUNK
                    v_d_chunk = nl.copy(v_tile[:, nl.ds(dmc_start, CHUNK)])
                    pv_psum = nl.ndarray((P, CHUNK), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv_psum, attn_T, v_d_chunk)
                    pv_s = nl.copy(pv_psum, dtype=nl.float32)
                    existing = nl.copy(pv_accum[:, nl.ds(dmc_start, CHUNK)])
                    updated = nl.add(existing, pv_s)
                    pv_accum[:, nl.ds(dmc_start, CHUNK)] = updated

                if d_mat_rem > 0:
                    dmc_start_rem = d_mat_chunks * CHUNK
                    v_d_rem = nl.copy(v_tile[:, nl.ds(dmc_start_rem, d_mat_rem)])
                    pv_rem_psum = nl.ndarray((P, d_mat_rem), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(pv_rem_psum, attn_T, v_d_rem)
                    pv_rem_s = nl.copy(pv_rem_psum, dtype=nl.float32)
                    existing_rem = nl.copy(pv_accum[:, nl.ds(dmc_start_rem, d_mat_rem)])
                    updated_rem = nl.add(existing_rem, pv_rem_s)
                    pv_accum[:, nl.ds(dmc_start_rem, d_mat_rem)] = updated_rem

            # ── Phase 4: Normalize and store ──
            pv_normed = nl.multiply(pv_accum, row_sum_recip)
            pv_out = nl.copy(pv_normed, dtype=q.dtype)

            nisa.dma_copy(
                dst=out[nl.ds(q_start, P), batch_id, :d],
                src=pv_out
            )

    return out
