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

NKI API: bundled neuronxcc.nki.isa return-style.
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
            # Tile along seq_k in CHUNK=512 pieces to stay within nc_matmul limit
            # Build QK result (P, seqlen_k) by tiling both d and seq_k
            qk_acc = nisa.memset((P, seqlen_k), value=0.0, dtype=nl.float32)

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
                    # Moving operand is K_chunk[128,512] — within 512 limit ✓
                    qk_psum = nisa.nc_matmul(q_buf, k_chunk)
                    qk_sbuf = nl.copy(qk_psum)

                    # Accumulate into the right seq_k columns
                    qk_slice = nl.copy(qk_acc[:, nl.ds(sk_off, CHUNK)])
                    qk_updated = nisa.tensor_tensor(qk_slice, qk_sbuf, nl.add)
                    # Write back (in-place update via copy)
                    nisa.dma_copy(dst=qk_acc[:, nl.ds(sk_off, CHUNK)], src=qk_updated)

            # Scale
            qk_scaled = nisa.tensor_scalar(qk_acc, nl.multiply, softmax_scale)

            # ── Phase 2: Softmax ──
            # Row max in 512 chunks
            pmaxes = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * CHUNK
                pmaxes[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.maximum, qk_scaled[:, nl.ds(sc_start, CHUNK)], axis=1)
            row_max = nisa.tensor_reduce(nl.maximum, pmaxes, axis=1)

            # Subtract max and exp
            qk_shifted = nisa.tensor_tensor(qk_scaled, row_max, nl.subtract)
            exp_qk = nisa.activation(nl.exp, qk_shifted)

            # Row sum in 512 chunks
            psums = nl.ndarray((P, num_sk_chunks), dtype=nl.float32, buffer=nl.sbuf)
            for sc in range(num_sk_chunks):
                sc_start = sc * CHUNK
                psums[:, nl.ds(sc, 1)] = nisa.tensor_reduce(
                    nl.add, exp_qk[:, nl.ds(sc_start, CHUNK)], axis=1)
            row_sum = nisa.tensor_reduce(nl.add, psums, axis=1)
            row_sum_recip = nisa.reciprocal(row_sum)

            # Cast exp to bf16 for PV matmul
            exp_bf16 = nl.copy(exp_qk, dtype=nl.bfloat16)

            # ── Phase 3: PV matmul ──
            # attn @ V: [P, seq_k] @ [seq_k, d] → [P, d]
            # Tile over seq_k (P=128 chunks) for both attn weights and V
            pv_accum = nisa.memset((P, d), value=0.0, dtype=nl.float32)

            for vi in range(num_v_tiles):
                v_start = vi * P

                # Load V tile: [P, d]
                # d could be 256 (< 512) or 512 or 1024 — handle with CHUNK loads
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

                # Extract attn weights: [P, P] from exp_bf16
                attn_chunk = nl.copy(exp_bf16[:, nl.ds(v_start, P)])

                # Transpose via identity matmul trick
                attn_T_psum = nisa.nc_matmul(attn_chunk, id_sbuf)
                attn_T = nl.copy(attn_T_psum)

                # nc_matmul: attn_T[P,P].T @ V[P,d]
                # V has free dim = d, which could be 256/512/1024
                # 1024 > 512 limit! Need to tile V along d too.
                d_mat_chunks = d // CHUNK
                d_mat_rem = d % CHUNK
                for dmc in range(d_mat_chunks):
                    dmc_start = dmc * CHUNK
                    v_d_chunk = nl.copy(v_tile[:, nl.ds(dmc_start, CHUNK)])
                    pv_chunk = nisa.nc_matmul(attn_T, v_d_chunk)
                    pv_s = nl.copy(pv_chunk)
                    existing = nl.copy(pv_accum[:, nl.ds(dmc_start, CHUNK)])
                    updated = nisa.tensor_tensor(existing, pv_s, nl.add)
                    nisa.dma_copy(dst=pv_accum[:, nl.ds(dmc_start, CHUNK)], src=updated)

                if d_mat_rem > 0:
                    dmc_start_rem = d_mat_chunks * CHUNK
                    v_d_rem = nl.copy(v_tile[:, nl.ds(dmc_start_rem, d_mat_rem)])
                    pv_rem = nisa.nc_matmul(attn_T, v_d_rem)
                    pv_rem_s = nl.copy(pv_rem)
                    existing_rem = nl.copy(pv_accum[:, nl.ds(dmc_start_rem, d_mat_rem)])
                    updated_rem = nisa.tensor_tensor(existing_rem, pv_rem_s, nl.add)
                    nisa.dma_copy(dst=pv_accum[:, nl.ds(dmc_start_rem, d_mat_rem)], src=updated_rem)

            # ── Phase 4: Normalize and store ──
            pv_normed = nisa.tensor_tensor(pv_accum, row_sum_recip, nl.multiply)
            pv_out = nl.copy(pv_normed, dtype=q.dtype)

            nisa.dma_copy(
                dst=out[nl.ds(q_start, P), batch_id, :d],
                src=pv_out
            )

    return out
