import neuronxcc.nki as nki
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor
from nki.compiler.kernel_builder import isa as nisa
from nki.compiler.kernel_builder.target_info import get_target_info


@nki.jit
def wan_flash_self_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    identity: Tensor,
    softmax_scale: float = None,
    actual_seqlen_k: int = None,
    use_dynamic_loop: bool = False,
):
    """Flash Attention Forward kernel using kernel_builder API.

    IO tensor layouts:
        - q: shape   (bs, d, seq_q)
        - k: shape   (bs, d, seq_k)
        - v: shape   (bs, seq_v, d)
        - identity: shape (128, 128) - identity matrix for transpose
        - out: shape (seq_q, bs, d) - output tensor, dtype matches q
    """
    # Detect target architecture
    target_info = get_target_info()
    trn1 = target_info.name == "trn1"
    dma_engine = nisa.engine.Gpsimd if trn1 else nisa.engine.Sync

    batch_size = q.shape[0]
    d = q.shape[1]
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]
    if actual_seqlen_k is None:
        actual_seqlen_k = seqlen_k
    section_len = 8192

    out = nb.ndarray((seqlen_q, batch_size, d), q.dtype, memspace=nb.shared_hbm)
    accum_buf = nb.ndarray((batch_size, seqlen_q, d), nb.float32, memspace=nb.hbm)
    sb_p = 128
    num_grps = (seqlen_q + sb_p - 1) // sb_p
    num_sections = seqlen_k // section_len

    num_2048_tiles_per_section = section_len // 2048
    num_512_tiles_per_section = section_len // 512
    num_128_tiles_per_section = section_len // 128

    # Private scratch pad for fp32 accumulation across sections
    result = accum_buf

    # Outer scope allocations (num_buffers=1 because accessed outside any pipelined loop)
    identity_load = nb.ndarray((128, 128), nb.bfloat16, name="identity_load")
    nisa.dma_copy(
        dst=identity_load,
        src=identity,
        name="load_identity",
        engine=dma_engine,
    )

    zero_bias_tensor = nb.ndarray((128, 1), nb.float32, name="zero_bias_tensor")
    nisa.memset(dst=zero_bias_tensor, value=0.0, name="init_zero_bias")

    def body(batch_id):
        # Section-pipelined tiles (accumulate data across grp_i iterations)
        # Has loop carried dependencies, cannot be buffered/pipelined
        running_max = nb.ndarray((sb_p, num_grps), nb.float32, name="running_max")
        running_sum = nb.ndarray((sb_p, num_grps), nb.float32, name="running_sum")
        div_25_sbuf = nb.ndarray((sb_p, num_grps), nb.float32, name="div_25_sbuf")

        for section_i in range(num_sections):
            num_2048_tiles_cur_section = num_2048_tiles_per_section
            num_512_tiles_cur_section = num_512_tiles_per_section

            # k_loaded - pipelined on section_i
            k_loaded = nb.ndarray(
                (d, num_512_tiles_per_section, 512),
                nb.bfloat16,
                num_buffers=2,
                name="k_loaded",
            )

            # Load k tiles for this section
            for tile_i in range(num_512_tiles_per_section):
                nisa.dma_copy(
                    dst=k_loaded[:, tile_i, :],
                    src=k[
                        batch_id,
                        :,
                        nb.ds(section_len * section_i + 512 * tile_i, 512),
                    ],
                    name=f"load_k[section={section_i}][tile={tile_i}]",
                )

            # v_loaded - pipelined on section_i
            v_loaded = nb.ndarray(
                (sb_p, num_128_tiles_per_section, d),
                nb.bfloat16,
                num_buffers=2,
                name="v_loaded",
            )

            # Load v tiles for this section
            for tile_i in range(num_128_tiles_per_section):
                nisa.dma_copy(
                    dst=v_loaded[:, tile_i, :],
                    src=v[
                        batch_id,
                        nb.ds(section_len * section_i + tile_i * 128, 128),
                        :,
                    ],
                    name=f"load_v[section={section_i}][tile={tile_i}]",
                )

            q_loaded_shape_p = d
            q_loaded_shape_n = sb_p

            reduce14_num_parts = 128
            num_blks = num_512_tiles_cur_section

            n_q = sb_p
            psum_n = sb_p * 4  # (128, 512)
            sbuf_n = psum_n * 4  # (128, 2048)

            exp_inst_elems = 2048
            exp_insts = 2048 // exp_inst_elems

            num_tps = exp_inst_elems // 128
            num_tp_grps = num_tps // 4
            num_tps_in_grp = 4
            n_per_part = num_tps_in_grp * 128

            # Main compute over groups - pipelined with all group-scope tiles
            for grp_i in range(num_grps):
                current_q_size = min(sb_p, seqlen_q - grp_i * sb_p)

                q_loaded = nb.ndarray(
                    (q_loaded_shape_p, q_loaded_shape_n),
                    nb.bfloat16,
                    num_buffers=3,
                    name="q_loaded",
                )
                if current_q_size < sb_p:
                    nisa.memset(dst=q_loaded, value=0.0)
                nisa.dma_copy(
                    dst=q_loaded[:, :current_q_size],
                    src=q[batch_id, :, nb.ds(grp_i * sb_p, current_q_size)],
                    name=f"load_q[section={section_i}][grp={grp_i}]",
                )

                mhlo_mul_2 = nb.ndarray(
                    (128, num_2048_tiles_cur_section, sbuf_n),
                    nb.float32,
                    num_buffers=1 if trn1 else 2,
                    name="qk_scores_scaled",
                )
                temp_reduce14_sbuf = nb.ndarray(
                    (reduce14_num_parts, num_blks),
                    nb.float32,
                    num_buffers=3,
                    name="qk_partial_max",
                )

                # LOOP 1: Populate temp_reduce14_sbuf with QK matmul results
                with nb.compiler.perfetto_group(
                    "QKt", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    for si in range(num_2048_tiles_cur_section):
                        mm1_psum_dot = nb.ndarray(
                            (128, psum_n),
                            nb.float32,
                            memspace=nb.psum,
                            num_buffers=2,
                            name="qk_matmul_psum",
                        )

                        for pi in range(4):
                            loc_512_tile_i = si * 4 + pi

                            nisa.matmul(
                                dst=mm1_psum_dot,
                                stationary=q_loaded,
                                moving=k_loaded[:, loc_512_tile_i, :],
                                accum=False,
                                name=f"qk_mm[section={section_i}][grp={grp_i}][si={si}][pi={pi}]",
                            )

                            nisa.tensor_scalar_cache_reduce(
                                dst=mhlo_mul_2[:, si, nb.ts(pi, 512)],
                                reduce_res=temp_reduce14_sbuf[:, nb.ds(si * 4 + pi, 1)],
                                src=mm1_psum_dot,
                                operand0=softmax_scale,
                                op0=nisa.arith_op.Multiply,
                                reduce_op=nisa.arith_op.Max,
                                name=f"qk_scale_max[section={section_i}][grp={grp_i}][si={si}][pi={pi}]",
                            )

                # Mask padded positions after QK matmul
                if actual_seqlen_k < seqlen_k:
                    section_base = section_i * section_len
                    for si in range(num_2048_tiles_per_section):
                        for pi in range(4):
                            global_start = section_base + si * 2048 + pi * 512
                            global_end = global_start + 512

                            if global_end <= actual_seqlen_k:
                                pass  # fully valid
                            elif global_start >= actual_seqlen_k:
                                nisa.memset(
                                    dst=mhlo_mul_2[:, si, nb.ds(pi * 512, 512)],
                                    value=float("-inf"),
                                )
                                nisa.memset(
                                    dst=temp_reduce14_sbuf[:, nb.ds(si * 4 + pi, 1)],
                                    value=float("-inf"),
                                )
                            else:
                                valid = actual_seqlen_k - global_start
                                nisa.memset(
                                    dst=mhlo_mul_2[
                                        :,
                                        si,
                                        nb.ds(pi * 512 + valid, 512 - valid),
                                    ],
                                    value=float("-inf"),
                                )
                                recomp_max = nb.ndarray(
                                    (128, 1),
                                    nb.float32,
                                    num_buffers=3,
                                    name="recomp_max",
                                )
                                nisa.tensor_reduce_arith(
                                    dst=recomp_max,
                                    src=mhlo_mul_2[:, si, nb.ds(pi * 512, 512)],
                                    op=nisa.arith_op.Max,
                                    num_r_dim=1,
                                )
                                nisa.tensor_copy(
                                    dst=temp_reduce14_sbuf[:, nb.ds(si * 4 + pi, 1)],
                                    src=recomp_max,
                                )

                with nb.compiler.perfetto_group(
                    "max_updates", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    final_reduce_max = nb.ndarray(
                        (128, 1), nb.float32, num_buffers=3, name="final_max"
                    )
                    nisa.tensor_reduce_arith(
                        dst=final_reduce_max,
                        src=temp_reduce14_sbuf,
                        op=nisa.arith_op.Max,
                        num_r_dim=1,
                        negated=True,
                        name=f"reduce_max[section={section_i}][grp={grp_i}]",
                    )

                    # Update running_max and scaling_factor
                    if section_i == 0:
                        nisa.tensor_copy(
                            dst=running_max[:, nb.ds(grp_i, 1)],
                            src=final_reduce_max,
                            name=f"init_running_max[section={section_i}][grp={grp_i}]",
                        )
                    else:
                        old_max_tile = nb.ndarray_like(final_reduce_max)
                        new_max_tile = nb.ndarray_like(final_reduce_max)
                        nisa.tensor_copy(
                            dst=old_max_tile,
                            src=running_max[:, nb.ds(grp_i, 1)],
                            name=f"load_old_max[section={section_i}][grp={grp_i}]",
                        )
                        nisa.tensor_copy(
                            dst=new_max_tile,
                            src=final_reduce_max,
                            name=f"load_new_max[section={section_i}][grp={grp_i}]",
                        )

                        prev_running_max = nb.ndarray_like(final_reduce_max)
                        nisa.activation(
                            dst=prev_running_max,
                            src=old_max_tile,
                            scale=-1.0,
                            bias=zero_bias_tensor,
                            op=nisa.activation_function.copy,
                            name=f"negate_old_max[section={section_i}][grp={grp_i}]",
                        )

                        combined_max_tile = nb.ndarray_like(final_reduce_max)
                        nisa.tensor_tensor_arith(
                            dst=combined_max_tile,
                            lhs=old_max_tile,
                            rhs=new_max_tile,
                            op=nisa.arith_op.Min,
                            name=f"combine_max[section={section_i}][grp={grp_i}]",
                        )

                        nisa.tensor_copy(
                            dst=running_max[:, nb.ds(grp_i, 1)],
                            src=combined_max_tile,
                            name=f"update_running_max[section={section_i}][grp={grp_i}]",
                        )

                        bias = nb.ndarray_like(final_reduce_max)
                        nisa.tensor_copy(
                            dst=bias,
                            src=combined_max_tile,
                            name=f"copy_bias[section={section_i}][grp={grp_i}]",
                        )

                        scaling_factor = nb.ndarray_like(final_reduce_max)
                        nisa.activation(
                            dst=scaling_factor,
                            src=prev_running_max,
                            bias=bias,
                            scale=1.0,
                            op=nisa.activation_function.exp,
                            name=f"compute_scale[section={section_i}][grp={grp_i}]",
                        )

                with nb.compiler.perfetto_group(
                    "softmax", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    exp6_sbuf = nb.ndarray(
                        (128, num_2048_tiles_cur_section, sbuf_n),
                        nb.bfloat16,
                        num_buffers=2,
                        name="attention_weights",
                    )
                    final_reduce_sum_b = nb.ndarray(
                        (128, section_len // exp_inst_elems),
                        nb.float32,
                        num_buffers=3,
                        name="partial_sum",
                    )

                    # LOOP 2: Compute exp(scores) with updated running_max
                    for si in range(num_2048_tiles_cur_section):
                        for pi in range(exp_insts):
                            bias_vec = nb.ndarray(
                                (128, 1), nb.float32, num_buffers=2, name="exp_bias"
                            )
                            nisa.tensor_copy(
                                dst=bias_vec,
                                src=running_max[:, nb.ds(grp_i, 1)],
                                name=f"load_max_bias[section={section_i}][grp={grp_i}][si={si}][pi={pi}]",
                            )
                            nisa.activation(
                                dst=exp6_sbuf[:, si, :],
                                reduce_res=final_reduce_sum_b[
                                    :, nb.ds(si * exp_insts + pi, 1)
                                ],
                                src=mhlo_mul_2[:, si, :],
                                bias=bias_vec,
                                scale=1.0,
                                op=nisa.activation_function.exp,
                                reduce_op=nisa.activation_reduce_op.Add,
                                name=f"softmax_exp[section={section_i}][grp={grp_i}][si={si}][pi={pi}]",
                            )

                with nb.compiler.perfetto_group(
                    "transpose", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    tp_sbuf = nb.ndarray(
                        (128, num_2048_tiles_cur_section, num_tp_grps, n_per_part),
                        nb.bfloat16,
                        num_buffers=2,
                        name="attention_weights_transposed",
                    )

                    # LOOP 3: Transpose operations
                    for si in range(num_2048_tiles_cur_section):
                        tp_psum_tile = nb.ndarray(
                            (128, 128 * num_tps_in_grp),
                            nb.float32,
                            memspace=nb.psum,
                            num_buffers=3,
                            name="transpose_psum",
                        )

                        for tp_grp in range(num_tp_grps):
                            for ti in range(num_tps_in_grp):
                                nisa.matmul(
                                    dst=tp_psum_tile[:, nb.ts(ti, 128)],
                                    stationary=exp6_sbuf[
                                        :,
                                        si,
                                        nb.ds(tp_grp * n_per_part + ti * 128, 128),
                                    ],
                                    moving=identity_load,
                                    name=f"transpose_mm[section={section_i}][grp={grp_i}][si={si}][tp_grp={tp_grp}][ti={ti}]",
                                )
                            if tp_grp % 2 == 0 and not trn1:
                                nisa.tensor_copy(
                                    dst=tp_sbuf[:, si, tp_grp, :],
                                    src=tp_psum_tile,
                                    name=f"copy_transposed_scalar[section={section_i}][grp={grp_i}][si={si}][tp_grp={tp_grp}]",
                                    engine=nisa.engine.Scalar,
                                )
                            else:
                                nisa.tensor_copy(
                                    dst=tp_sbuf[:, si, tp_grp, :],
                                    src=tp_psum_tile,
                                    name=f"copy_transposed_vector[section={section_i}][grp={grp_i}][si={si}][tp_grp={tp_grp}]",
                                    engine=nisa.engine.Vector,
                                )

                with nb.compiler.perfetto_group(
                    "running_sum", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    temp_reduce_sum = nb.ndarray_like(final_reduce_max)
                    nisa.tensor_reduce_arith(
                        dst=temp_reduce_sum,
                        src=final_reduce_sum_b,
                        op=nisa.arith_op.Add,
                        num_r_dim=1,
                        name=f"reduce_sum[section={section_i}][grp={grp_i}]",
                    )

                    final_reduce_sum_b_collect = nb.ndarray_like(temp_reduce_sum)
                    nisa.tensor_copy(
                        dst=final_reduce_sum_b_collect,
                        src=temp_reduce_sum,
                        name=f"collect_sum[section={section_i}][grp={grp_i}]",
                    )

                    # Update running_sum
                    if section_i == 0:
                        nisa.tensor_copy(
                            dst=running_sum[:, nb.ds(grp_i, 1)],
                            src=final_reduce_sum_b_collect,
                            name=f"init_running_sum[section={section_i}][grp={grp_i}]",
                        )
                    if section_i > 0:
                        prev_running_sum = nb.ndarray_like(final_reduce_sum_b_collect)
                        nisa.tensor_copy(
                            dst=prev_running_sum,
                            src=running_sum[:, nb.ds(grp_i, 1)],
                            name=f"load_prev_sum[section={section_i}][grp={grp_i}]",
                        )
                        nisa.scalar_tensor_tensor_arith(
                            dst=running_sum[:, nb.ds(grp_i, 1)],
                            src0=prev_running_sum,
                            src1=final_reduce_sum_b_collect,
                            imm0=scaling_factor,
                            op0=nisa.arith_op.Multiply,
                            op1=nisa.arith_op.Add,
                            name=f"update_running_sum[section={section_i}][grp={grp_i}]",
                        )
                    if section_i == num_sections - 1:
                        nisa.activation(
                            dst=div_25_sbuf[:, nb.ds(grp_i, 1)],
                            src=running_sum[:, nb.ds(grp_i, 1)],
                            op=nisa.activation_function.reciprocal,
                            bias=zero_bias_tensor,
                            scale=1.0,
                            name=f"compute_reciprocal[section={section_i}][grp={grp_i}]",
                        )

                with nb.compiler.perfetto_group(
                    "PV", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    mm2_sbuf = nb.ndarray(
                        (sb_p, d), nb.float32, num_buffers=3, name="pv_accumulator"
                    )

                    for mm2i in range(num_2048_tiles_cur_section):
                        mm2_psum = nb.ndarray(
                            (sb_p, d),
                            nb.float32,
                            memspace=nb.psum,
                            num_buffers=3,
                            name="pv_matmul_psum",
                        )

                        num_tp_grps_in_2048_tile = 4
                        for tp_grp_i in range(num_tp_grps_in_2048_tile):
                            mm2_num_grps = 4
                            for mm2_si in range(mm2_num_grps):
                                v_tile_idx = mm2i * 16 + tp_grp_i * 4 + mm2_si
                                is_first = tp_grp_i == 0 and mm2_si == 0

                                nisa.matmul(
                                    dst=mm2_psum,
                                    stationary=tp_sbuf[
                                        :, mm2i, tp_grp_i, nb.ts(mm2_si, 128)
                                    ],
                                    moving=v_loaded[:, v_tile_idx, :],
                                    accum=not is_first,
                                    name=f"pv_mm[section={section_i}][grp={grp_i}][mm2i={mm2i}][tp_grp={tp_grp_i}][si={mm2_si}]",
                                )

                        if mm2i == 0:
                            nisa.tensor_copy(
                                dst=mm2_sbuf,
                                src=mm2_psum,
                                name=f"load_pv_acc[section={section_i}][grp={grp_i}][mm2i={mm2i}]",
                            )
                        else:
                            nisa.tensor_tensor_arith(
                                dst=mm2_sbuf,
                                lhs=mm2_sbuf,
                                rhs=mm2_psum,
                                op=nisa.arith_op.Add,
                                name=f"accumulate_pv[section={section_i}][grp={grp_i}][mm2i={mm2i}]",
                            )

                with nb.compiler.perfetto_group(
                    "store", loop_vars={"section_i": section_i, "grp_i": grp_i}
                ):
                    # Accumulate PV across sections
                    if section_i == 0:
                        current_pv = mm2_sbuf
                    else:
                        prev_output = nb.ndarray(
                            (128, d), result.dtype, num_buffers=3, name="prev_output"
                        )
                        if current_q_size < sb_p:
                            nisa.memset(dst=prev_output, value=0.0)
                        nisa.dma_copy(
                            dst=prev_output[:current_q_size, :],
                            src=result[
                                batch_id,
                                nb.ds(grp_i * sb_p, current_q_size),
                                :d,
                            ],
                            name=f"load_prev_output[section={section_i}][grp={grp_i}]",
                            engine=dma_engine,
                        )
                        current_pv = nb.ndarray_like(prev_output)
                        nisa.scalar_tensor_tensor_arith(
                            dst=current_pv,
                            src0=prev_output,
                            src1=mm2_sbuf,
                            imm0=scaling_factor,
                            op0=nisa.arith_op.Multiply,
                            op1=nisa.arith_op.Add,
                            name=f"combine_outputs[section={section_i}][grp={grp_i}]",
                        )

                    # Store: normalize only on last section
                    if section_i < num_sections - 1:
                        nisa.dma_copy(
                            dst=result[
                                batch_id,
                                nb.ds(grp_i * sb_p, current_q_size),
                                :d,
                            ],
                            src=current_pv[:current_q_size, :],
                            name=f"store_result_intermediate[section={section_i}][grp={grp_i}]",
                            engine=dma_engine,
                        )
                    else:
                        mm2_div_sbuf = nb.ndarray((sb_p, d), q.dtype, name="pv_final_scaled")
                        nisa.activation(
                            dst=mm2_div_sbuf,
                            src=current_pv,
                            op=nisa.activation_function.copy,
                            bias=zero_bias_tensor,
                            scale=div_25_sbuf[:, nb.ds(grp_i, 1)],
                            name=f"apply_scale[section={section_i}][grp={grp_i}]",
                        )
                        nisa.dma_copy(
                            dst=out[
                                nb.ds(grp_i * sb_p, current_q_size),
                                batch_id,
                                :d,
                            ],
                            src=mm2_div_sbuf[:current_q_size, :],
                            name=f"store_result_final[section={section_i}][grp={grp_i}]",
                            engine=dma_engine,
                        )

    if use_dynamic_loop:
        loop_bound_sbuf = nb.ndarray((1, 1), nb.int32, name="loop_bound")
        nisa.memset(dst=loop_bound_sbuf, value=batch_size)  
        loop_bound = nisa.load_register(loop_bound_sbuf[0:1, :])
    else:
        loop_bound = batch_size
    nb.fori_loop(loop_bound, body)

    return out
