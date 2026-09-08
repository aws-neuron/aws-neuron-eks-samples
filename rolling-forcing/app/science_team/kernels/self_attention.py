# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Neuron Science Team, Amazon Annapurna Labs
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Any, Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa import engine

from kernels.nkilib_compat import assert_shape, kernel_assert
from kernels.nkilib_compat import PSUM_BANK_SIZE, div_ceil
from kernels.nkilib_compat import ModularAllocator
from kernels.nkilib_compat import TensorView

import nki
from torch_neuronx.nki_hop import wrap_nki


_FLOAT32_MIN = -3.4028235e38

_MAX_SEQLEN = 131072
_MAX_HEAD_DIM = 128

_Q_GRP_SZ = 128
_V_TILE_SZ = 128
_K_TILE_SZ = 512
_EXP_TILE_SZ = 512
_LARGE_TILE_SZ = 2048
_FLASH_ATTENTION_THRESHOLD = 10 * 1024
_FLASH_ATTENTION_SECTION_LENGTH = 8 * 1024


@wrap_nki
@nki.jit
def wan_flash_self_attn(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    softmax_scale: Optional[float] = None,
    actual_seqlen_k: Optional[int] = None,
    use_dynamic_loop: bool = False,
):
    batch_size, d, seqlen_q = q.shape
    batch_size_kv, _, seqlen_k = k.shape
    assert_shape(q, (batch_size, d, seqlen_q), "q")
    assert_shape(k, (batch_size_kv, d, seqlen_k), "k")
    assert_shape(v, (batch_size_kv, seqlen_k, d), "v")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    kernel_assert(
        batch_size_kv == batch_size,
        f"this kernel requires batch_size_kv == batch_size, got {batch_size=}, {batch_size_kv=}",
    )

    if actual_seqlen_k is None:
        actual_seqlen_k = seqlen_k
    kernel_assert(
        0 < actual_seqlen_k <= seqlen_k,
        f"actual_seqlen_k must be in (0, seqlen_k]={seqlen_k}, got {actual_seqlen_k}",
    )

    kernel_assert(seqlen_q <= _MAX_SEQLEN, f"seqlen_q={seqlen_q} exceeds {_MAX_SEQLEN}")
    kernel_assert(seqlen_k <= _MAX_SEQLEN, f"seqlen_k={seqlen_k} exceeds {_MAX_SEQLEN}")
    kernel_assert(d > 0 and d <= _MAX_HEAD_DIM, f"d must be in (0,{_MAX_HEAD_DIM}], got {d=}")

    result = nl.ndarray(shape=(seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        actual_seqlen_k=actual_seqlen_k,
        d=d,
        bs=batch_size,
        scale=softmax_scale,
        dtype=q.dtype,
    )

    use_flash_attn = actual_seqlen_k > _FLASH_ATTENTION_THRESHOLD
    if use_flash_attn:
        partial_out = nl.ndarray(
            shape=(batch_size, seqlen_q, d), dtype=nl.float32, buffer=nl.hbm,
            name="partial_out_fp32",
        )
    else:
        partial_out = result

    if use_dynamic_loop:
        batch_id_reg = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf, name="batch_id_reg")
        nisa.memset(batch_id_reg, value=0)
        for _ in nl.dynamic_range(0, batch_size):
            _wan_flash_attn_kernel_impl(q, k, v, result, partial_out, ac, batch_id_reg)
            nisa.tensor_scalar(
                dst=batch_id_reg, data=batch_id_reg, op0=nl.add, operand0=1
            )
    else:
        for batch_id in range(batch_size):
            _wan_flash_attn_kernel_impl(q, k, v, result, partial_out, ac, batch_id)

    return result


@dataclass
class AttnConfig(nl.NKIObject):
    seqlen_q: int = None
    seqlen_k: int = None
    actual_seqlen_k: int = None
    d: int = None
    bs: int = None
    scale: float = None
    dtype: Any = None


@dataclass
class AttnTileParams(nl.NKIObject):
    sb_p: int = None
    num_grps: int = None
    num_q_grps_per_load: int = None

    num_large_tiles_per_section: int = None
    num_k_tiles_per_section: int = None
    num_v_tiles_per_section: int = None

    exp_inst_elems: int = None
    num_exp_insts_per_large_tile: int = None
    num_tps_in_mm2_grp: int = None
    mm2_grp_sz: int = None

    section_len: int = None
    num_sections: int = None


@dataclass
class SectionParams(nl.NKIObject):
    section_idx = None
    section_offset = None


@dataclass
class AttnInternalBuffers(nl.NKIObject):

    q_sb = None
    k_sb = None
    v_sb = None

    mm1_psum = None
    mm1_masked = None
    mm1_partial_max = None
    mm1_section_max = None
    mm1_running_max = None
    prev_mm1_running_max = None
    flash_attn_correction_factor = None

    exp_sb = None
    exp_partial_sum = None
    exp_section_sum = None
    exp_tp_sb = None
    exp_running_sum = None
    exp_sum_reciprocal = None

    mm2_psum = None
    mm2_sb = None
    mm2_prev_output = None
    mm2_accum_flash_attn = None
    mm2_final = None

    zero_bias_tensor = None


def _ap_with_batch(tensor, pattern, offset, batch_off):
    if isinstance(batch_off, int):
        stride0 = 1
        for s in tensor.shape[1:]:
            stride0 *= s
        return tensor.ap(pattern=pattern, offset=offset + batch_off * stride0)
    return tensor.ap(
        pattern=pattern,
        offset=offset,
        scalar_offset=batch_off,
        indirect_dim=0,
    )


def _compute_tile_parameters(ac: AttnConfig) -> AttnTileParams:
    atp = AttnTileParams()

    atp.sb_p = nl.tile_size.pmax
    kernel_assert(_Q_GRP_SZ == atp.sb_p, f"expect _Q_GRP_SZ == sb_p, got {_Q_GRP_SZ=}, {atp.sb_p=}")
    kernel_assert(_V_TILE_SZ == atp.sb_p, f"expect _V_TILE_SZ == sb_p, got {_V_TILE_SZ=}, {atp.sb_p=}")

    atp.num_grps = div_ceil(ac.seqlen_q, atp.sb_p)
    num_q_grps_per_load_dtype = 4 if ac.dtype == nl.float32 else 8
    atp.num_q_grps_per_load = min(num_q_grps_per_load_dtype, atp.num_grps)

    total_seqlen_k = ac.actual_seqlen_k
    use_flash_attn = total_seqlen_k > _FLASH_ATTENTION_THRESHOLD
    atp.section_len = (
        min(total_seqlen_k, _FLASH_ATTENTION_SECTION_LENGTH) if use_flash_attn else total_seqlen_k
    )
    atp.num_sections = div_ceil(total_seqlen_k, atp.section_len)
    if not use_flash_attn:
        kernel_assert(atp.num_sections == 1, "must only have 1 section if not using flash_attn")

    atp.num_large_tiles_per_section = div_ceil(atp.section_len, _LARGE_TILE_SZ)
    atp.num_k_tiles_per_section = div_ceil(atp.section_len, _K_TILE_SZ)
    atp.num_v_tiles_per_section = div_ceil(atp.section_len, _V_TILE_SZ)

    atp.exp_inst_elems = _EXP_TILE_SZ
    atp.num_exp_insts_per_large_tile = _LARGE_TILE_SZ // atp.exp_inst_elems
    atp.num_tps_in_mm2_grp = _K_TILE_SZ // atp.sb_p
    atp.mm2_grp_sz = _K_TILE_SZ

    return atp


def _wan_flash_attn_kernel_impl(q, k, v, o, partial_out, ac: AttnConfig, batch_off):
    atp = _compute_tile_parameters(ac)

    allocator = ModularAllocator(initial_address=0)
    bufs = AttnInternalBuffers()

    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)

    bufs.mm1_running_max = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    bufs.exp_running_sum = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, atp.num_grps), dtype=nl.float32)

    sbuf_addr_outer = allocator.get_current_address()

    for section_idx in range(atp.num_sections):
        sp = SectionParams(
            section_idx=section_idx,
            section_offset=atp.section_len * section_idx,
        )

        allocator.set_current_address(sbuf_addr_outer)
        _allocate_attention_buffers(allocator, ac, atp, bufs)
        sbuf_addr = allocator.get_current_address()

        _load_k_tile(k, bufs.k_sb, sp, atp.num_k_tiles_per_section, batch_off)
        _load_v_tile(v, bufs.v_sb, sp, atp.num_v_tiles_per_section, batch_off)

        if atp.num_grps <= 1:
            _load_q_impl(0, ac, atp, sp, bufs, q, sbuf_addr, batch_off)
            _qk_and_max_impl(0, ac, atp, sp, bufs)
            _update_max_impl(0, ac, atp, sp, bufs)
            _exp_impl(0, ac, atp, sp, bufs)
            _pv_impl(0, ac, atp, sp, bufs)
            _write_back_impl(0, ac, atp, sp, bufs, o, partial_out, batch_off)
        else:
            _load_q_impl(0, ac, atp, sp, bufs, q, sbuf_addr, batch_off)
            _qk_and_max_impl(0, ac, atp, sp, bufs)
            _update_max_impl(0, ac, atp, sp, bufs)
            _exp_impl(0, ac, atp, sp, bufs)

            _load_q_impl(1, ac, atp, sp, bufs, q, sbuf_addr, batch_off)
            _qk_and_max_impl(1, ac, atp, sp, bufs)
            _update_max_impl(1, ac, atp, sp, bufs)

            for grp_i in range(0, atp.num_grps - 2):
                _load_q_impl(grp_i + 2, ac, atp, sp, bufs, q, sbuf_addr, batch_off)
                _exp_impl(grp_i + 1, ac, atp, sp, bufs)
                _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs)
                _write_back_impl(grp_i, ac, atp, sp, bufs, o, partial_out, batch_off)
                _update_max_impl(grp_i + 2, ac, atp, sp, bufs)

            _pv_impl(atp.num_grps - 2, ac, atp, sp, bufs)
            _write_back_impl(atp.num_grps - 2, ac, atp, sp, bufs, o, partial_out, batch_off)
            _exp_impl(atp.num_grps - 1, ac, atp, sp, bufs)
            _pv_impl(atp.num_grps - 1, ac, atp, sp, bufs)
            _write_back_impl(atp.num_grps - 1, ac, atp, sp, bufs, o, partial_out, batch_off)


def _allocate_attention_buffers(
    allocator: ModularAllocator,
    ac: AttnConfig,
    atp: AttnTileParams,
    bufs: AttnInternalBuffers,
) -> None:
    mm1_p, mm1_n = atp.sb_p, nl.tile_size.psum_fmax
    mm2_p, mm2_n = atp.sb_p, ac.d

    bufs.k_sb = allocator.alloc_sbuf_tensor(
        shape=(ac.d, _K_TILE_SZ),
        dtype=nl.bfloat16,
        block_dim=[atp.num_k_tiles_per_section],
        num_free_tiles=[atp.num_k_tiles_per_section],
        align_to=32,
    )
    bufs.v_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d),
        dtype=nl.bfloat16,
        block_dim=[atp.num_v_tiles_per_section],
        num_free_tiles=[atp.num_v_tiles_per_section],
    )

    bufs.q_sb = allocator.alloc_sbuf_tensor(
        shape=(ac.d, atp.sb_p * atp.num_q_grps_per_load),
        dtype=nl.bfloat16,
        block_dim=[div_ceil(atp.num_grps, atp.num_q_grps_per_load)],
        num_free_tiles=[2],
        align_to=32,
    )

    bufs.flash_attn_correction_factor = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.mm1_partial_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.num_k_tiles_per_section), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2], align_to=4,
    )
    bufs.mm1_section_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    n_final_reduce_sum_elts = div_ceil(atp.section_len, atp.exp_inst_elems)
    bufs.exp_partial_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, n_final_reduce_sum_elts), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.exp_section_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.prev_mm1_running_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.mm2_prev_output = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.mm2_accum_flash_attn = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.mm2_final = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.mm2_sb = allocator.alloc_sbuf_tensor(
        shape=(mm2_p, mm2_n), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )

    bufs.mm1_masked = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ), dtype=nl.float32,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=[2, atp.num_large_tiles_per_section],
    )
    bufs.exp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, _LARGE_TILE_SZ), dtype=nl.bfloat16,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section],
        num_free_tiles=[1, atp.num_large_tiles_per_section],
    )

    bufs.mm1_psum = []
    for _grp_idx in range(atp.num_grps):
        grp_row = []
        for _large_tile_idx in range(atp.num_large_tiles_per_section):
            tile_row = []
            for k_tile_idx in range(4):
                mm1_psum_tile = nl.ndarray(
                    (mm1_p, mm1_n), dtype=nl.float32, buffer=nl.psum,
                    address=(0, (k_tile_idx % 4) * PSUM_BANK_SIZE),
                )
                tile_row.append(mm1_psum_tile)
            grp_row.append(tile_row)
        bufs.mm1_psum.append(grp_row)

    bufs.exp_tp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.mm2_grp_sz), dtype=nl.bfloat16,
        block_dim=[atp.num_grps, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp],
        num_free_tiles=[2, atp.num_large_tiles_per_section, atp.num_tps_in_mm2_grp],
        align_to=32,
    )

    bufs.mm2_psum = []
    for _grp_idx in range(atp.num_grps):
        grp_row = []
        for large_tile_idx in range(atp.num_large_tiles_per_section):
            mm2_psum_tile = nl.ndarray(
                (mm2_p, mm2_n), dtype=nl.float32, buffer=nl.psum,
                address=(0, (4 + (large_tile_idx % 4)) * PSUM_BANK_SIZE),
            )
            grp_row.append(mm2_psum_tile)
        bufs.mm2_psum.append(grp_row)


def _load_k_tile(k, out, sp: SectionParams, num_tiles: int, batch_off) -> None:
    _, _, seqlen = k.shape
    _, n = out[0].shape if num_tiles > 0 else (0, 0)
    kernel_assert(num_tiles == 0 or n == _K_TILE_SZ, f"expect tile of size {_K_TILE_SZ}")
    d = k.shape[1]

    for tile in range(num_tiles):
        seqlen_offset = sp.section_offset + tile * _K_TILE_SZ
        num_f = min(seqlen - seqlen_offset, _K_TILE_SZ)
        if num_f <= 0:
            continue
        out_dst_pat = out[tile].ap(pattern=[[_K_TILE_SZ, d], [1, num_f]], offset=0)
        k_src_pat = _ap_with_batch(
            k,
            [[seqlen, d], [1, num_f]],
            seqlen_offset,
            batch_off,
        )
        nisa.dma_copy(dst=out_dst_pat, src=k_src_pat)


def _load_v_tile(v, out, sp: SectionParams, num_tiles: int, batch_off) -> None:
    if num_tiles == 0:
        return
    _, seqlen, _ = v.shape
    p, n = out[0].shape
    d = n

    for tile in range(num_tiles):
        seqlen_offset = sp.section_offset + p * tile
        num_p = min(seqlen - seqlen_offset, p)
        if num_p <= 0:
            continue
        out_dst_pat = out[tile].ap(pattern=[[n, num_p], [1, n]], offset=0)
        v_src_pat = _ap_with_batch(
            v,
            [[d, num_p], [1, n]],
            seqlen_offset * d,
            batch_off,
        )
        nisa.dma_copy(dst=out_dst_pat, src=v_src_pat)


def _load_q_tile(q, out, grp_i: int, seqlen_offset: int, grps_per_load: int, batch_off) -> None:
    _, d, seqlen = q.shape
    num_f = min(seqlen - seqlen_offset, _Q_GRP_SZ * grps_per_load)
    out_dst_pat = out[grp_i // grps_per_load].ap(
        pattern=[[_Q_GRP_SZ * grps_per_load, d], [1, num_f]], offset=0,
    )
    q_src_pat = _ap_with_batch(
        q,
        [[seqlen, d], [1, num_f]],
        seqlen_offset,
        batch_off,
    )
    nisa.dma_copy(dst=out_dst_pat, src=q_src_pat)


def _load_q_impl(grp_i, ac, atp, sp, bufs, q, sbuf_addr, batch_off):
    if grp_i % atp.num_q_grps_per_load == 0:
        _load_q_tile(q, bufs.q_sb, grp_i, grp_i * _Q_GRP_SZ, atp.num_q_grps_per_load, batch_off)


def _qk_and_max_impl(grp_i, ac, atp, sp, bufs):
    nisa.memset(bufs.mm1_partial_max[grp_i], value=_FLOAT32_MIN)
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _qk_and_max_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _update_max_impl(grp_i, ac, atp, sp, bufs):
    nisa.tensor_reduce(
        bufs.mm1_section_max[grp_i][:, 0],
        nl.maximum,
        bufs.mm1_partial_max[grp_i],
        1,
        negate=True,
    )

    if atp.num_sections != 1:
        if sp.section_idx == 0:
            nisa.tensor_copy(bufs.mm1_running_max[:, grp_i], bufs.mm1_section_max[grp_i])
            nisa.memset(bufs.flash_attn_correction_factor[grp_i][...], value=0.0)
        else:
            nisa.activation(
                bufs.prev_mm1_running_max[grp_i][...],
                nl.copy,
                bufs.mm1_running_max[:, grp_i],
                scale=-1.0,
                bias=bufs.zero_bias_tensor,
            )
            nisa.tensor_tensor(
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_running_max[:, grp_i],
                bufs.mm1_section_max[grp_i],
                op=nl.minimum,
            )
            nisa.activation(
                bufs.flash_attn_correction_factor[grp_i][:, 0],
                nl.exp,
                bufs.prev_mm1_running_max[grp_i],
                bias=bufs.mm1_running_max[:, grp_i],
                scale=1.0,
            )
    else:
        nisa.tensor_copy(bufs.mm1_running_max[:, grp_i], bufs.mm1_section_max[grp_i])


def _exp_impl(grp_i, ac, atp, sp, bufs):
    q_seqlen_offset = grp_i * atp.sb_p
    nisa.memset(bufs.exp_partial_sum[grp_i][...], value=0.0)

    for large_tile_idx in range(atp.num_large_tiles_per_section):
        kernel_assert(atp.exp_inst_elems == 512, "Internal validation failed.")
        for exp_tile_idx in range(atp.num_exp_insts_per_large_tile):
            k_start_pos = (
                sp.section_offset
                + large_tile_idx * _LARGE_TILE_SZ
                + exp_tile_idx * atp.exp_inst_elems
            )
            num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
            num_f = min(ac.actual_seqlen_k - k_start_pos, atp.exp_inst_elems)

            if num_f <= 0:
                continue

            nisa.activation_reduce(
                bufs.exp_sb[grp_i][large_tile_idx][
                    :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
                ],
                op=nl.exp,
                data=bufs.mm1_masked[grp_i][large_tile_idx][
                    :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
                ],
                reduce_op=nl.add,
                reduce_res=bufs.exp_partial_sum[grp_i][
                    :num_p,
                    large_tile_idx * atp.num_exp_insts_per_large_tile + exp_tile_idx,
                ],
                bias=bufs.mm1_running_max[:num_p, grp_i],
            )

            num_f_outer = num_f // atp.sb_p
            num_f_inner = num_f % atp.sb_p
            if num_f_outer >= 1:
                nisa.dma_transpose(
                    dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [atp.mm2_grp_sz, atp.sb_p], [1, 1],
                            [atp.sb_p, num_f_outer], [1, num_p],
                        ]
                    ),
                    src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [_LARGE_TILE_SZ, num_p], [1, 1],
                            [atp.sb_p, num_f_outer], [1, atp.sb_p],
                        ],
                        offset=exp_tile_idx * atp.mm2_grp_sz,
                    ),
                )
            if num_f_inner > 0:
                nisa.dma_transpose(
                    dst=bufs.exp_tp_sb[grp_i][large_tile_idx][exp_tile_idx].ap(
                        [
                            [atp.mm2_grp_sz, num_f_inner], [1, 1],
                            [atp.sb_p, 1], [1, num_p],
                        ],
                        offset=num_f_outer * atp.sb_p,
                    ),
                    src=bufs.exp_sb[grp_i][large_tile_idx].ap(
                        [
                            [_LARGE_TILE_SZ, num_p], [1, 1],
                            [atp.sb_p, 1], [1, num_f_inner],
                        ],
                        offset=exp_tile_idx * atp.mm2_grp_sz + num_f_outer * atp.sb_p,
                    ),
                )


def _pv_impl(grp_i, ac, atp, sp, bufs):
    nisa.memset(bufs.mm2_sb[grp_i][...], value=0.0)
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)


def _fused_qkmax_and_pv_impl(grp_i, ac, atp, sp, bufs):
    qkmax_grp = grp_i + 2
    nisa.memset(bufs.mm1_partial_max[qkmax_grp][...], value=_FLOAT32_MIN)
    for large_tile_idx in range(atp.num_large_tiles_per_section):
        _pv_large_tile_impl(grp_i, large_tile_idx, ac, atp, sp, bufs)
        _qk_and_max_large_tile_impl(qkmax_grp, large_tile_idx, ac, atp, sp, bufs)


def _write_back_impl(grp_i, ac, atp, sp, bufs, o, partial_out, batch_off):
    q_seqlen_offset = grp_i * atp.sb_p
    is_last_section = sp.section_idx == atp.num_sections - 1

    nisa.tensor_reduce(bufs.exp_section_sum[grp_i][...], nl.add, bufs.exp_partial_sum[grp_i], axis=1)
    if atp.num_sections != 1:
        if sp.section_idx == 0:
            nisa.tensor_copy(bufs.exp_running_sum[:, grp_i], bufs.exp_section_sum[grp_i])
        else:
            nisa.tensor_scalar(
                bufs.exp_running_sum[:, grp_i],
                bufs.exp_running_sum[:, grp_i],
                nl.multiply,
                bufs.flash_attn_correction_factor[grp_i],
                op1=nl.add,
                operand1=bufs.exp_section_sum[grp_i],
            )
        if is_last_section:
            nisa.reciprocal(bufs.exp_sum_reciprocal[:, grp_i], bufs.exp_running_sum[:, grp_i])
    else:
        nisa.reciprocal(bufs.exp_sum_reciprocal[:, grp_i], bufs.exp_section_sum[grp_i])

    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
    num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    if atp.num_sections != 1:
        if sp.section_idx == 0:
            if is_last_section:
                _scale_reciprocal_write_back_impl(
                    bufs.mm2_sb[grp_i], grp_i, ac, atp, bufs, o, num_p, num_f, batch_off
                )
            else:
                _write_back_o_impl(
                    bufs.mm2_sb[grp_i], grp_i, ac, atp, partial_out, num_p, num_f, batch_off
                )
        else:
            prev_dst_pat = bufs.mm2_prev_output[grp_i].ap(
                pattern=[[ac.d, num_p], [1, ac.d]], offset=0,
            )
            partial_src_pat = _ap_with_batch(
                partial_out,
                [[ac.d, num_p], [1, ac.d]],
                grp_i * atp.sb_p * ac.d,
                batch_off,
            )
            nisa.dma_copy(dst=prev_dst_pat, src=partial_src_pat)
            nisa.scalar_tensor_tensor(
                bufs.mm2_accum_flash_attn[grp_i][:num_p, : ac.d],
                data=bufs.mm2_prev_output[grp_i][:num_p, : ac.d],
                op0=nl.multiply,
                operand0=bufs.flash_attn_correction_factor[grp_i][:num_p, 0],
                op1=nl.add,
                operand1=bufs.mm2_sb[grp_i][:num_p, : ac.d],
            )
            if is_last_section:
                _scale_reciprocal_write_back_impl(
                    bufs.mm2_accum_flash_attn[grp_i], grp_i, ac, atp, bufs, o, num_p, num_f, batch_off
                )
            else:
                _write_back_o_impl(
                    bufs.mm2_accum_flash_attn[grp_i],
                    grp_i, ac, atp, partial_out, num_p, num_f, batch_off,
                )
    else:
        _scale_reciprocal_write_back_impl(
            bufs.mm2_sb[grp_i], grp_i, ac, atp, bufs, o, num_p, num_f, batch_off
        )


def _scale_reciprocal_write_back_impl(src_buf, grp_i, ac, atp, bufs, o, num_p, num_f, batch_off):
    nisa.tensor_scalar(
        bufs.mm2_final[grp_i][:num_p, : ac.d],
        src_buf[:num_p, : ac.d],
        nl.multiply,
        bufs.exp_sum_reciprocal[:num_p, grp_i],
        engine=engine.vector,
    )
    _write_back_o_final_impl(bufs.mm2_final[grp_i], grp_i, ac, atp, o, num_p, num_f, batch_off)


def _write_back_o_impl(src_buf, grp_i, ac, atp, o, num_p, num_f, batch_off):
    o_dst_pat = _ap_with_batch(
        o,
        [[ac.d, num_p], [1, ac.d]],
        grp_i * atp.sb_p * ac.d,
        batch_off,
    )
    src_pat = src_buf.ap(pattern=[[ac.d, num_p], [1, ac.d]], offset=0)
    nisa.dma_copy(dst=o_dst_pat, src=src_pat)


def _write_back_o_final_impl(src_buf, grp_i, ac, atp, o, num_p, num_f, batch_off):
    o_view = (
        TensorView(o)
        .select(dim=1, index=batch_off)
        .slice(dim=0, start=grp_i * atp.sb_p, end=grp_i * atp.sb_p + num_p)
        .get_view()
    )
    src_pat = src_buf.ap(pattern=[[ac.d, num_p], [1, ac.d]], offset=0)
    nisa.dma_copy(dst=o_view, src=src_pat)


def _qk_and_max_large_tile_impl(qkmax_grp, large_tile_idx, ac, atp, sp, bufs):
    q_seqlen_offset = qkmax_grp * atp.sb_p

    num_k_tiles_in_large_tile = _LARGE_TILE_SZ // _K_TILE_SZ
    for k_tile_idx in range(num_k_tiles_in_large_tile):
        mm1_psum_tile = bufs.mm1_psum[qkmax_grp][large_tile_idx][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[qkmax_grp][large_tile_idx]
        mm1_partial_max_tile = bufs.mm1_partial_max[qkmax_grp]

        k_tile_idx_in_section = large_tile_idx * num_k_tiles_in_large_tile + k_tile_idx
        k_start_pos = sp.section_offset + k_tile_idx_in_section * _K_TILE_SZ

        if (
            q_seqlen_offset >= ac.seqlen_q
            or k_start_pos >= ac.actual_seqlen_k
            or k_tile_idx_in_section >= atp.num_k_tiles_per_section
        ):
            continue

        num_f = min(ac.actual_seqlen_k - k_start_pos, _K_TILE_SZ)
        num_q_free = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

        nisa.nc_matmul(
            mm1_psum_tile[:num_q_free, :num_f],
            bufs.q_sb[qkmax_grp // atp.num_q_grps_per_load][
                : ac.d,
                nl.ds((qkmax_grp % atp.num_q_grps_per_load) * _Q_GRP_SZ, num_q_free),
            ],
            bufs.k_sb[k_tile_idx_in_section][:, :num_f],
        )

        num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

        nisa.tensor_scalar_reduce(
            mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, num_f)],
            data=mm1_psum_tile[:num_p, :num_f],
            op0=nl.multiply,
            operand0=ac.scale,
            reduce_op=nl.maximum,
            reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx_in_section],
        )


def _pv_large_tile_impl(pv_grp, large_tile_idx, ac, atp, sp, bufs):
    q_seqlen_offset = pv_grp * atp.sb_p
    num_mm2_grps_in_large_tile = _LARGE_TILE_SZ // atp.mm2_grp_sz
    mm2_psum_set = False
    mm2_psum_tile = bufs.mm2_psum[pv_grp][large_tile_idx]

    for mm2_grp_i in range(num_mm2_grps_in_large_tile):
        num_mm2_per_grp = atp.mm2_grp_sz // _V_TILE_SZ
        num_mm2_per_large_tile = num_mm2_per_grp * num_mm2_grps_in_large_tile
        exp_tp_sb_tile = bufs.exp_tp_sb[pv_grp][large_tile_idx][mm2_grp_i]

        k_start_pos_512 = (
            sp.section_offset + large_tile_idx * _LARGE_TILE_SZ + mm2_grp_i * atp.mm2_grp_sz
        )

        for mm2_i in range(num_mm2_per_grp):
            v_tile_idx = (
                large_tile_idx * num_mm2_per_large_tile + mm2_grp_i * num_mm2_per_grp + mm2_i
            )
            k_start_pos = k_start_pos_512 + mm2_i * _V_TILE_SZ
            num_p = min(ac.actual_seqlen_k - k_start_pos, _V_TILE_SZ)
            num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

            if v_tile_idx >= atp.num_v_tiles_per_section or num_p <= 0 or num_f <= 0:
                continue
            mm2_psum_set = True

            nisa.nc_matmul(
                mm2_psum_tile[:num_f, : ac.d],
                exp_tp_sb_tile[:num_p, nl.ds(mm2_i * _V_TILE_SZ, num_f)],
                bufs.v_sb[v_tile_idx][:num_p, : ac.d],
            )

    k_start_pos = sp.section_offset + large_tile_idx * _LARGE_TILE_SZ
    if k_start_pos < ac.actual_seqlen_k and mm2_psum_set:
        num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
        if large_tile_idx == 0:
            nisa.tensor_copy(bufs.mm2_sb[pv_grp][:num_p, :], mm2_psum_tile[:num_p, :])
        else:
            nisa.tensor_tensor(
                bufs.mm2_sb[pv_grp][:num_p, :],
                bufs.mm2_sb[pv_grp][:num_p, :],
                mm2_psum_tile[:num_p, :],
                nl.add,
            )


