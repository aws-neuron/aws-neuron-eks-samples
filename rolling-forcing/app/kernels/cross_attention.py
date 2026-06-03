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
from torch_neuronx import wrap_nki

_FLOAT32_MIN = -3.4028235e38

_MAX_SEQLEN_Q = 131072
_MAX_HEAD_DIM = 128

_Q_GRP_SZ = 128
_V_TILE_SZ = 128
_K_TILE_SZ = 512
_EXP_TILE_SZ = 512


@wrap_nki
@nki.jit
def wan_cross_attn(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    softmax_scale: Optional[float] = None,
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
    kernel_assert(
        seqlen_k == _K_TILE_SZ,
        f"wan_cross_attn requires seqlen_k == {_K_TILE_SZ}, got {seqlen_k}",
    )
    kernel_assert(seqlen_q <= _MAX_SEQLEN_Q, f"seqlen_q={seqlen_q} exceeds {_MAX_SEQLEN_Q}")
    kernel_assert(d > 0 and d <= _MAX_HEAD_DIM, f"d must be in (0,{_MAX_HEAD_DIM}], got {d=}")

    result = nl.ndarray(shape=(seqlen_q, batch_size, d), dtype=q.dtype, buffer=nl.shared_hbm)

    ac = AttnConfig(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        d=d,
        bs=batch_size,
        scale=softmax_scale,
        dtype=q.dtype,
    )

    for batch_id in range(batch_size):
        _wan_cross_attn_impl(q, k, v, result, ac, batch_id)

    return result


@dataclass
class AttnConfig(nl.NKIObject):
    seqlen_q: int = None
    seqlen_k: int = None
    d: int = None
    bs: int = None
    scale: float = None
    dtype: Any = None


@dataclass
class AttnTileParams(nl.NKIObject):
    sb_p: int = None
    num_grps: int = None
    num_q_grps_per_load: int = None

    num_k_tiles: int = None
    num_v_tiles: int = None

    exp_inst_elems: int = None
    num_exp_insts: int = None
    num_tps_in_mm2_grp: int = None
    mm2_grp_sz: int = None


@dataclass
class AttnInternalBuffers(nl.NKIObject):

    q_sb = None
    k_sb = None
    v_sb = None

    mm1_psum = None
    mm1_masked = None
    mm1_partial_max = None
    mm1_section_max = None

    exp_sb = None
    exp_partial_sum = None
    exp_section_sum = None
    exp_tp_sb = None
    exp_sum_reciprocal = None

    mm2_psum = None
    mm2_sb = None
    mm2_final = None


def _compute_tile_parameters(ac: AttnConfig) -> AttnTileParams:
    atp = AttnTileParams()

    atp.sb_p = nl.tile_size.pmax
    kernel_assert(_Q_GRP_SZ == atp.sb_p, f"expect _Q_GRP_SZ == sb_p, got {_Q_GRP_SZ=}, {atp.sb_p=}")
    kernel_assert(_V_TILE_SZ == atp.sb_p, f"expect _V_TILE_SZ == sb_p, got {_V_TILE_SZ=}, {atp.sb_p=}")
    kernel_assert(ac.seqlen_k == _K_TILE_SZ, f"expect seqlen_k == {_K_TILE_SZ}, got {ac.seqlen_k}")

    atp.num_grps = div_ceil(ac.seqlen_q, atp.sb_p)
    num_q_grps_per_load_dtype = 4 if ac.dtype == nl.float32 else 8
    atp.num_q_grps_per_load = min(num_q_grps_per_load_dtype, atp.num_grps)

    atp.num_k_tiles = ac.seqlen_k // _K_TILE_SZ
    atp.num_v_tiles = ac.seqlen_k // _V_TILE_SZ

    atp.exp_inst_elems = _EXP_TILE_SZ
    atp.num_exp_insts = ac.seqlen_k // atp.exp_inst_elems
    atp.num_tps_in_mm2_grp = _K_TILE_SZ // atp.sb_p
    atp.mm2_grp_sz = _K_TILE_SZ

    return atp


def _wan_cross_attn_impl(q, k, v, o, ac: AttnConfig, batch_id: int):
    atp = _compute_tile_parameters(ac)

    allocator = ModularAllocator(initial_address=0)
    bufs = AttnInternalBuffers()

    bufs.zero_bias_tensor = allocator.alloc_sbuf_tensor(shape=(atp.sb_p, 1), dtype=nl.float32)
    nisa.memset(bufs.zero_bias_tensor, 0.0)

    _allocate_attention_buffers(allocator, ac, atp, bufs)
    sbuf_addr = allocator.get_current_address()

    _load_k_tile(k, bufs.k_sb, atp, batch_id)
    _load_v_tile(v, bufs.v_sb, atp, batch_id)

    if atp.num_grps <= 1:
        _load_q_impl(0, ac, atp, bufs, q, batch_id)
        _qk_and_max_impl(0, ac, atp, bufs)
        _exp_impl(0, ac, atp, bufs)
        _pv_impl(0, ac, atp, bufs)
        _write_back_impl(0, ac, atp, bufs, o, batch_id)
    else:
        _load_q_impl(0, ac, atp, bufs, q, batch_id)
        _qk_and_max_impl(0, ac, atp, bufs)
        _exp_impl(0, ac, atp, bufs)

        _load_q_impl(1, ac, atp, bufs, q, batch_id)
        _qk_and_max_impl(1, ac, atp, bufs)

        for grp_i in range(0, atp.num_grps - 2):
            _load_q_impl(grp_i + 2, ac, atp, bufs, q, batch_id)
            _exp_impl(grp_i + 1, ac, atp, bufs)
            _fused_qkmax_and_pv_impl(grp_i, ac, atp, bufs)
            _write_back_impl(grp_i, ac, atp, bufs, o, batch_id)

        _pv_impl(atp.num_grps - 2, ac, atp, bufs)
        _write_back_impl(atp.num_grps - 2, ac, atp, bufs, o, batch_id)
        _exp_impl(atp.num_grps - 1, ac, atp, bufs)
        _pv_impl(atp.num_grps - 1, ac, atp, bufs)
        _write_back_impl(atp.num_grps - 1, ac, atp, bufs, o, batch_id)


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
        block_dim=[atp.num_k_tiles],
        num_free_tiles=[atp.num_k_tiles],
        align_to=32,
    )
    bufs.v_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.d),
        dtype=nl.bfloat16,
        block_dim=[atp.num_v_tiles],
        num_free_tiles=[atp.num_v_tiles],
    )

    bufs.q_sb = allocator.alloc_sbuf_tensor(
        shape=(ac.d, atp.sb_p * atp.num_q_grps_per_load),
        dtype=nl.bfloat16,
        block_dim=[div_ceil(atp.num_grps, atp.num_q_grps_per_load)],
        num_free_tiles=[2],
        align_to=32,
    )

    bufs.mm1_partial_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.num_k_tiles), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2], align_to=4,
    )
    bufs.mm1_section_max = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.exp_partial_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.num_exp_insts), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.exp_section_sum = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
        block_dim=[atp.num_grps], num_free_tiles=[2],
    )
    bufs.exp_sum_reciprocal = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, 1), dtype=nl.float32,
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
        shape=(atp.sb_p, ac.seqlen_k), dtype=nl.float32,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )
    bufs.exp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, ac.seqlen_k), dtype=nl.bfloat16,
        block_dim=[atp.num_grps],
        num_free_tiles=[2],
    )

    bufs.mm1_psum = []
    for _grp_idx in range(atp.num_grps):
        tile_row = []
        for k_tile_idx in range(atp.num_k_tiles):
            mm1_psum_tile = nl.ndarray(
                (mm1_p, mm1_n), dtype=nl.float32, buffer=nl.psum,
                address=(0, (k_tile_idx % 4) * PSUM_BANK_SIZE),
            )
            tile_row.append(mm1_psum_tile)
        bufs.mm1_psum.append(tile_row)

    bufs.exp_tp_sb = allocator.alloc_sbuf_tensor(
        shape=(atp.sb_p, atp.mm2_grp_sz), dtype=nl.bfloat16,
        block_dim=[atp.num_grps, atp.num_tps_in_mm2_grp],
        num_free_tiles=[2, atp.num_tps_in_mm2_grp],
        align_to=32,
    )

    bufs.mm2_psum = []
    for _grp_idx in range(atp.num_grps):
        mm2_psum_tile = nl.ndarray(
            (mm2_p, mm2_n), dtype=nl.float32, buffer=nl.psum,
            address=(0, 4 * PSUM_BANK_SIZE),
        )
        bufs.mm2_psum.append(mm2_psum_tile)


def _load_k_tile(k, out, atp: AttnTileParams, batch_id: int) -> None:
    _, _, seqlen = k.shape
    d = k.shape[1]

    for tile in range(atp.num_k_tiles):
        seqlen_offset = tile * _K_TILE_SZ
        out_dst_pat = out[tile].ap(
            pattern=[[_K_TILE_SZ, d], [1, _K_TILE_SZ]], offset=0,
        )
        k_src_pat = k.ap(
            pattern=[[seqlen, d], [1, _K_TILE_SZ]],
            offset=batch_id * d * seqlen + seqlen_offset,
        )
        nisa.dma_copy(dst=out_dst_pat, src=k_src_pat)


def _load_v_tile(v, out, atp: AttnTileParams, batch_id: int) -> None:
    _, seqlen, _ = v.shape
    p, n = out[0].shape
    d = n

    for tile in range(atp.num_v_tiles):
        seqlen_offset = p * tile
        out_dst_pat = out[tile].ap(pattern=[[n, p], [1, n]], offset=0)
        v_src_pat = v.ap(
            pattern=[[d, p], [1, n]],
            offset=batch_id * seqlen * d + seqlen_offset * d,
        )
        nisa.dma_copy(dst=out_dst_pat, src=v_src_pat)


def _load_q_tile(q, out, grp_i: int, seqlen_offset: int, grps_per_load: int, batch_id: int) -> None:
    _, d, seqlen = q.shape
    num_f = min(seqlen - seqlen_offset, _Q_GRP_SZ * grps_per_load)
    out_dst_pat = out[grp_i // grps_per_load].ap(
        pattern=[[_Q_GRP_SZ * grps_per_load, d], [1, num_f]], offset=0,
    )
    q_src_pat = q.ap(
        pattern=[[seqlen, d], [1, num_f]],
        offset=batch_id * d * seqlen + seqlen_offset,
    )
    nisa.dma_copy(dst=out_dst_pat, src=q_src_pat)


def _load_q_impl(grp_i, ac, atp, bufs, q, batch_id):
    if grp_i % atp.num_q_grps_per_load == 0:
        _load_q_tile(q, bufs.q_sb, grp_i, grp_i * _Q_GRP_SZ, atp.num_q_grps_per_load, batch_id)


def _qk_and_max_impl(grp_i, ac, atp, bufs):
    q_seqlen_offset = grp_i * atp.sb_p
    nisa.memset(bufs.mm1_partial_max[grp_i], value=_FLOAT32_MIN)

    for k_tile_idx in range(atp.num_k_tiles):
        mm1_psum_tile = bufs.mm1_psum[grp_i][k_tile_idx]
        mm1_masked_tile = bufs.mm1_masked[grp_i]
        mm1_partial_max_tile = bufs.mm1_partial_max[grp_i]

        if q_seqlen_offset >= ac.seqlen_q:
            continue

        num_q_free = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
        num_p = num_q_free

        nisa.nc_matmul(
            mm1_psum_tile[:num_q_free, :_K_TILE_SZ],
            bufs.q_sb[grp_i // atp.num_q_grps_per_load][
                : ac.d,
                nl.ds((grp_i % atp.num_q_grps_per_load) * _Q_GRP_SZ, num_q_free),
            ],
            bufs.k_sb[k_tile_idx][:, :_K_TILE_SZ],
        )

        nisa.tensor_scalar_reduce(
            mm1_masked_tile[:num_p, nl.ds(k_tile_idx * _K_TILE_SZ, _K_TILE_SZ)],
            data=mm1_psum_tile[:num_p, :_K_TILE_SZ],
            op0=nl.multiply,
            operand0=ac.scale,
            reduce_op=nl.maximum,
            reduce_res=mm1_partial_max_tile[:num_p, k_tile_idx],
        )


def _exp_impl(grp_i, ac, atp, bufs):
    q_seqlen_offset = grp_i * atp.sb_p

    nisa.tensor_reduce(
        bufs.mm1_section_max[grp_i][:, 0],
        nl.maximum,
        bufs.mm1_partial_max[grp_i],
        1,
        negate=True,
    )

    nisa.memset(bufs.exp_partial_sum[grp_i][...], value=0.0)

    for exp_tile_idx in range(atp.num_exp_insts):
        num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
        num_f = atp.exp_inst_elems

        if num_p <= 0:
            continue

        nisa.activation_reduce(
            bufs.exp_sb[grp_i][
                :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
            ],
            op=nl.exp,
            data=bufs.mm1_masked[grp_i][
                :num_p, nl.ds(exp_tile_idx * atp.exp_inst_elems, num_f)
            ],
            reduce_op=nl.add,
            reduce_res=bufs.exp_partial_sum[grp_i][:num_p, exp_tile_idx],
            bias=bufs.mm1_section_max[grp_i][:num_p, 0],
        )

        num_f_outer = num_f // atp.sb_p
        nisa.dma_transpose(
            dst=bufs.exp_tp_sb[grp_i][exp_tile_idx].ap(
                [
                    [atp.mm2_grp_sz, atp.sb_p], [1, 1],
                    [atp.sb_p, num_f_outer], [1, num_p],
                ]
            ),
            src=bufs.exp_sb[grp_i].ap(
                [
                    [ac.seqlen_k, num_p], [1, 1],
                    [atp.sb_p, num_f_outer], [1, atp.sb_p],
                ],
                offset=exp_tile_idx * atp.mm2_grp_sz,
            ),
        )


def _pv_impl(grp_i, ac, atp, bufs):
    q_seqlen_offset = grp_i * atp.sb_p
    if q_seqlen_offset >= ac.seqlen_q:
        return

    num_f = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)
    num_p = num_f
    mm2_psum_tile = bufs.mm2_psum[grp_i]

    for mm2_grp_i in range(atp.num_exp_insts):
        num_mm2_per_grp = atp.mm2_grp_sz // _V_TILE_SZ
        exp_tp_sb_tile = bufs.exp_tp_sb[grp_i][mm2_grp_i]

        for mm2_i in range(num_mm2_per_grp):
            v_tile_idx = mm2_grp_i * num_mm2_per_grp + mm2_i
            nisa.nc_matmul(
                mm2_psum_tile[:num_f, : ac.d],
                exp_tp_sb_tile[:, nl.ds(mm2_i * _V_TILE_SZ, num_f)],
                bufs.v_sb[v_tile_idx][:, : ac.d],
            )

    nisa.tensor_copy(bufs.mm2_sb[grp_i][:num_p, :], mm2_psum_tile[:num_p, :])


def _fused_qkmax_and_pv_impl(grp_i, ac, atp, bufs):
    qkmax_grp = grp_i + 2
    _pv_impl(grp_i, ac, atp, bufs)
    _qk_and_max_impl(qkmax_grp, ac, atp, bufs)


def _write_back_impl(grp_i, ac, atp, bufs, o, batch_id):
    q_seqlen_offset = grp_i * atp.sb_p
    if q_seqlen_offset >= ac.seqlen_q:
        return

    num_p = min(ac.seqlen_q - q_seqlen_offset, _Q_GRP_SZ)

    nisa.tensor_reduce(
        bufs.exp_section_sum[grp_i][...], nl.add, bufs.exp_partial_sum[grp_i], axis=1,
    )
    nisa.reciprocal(bufs.exp_sum_reciprocal[grp_i][:, 0], bufs.exp_section_sum[grp_i][:, 0])

    nisa.tensor_scalar(
        bufs.mm2_final[grp_i][:num_p, : ac.d],
        bufs.mm2_sb[grp_i][:num_p, : ac.d],
        nl.multiply,
        bufs.exp_sum_reciprocal[grp_i][:num_p, 0],
        engine=engine.vector,
    )

    o_view = (
        TensorView(o)
        .select(dim=1, index=batch_id)
        .slice(dim=0, start=grp_i * atp.sb_p, end=grp_i * atp.sb_p + num_p)
        .get_view()
    )
    src_pat = bufs.mm2_final[grp_i].ap(pattern=[[ac.d, num_p], [1, ac.d]], offset=0)
    nisa.dma_copy(dst=o_view, src=src_pat)


