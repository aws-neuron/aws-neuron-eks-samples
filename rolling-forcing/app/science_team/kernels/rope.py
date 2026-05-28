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

import nki.isa as nisa
import nki.language as nl

from kernels.nkilib_compat import kernel_assert
from kernels.nkilib_compat import PSUM_BANK_SIZE

import nki
from torch_neuronx.nki_hop import wrap_nki


_P = 128


@wrap_nki
@nki.jit
def causal_rope_rotation(
    x: nl.ndarray,
    cos_sin: nl.ndarray,
    head_start: int = 0,
    head_end: int = 12,
    head_dim: int = 128,
):
    kernel_assert(
        head_end > head_start,
        f"head_end ({head_end}) must be > head_start ({head_start})",
    )
    kernel_assert(
        head_end <= x.shape[1],
        f"head_end ({head_end}) must be <= num_heads_full ({x.shape[1]})",
    )

    seq_len = x.shape[0]
    N = head_end - head_start
    D = head_dim

    out = nl.ndarray(shape=(seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    num_tiles = (seq_len + _P - 1) // _P

    for tile_i in range(num_tiles):
        tile_start = tile_i * _P
        tile_size = min(_P, seq_len - tile_start)

        cos_sin_tile = nl.ndarray(
            (_P, 2 * D), dtype=nl.float32, buffer=nl.sbuf,
        )
        if tile_size < _P:
            nisa.memset(cos_sin_tile, value=0.0)
        nisa.dma_copy(
            dst=cos_sin_tile[:tile_size, :],
            src=cos_sin[nl.ds(tile_start, tile_size), :],
        )

        cos_tile = (
            cos_sin_tile[:, nl.ds(0, D)].expand_dim(1).broadcast(1, N)
        )
        sin_tile = (
            cos_sin_tile[:, nl.ds(D, D)].expand_dim(1).broadcast(1, N)
        )

        x_all = nl.ndarray(
            (_P, N, D), dtype=nl.bfloat16, buffer=nl.sbuf,
        )
        if tile_size < _P:
            nisa.memset(x_all, value=0.0)
        nisa.dma_copy(
            dst=x_all[:tile_size, :, :],
            src=x[nl.ds(tile_start, tile_size), nl.ds(head_start, N), :],
        )

        x_shaped = x_all.reshape_dim(2, (-1, 2))
        x_even = x_shaped[:, :, :, nl.ds(0, 1)]
        x_odd = x_shaped[:, :, :, nl.ds(1, 1)]

        sin_shaped = sin_tile.reshape_dim(2, (-1, 2))
        sin_even = sin_shaped[:, :, :, nl.ds(0, 1)]
        sin_odd = sin_shaped[:, :, :, nl.ds(1, 1)]

        x_cos_all = nl.ndarray(
            (_P, N, D), dtype=nl.float32, buffer=nl.sbuf,
        )
        nisa.tensor_tensor(x_cos_all, x_all, cos_tile, op=nl.multiply)

        x_sin_all = nl.ndarray(
            (_P, N, D), dtype=nl.float32, buffer=nl.sbuf,
        )
        x_sin_shaped = x_sin_all.reshape_dim(2, (-1, 2))
        nisa.tensor_tensor(
            x_sin_shaped[:, :, :, nl.ds(0, 1)],
            x_odd, sin_even, op=nl.multiply,
        )
        nisa.tensor_tensor(
            x_sin_shaped[:, :, :, nl.ds(1, 1)],
            x_even, sin_odd, op=nl.multiply,
        )

        out_all = nl.ndarray(
            (_P, N, D), dtype=x.dtype, buffer=nl.sbuf,
        )
        nisa.tensor_tensor(out_all, x_cos_all, x_sin_all, op=nl.add)

        nisa.dma_copy(
            dst=out[nl.ds(tile_start, tile_size), :, :],
            src=out_all[:tile_size, :, :],
        )

    return out


@wrap_nki
@nki.jit
def build_rope_grids(
    freqs_cos: nl.ndarray,
    freqs_sin: nl.ndarray,
    sign_pattern: nl.ndarray,
    start_frame: nl.ndarray,
    F: int = 15,
    H: int = 30,
    W: int = 52,
    head_dim: int = 128,
):
    c = head_dim // 2
    D = head_dim
    s0 = c - 2 * (c // 3)
    s1 = c // 3

    kernel_assert(H <= _P, f"build_rope_grids requires H <= {_P}, got {H}")

    combined_out = nl.ndarray(
        shape=(F * H, W * 2 * D), dtype=nl.float32, buffer=nl.shared_hbm,
    )

    sign_H = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sign_H, src=sign_pattern[:H, :])
    sign_4d = sign_H.reshape_dim(1, (c, 2)).expand_dim(1).broadcast(1, W)

    ones_H = nl.ndarray((1, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_H, value=1.0)

    sf_sbuf = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sf_sbuf, src=start_frame)

    w_cos_raw = nl.ndarray((_P, s1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(w_cos_raw, value=0.0)
    nisa.dma_copy(
        dst=w_cos_raw[:W, :],
        src=freqs_cos[nl.ds(0, W), nl.ds(s0 + s1, s1)],
    )

    w_cos_flat = nl.ndarray(
        (1, W * s1), dtype=nl.float32, buffer=nl.sbuf,
    )
    for wi in range(W):
        nisa.dma_copy(
            dst=w_cos_flat[:, nl.ds(wi * s1, s1)],
            src=w_cos_raw[nl.ds(wi, 1), :],
        )

    w_sin_raw = nl.ndarray((_P, s1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(w_sin_raw, value=0.0)
    nisa.dma_copy(
        dst=w_sin_raw[:W, :],
        src=freqs_sin[nl.ds(0, W), nl.ds(s0 + s1, s1)],
    )

    w_sin_flat = nl.ndarray(
        (1, W * s1), dtype=nl.float32, buffer=nl.sbuf,
    )
    for wi in range(W):
        nisa.dma_copy(
            dst=w_sin_flat[:, nl.ds(wi * s1, s1)],
            src=w_sin_raw[nl.ds(wi, 1), :],
        )

    MATMUL_FREE_MAX = 512
    ws = W * s1
    num_chunks = (ws + MATMUL_FREE_MAX - 1) // MATMUL_FREE_MAX
    ws_padded = num_chunks * MATMUL_FREE_MAX

    w_cos_pad = nl.ndarray(
        (1, ws_padded), dtype=nl.float32, buffer=nl.sbuf,
    )
    nisa.memset(w_cos_pad, value=0.0)
    nisa.tensor_copy(dst=w_cos_pad[:, nl.ds(0, ws)], src=w_cos_flat)

    w_sin_pad = nl.ndarray(
        (1, ws_padded), dtype=nl.float32, buffer=nl.sbuf,
    )
    nisa.memset(w_sin_pad, value=0.0)
    nisa.tensor_copy(dst=w_sin_pad[:, nl.ds(0, ws)], src=w_sin_flat)

    w_cos_bc_pad = nl.ndarray(
        (H, ws_padded), dtype=nl.float32, buffer=nl.sbuf,
    )
    w_sin_bc_pad = nl.ndarray(
        (H, ws_padded), dtype=nl.float32, buffer=nl.sbuf,
    )
    psum_chunk = nl.ndarray(
        (H, MATMUL_FREE_MAX), dtype=nl.float32, buffer=nl.psum,
        address=(0, 0 * PSUM_BANK_SIZE),
    )
    for chunk_off in range(0, ws_padded, MATMUL_FREE_MAX):
        nisa.nc_matmul(
            psum_chunk, ones_H, w_cos_pad[:, nl.ds(chunk_off, MATMUL_FREE_MAX)],
            accumulate=False,
        )
        nisa.tensor_copy(
            dst=w_cos_bc_pad[:, nl.ds(chunk_off, MATMUL_FREE_MAX)],
            src=psum_chunk,
        )
        nisa.nc_matmul(
            psum_chunk, ones_H, w_sin_pad[:, nl.ds(chunk_off, MATMUL_FREE_MAX)],
            accumulate=False,
        )
        nisa.tensor_copy(
            dst=w_sin_bc_pad[:, nl.ds(chunk_off, MATMUL_FREE_MAX)],
            src=psum_chunk,
        )

    w_cos_bc = w_cos_bc_pad[:, nl.ds(0, ws)]
    w_sin_bc = w_sin_bc_pad[:, nl.ds(0, ws)]

    h_cos = nl.ndarray((H, s1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_cos, src=freqs_cos[nl.ds(0, H), nl.ds(s0, s1)])

    h_sin = nl.ndarray((H, s1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=h_sin, src=freqs_sin[nl.ds(0, H), nl.ds(s0, s1)])

    frame_cos = nl.ndarray((_P, c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(frame_cos, value=0.0)
    nisa.dma_copy(
        dst=frame_cos[:F, :],
        src=freqs_cos.ap(
            pattern=[[freqs_cos.shape[1], F], [1, c]],
            offset=0,
            scalar_offset=sf_sbuf,
            indirect_dim=0,
        ),
    )

    frame_sin = nl.ndarray((_P, c), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(frame_sin, value=0.0)
    nisa.dma_copy(
        dst=frame_sin[:F, :],
        src=freqs_sin.ap(
            pattern=[[freqs_sin.shape[1], F], [1, c]],
            offset=0,
            scalar_offset=sf_sbuf,
            indirect_dim=0,
        ),
    )

    psum_fs = nl.ndarray(
        (H, s0), dtype=nl.float32, buffer=nl.psum,
        address=(0, 1 * PSUM_BANK_SIZE),
    )

    for f in range(F):
        fc_row = nl.ndarray((1, s0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=fc_row, src=frame_cos[nl.ds(f, 1), nl.ds(0, s0)])
        nisa.nc_matmul(psum_fs, ones_H, fc_row, accumulate=False)
        fc_bc = nl.ndarray((H, s0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(fc_bc, psum_fs)

        fs_row = nl.ndarray((1, s0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=fs_row, src=frame_sin[nl.ds(f, 1), nl.ds(0, s0)])
        nisa.nc_matmul(psum_fs, ones_H, fs_row, accumulate=False)
        fs_bc = nl.ndarray((H, s0), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(fs_bc, psum_fs)

        cos_full = nl.ndarray((H, W * c), dtype=nl.float32, buffer=nl.sbuf)
        cos_3d = cos_full.reshape_dim(1, (W, c))
        nisa.tensor_copy(
            dst=cos_3d[:, :, nl.ds(0, s0)],
            src=fc_bc.expand_dim(1).broadcast(1, W),
        )
        nisa.tensor_copy(
            dst=cos_3d[:, :, nl.ds(s0, s1)],
            src=h_cos.expand_dim(1).broadcast(1, W),
        )
        nisa.tensor_copy(
            dst=cos_3d[:, :, nl.ds(s0 + s1, s1)],
            src=w_cos_bc.reshape_dim(1, (W, s1)),
        )

        sin_full = nl.ndarray((H, W * c), dtype=nl.float32, buffer=nl.sbuf)
        sin_3d = sin_full.reshape_dim(1, (W, c))
        nisa.tensor_copy(
            dst=sin_3d[:, :, nl.ds(0, s0)],
            src=fs_bc.expand_dim(1).broadcast(1, W),
        )
        nisa.tensor_copy(
            dst=sin_3d[:, :, nl.ds(s0, s1)],
            src=h_sin.expand_dim(1).broadcast(1, W),
        )
        nisa.tensor_copy(
            dst=sin_3d[:, :, nl.ds(s0 + s1, s1)],
            src=w_sin_bc.reshape_dim(1, (W, s1)),
        )

        cos_e = cos_3d.expand_dim(3).broadcast(3, 2)
        sin_e = sin_3d.expand_dim(3).broadcast(3, 2)

        sin_s = nl.ndarray((H, W * D), dtype=nl.float32, buffer=nl.sbuf)
        sin_s_4d = sin_s.reshape_dim(1, (W, c, 2))
        nisa.tensor_tensor(sin_s_4d, sin_e, sign_4d, op=nl.multiply)

        combined_tile = nl.ndarray(
            (H, W * 2 * D), dtype=nl.float32, buffer=nl.sbuf,
        )
        combined_3d = combined_tile.reshape_dim(1, (W, 2 * D))
        nisa.tensor_copy(
            dst=combined_3d[:, :, nl.ds(0, D)].reshape_dim(2, (c, 2)),
            src=cos_e,
        )
        nisa.tensor_copy(
            dst=combined_3d[:, :, nl.ds(D, D)].reshape_dim(2, (c, 2)),
            src=sin_s_4d,
        )

        nisa.dma_copy(
            dst=combined_out[nl.ds(f * H, H), :],
            src=combined_tile,
        )

    return combined_out


