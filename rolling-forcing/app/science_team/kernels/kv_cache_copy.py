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

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from kernels.nki_op_compat import nki_op

from utils import _compile


_TILE_THRESHOLD = 1024
_TILE_ROWS_LARGE_PAYLOAD = 128
_TILE_ROWS_SMALL_PAYLOAD = 1024


def _tile_rows(payload_per_row: int) -> int:
    if payload_per_row > _TILE_THRESHOLD:
        return _TILE_ROWS_LARGE_PAYLOAD
    return _TILE_ROWS_SMALL_PAYLOAD


@nki.jit
def _cache_copy_kernel(dst, src):
    seqlen = src.shape[0]
    payload_per_row = src.shape[1] * src.shape[2]
    tile_rows = _tile_rows(payload_per_row)

    num_tiles = (seqlen + tile_rows - 1) // tile_rows
    for tile_i in range(num_tiles):
        tile_start = tile_i * tile_rows
        current_size = min(tile_rows, seqlen - tile_start)
        nisa.dma_copy(
            dst=dst[nl.ds(tile_start, current_size), :, :],
            src=src[nl.ds(tile_start, current_size), :, :],
        )

    return dst


@nki.jit
def _kv_cache_copy_kernel(k_dst, k_src, v_dst, v_src):
    seqlen = k_src.shape[0]
    payload_per_row = k_src.shape[1] * k_src.shape[2]
    tile_rows = _tile_rows(payload_per_row)

    num_tiles = (seqlen + tile_rows - 1) // tile_rows
    for tile_i in range(num_tiles):
        tile_start = tile_i * tile_rows
        current_size = min(tile_rows, seqlen - tile_start)
        nisa.dma_copy(
            dst=k_dst[nl.ds(tile_start, current_size), :, :],
            src=k_src[nl.ds(tile_start, current_size), :, :],
        )
        nisa.dma_copy(
            dst=v_dst[nl.ds(tile_start, current_size), :, :],
            src=v_src[nl.ds(tile_start, current_size), :, :],
        )

    return k_dst, v_dst


@nki_op("dit_flint::cache_copy", mutates_args={"dst"})
def _cache_copy_op(dst: torch.Tensor, src: torch.Tensor) -> None:
    _cache_copy_kernel(dst, src)


@nki_op("dit_flint::kv_cache_copy", mutates_args={"k_dst", "v_dst"})
def _kv_cache_copy_op(
    k_dst: torch.Tensor,
    k_src: torch.Tensor,
    v_dst: torch.Tensor,
    v_src: torch.Tensor,
) -> None:
    _kv_cache_copy_kernel(k_dst, k_src, v_dst, v_src)


@_compile
def _cache_copy_compiled(dst: torch.Tensor, src: torch.Tensor) -> None:
    _cache_copy_op(dst, src)


@_compile
def _kv_cache_copy_compiled(
    k_dst: torch.Tensor,
    k_src: torch.Tensor,
    v_dst: torch.Tensor,
    v_src: torch.Tensor,
) -> None:
    _kv_cache_copy_op(k_dst, k_src, v_dst, v_src)


def cache_copy(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    _cache_copy_compiled(dst, src)
    return dst


def kv_cache_copy(
    k_dst: torch.Tensor,
    k_src: torch.Tensor,
    v_dst: torch.Tensor,
    v_src: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _kv_cache_copy_compiled(k_dst, k_src, v_dst, v_src)
    return k_dst, v_dst
