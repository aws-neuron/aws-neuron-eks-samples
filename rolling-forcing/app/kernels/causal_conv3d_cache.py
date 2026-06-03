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
from torch_neuronx import nki_op

from utils import _compile

_TILE_P = 128


@nki.jit
def _causal_conv3d_cache_update_shift_kernel(cache, x, HW: int):
    C = cache.shape[0]
    num_p_tiles = (C + _TILE_P - 1) // _TILE_P

    for p_i in range(num_p_tiles):
        p_start = p_i * _TILE_P
        p_size = min(_TILE_P, C - p_start)
        buf = nl.ndarray((p_size, HW), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf[:, :], src=cache[nl.ds(p_start, p_size), HW:])
        nisa.dma_copy(dst=cache[nl.ds(p_start, p_size), :HW], src=buf[:, :])

    for p_i in range(num_p_tiles):
        p_start = p_i * _TILE_P
        p_size = min(_TILE_P, C - p_start)
        nisa.dma_copy(
            dst=cache[nl.ds(p_start, p_size), HW:],
            src=x[nl.ds(p_start, p_size), :],
        )
    return cache


@nki.jit
def _causal_conv3d_cache_update_copy_kernel(cache, x):
    C = cache.shape[0]
    cache_size = cache.shape[1]
    x_offset = x.shape[1] - cache_size
    num_p_tiles = (C + _TILE_P - 1) // _TILE_P

    for p_i in range(num_p_tiles):
        p_start = p_i * _TILE_P
        p_size = min(_TILE_P, C - p_start)
        nisa.dma_copy(
            dst=cache[nl.ds(p_start, p_size), :],
            src=x[nl.ds(p_start, p_size), x_offset:],
        )
    return cache


@nki_op("dit_flint::causal_conv3d_cache_update_shift", mutates_args={"cache"})
def _causal_conv3d_cache_update_shift_op(
    cache: torch.Tensor, x: torch.Tensor, HW: int
) -> None:
    _causal_conv3d_cache_update_shift_kernel(cache, x, HW)


@nki_op("dit_flint::causal_conv3d_cache_update_copy", mutates_args={"cache"})
def _causal_conv3d_cache_update_copy_op(
    cache: torch.Tensor, x: torch.Tensor
) -> None:
    _causal_conv3d_cache_update_copy_kernel(cache, x)


@_compile
def _causal_conv3d_cache_update_shift_compiled(
    cache: torch.Tensor, x: torch.Tensor, HW: int
) -> None:
    _causal_conv3d_cache_update_shift_op(cache, x, HW)


@_compile
def _causal_conv3d_cache_update_copy_compiled(
    cache: torch.Tensor, x: torch.Tensor
) -> None:
    _causal_conv3d_cache_update_copy_op(cache, x)


def causal_conv3d_cache_update_shift(
    cache: torch.Tensor, x: torch.Tensor, HW: int
) -> torch.Tensor:
    _causal_conv3d_cache_update_shift_compiled(cache, x, HW)
    return cache


def causal_conv3d_cache_update_copy(
    cache: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    _causal_conv3d_cache_update_copy_compiled(cache, x)
    return cache
