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
import torch
from torch_neuronx import nki_op

from utils import _compile


@nki.jit
def _extract_w_edges_kernel(x, out, W: int, radius: int):
    nisa.dma_copy(dst=out[:radius, :], src=x[:radius, :])
    nisa.dma_copy(dst=out[radius:, :], src=x[W - radius:, :])
    return out


@nki_op("dit_flint::extract_w_edges", mutates_args={"out"})
def _extract_w_edges_op(
    x: torch.Tensor, out: torch.Tensor, W: int, radius: int
) -> None:
    _extract_w_edges_kernel(x, out, W, radius)


@_compile
def _extract_w_edges_compiled(
    x: torch.Tensor, out: torch.Tensor, W: int, radius: int
) -> None:
    _extract_w_edges_op(x, out, W, radius)


def extract_w_edges(
    x: torch.Tensor, out: torch.Tensor, W: int, radius: int
) -> torch.Tensor:
    _extract_w_edges_compiled(x, out, W, radius)
    return out
