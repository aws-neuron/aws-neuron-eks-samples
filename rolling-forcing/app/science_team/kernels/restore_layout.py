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

import nki
from torch_neuronx.nki_hop import wrap_nki


_FRAME_SEQLEN = 1560
_L_CU = 3 * _FRAME_SEQLEN
_L_DN = 15 * _FRAME_SEQLEN


@wrap_nki
@nki.jit
def restore_layout(gathered: nl.ndarray, N: int = 2):
    L_full, dim = gathered.shape
    kernel_assert(
        L_full == _L_CU + _L_DN,
        f"restore_layout expects L_full = {_L_CU + _L_DN}, got {L_full}",
    )

    L_full_N = L_full // N
    L_cu_N = _L_CU // N
    L_dn_N = _L_DN // N

    out = nl.ndarray(shape=(L_full, dim), dtype=gathered.dtype, buffer=nl.shared_hbm)

    for w in range(N):
        nisa.dma_copy(
            dst=out[nl.ds(w * L_cu_N, L_cu_N), :],
            src=gathered[nl.ds(w * L_full_N, L_cu_N), :],
        )
        nisa.dma_copy(
            dst=out[nl.ds(_L_CU + w * L_dn_N, L_dn_N), :],
            src=gathered[nl.ds(w * L_full_N + L_cu_N, L_dn_N), :],
        )

    return out
