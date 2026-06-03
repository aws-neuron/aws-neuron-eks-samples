# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import nki.isa as nisa
import nki.language as nl

import nki
from torch_neuronx import wrap_nki


_FRAME_SEQLEN = 1560
_L_CU = 3 * _FRAME_SEQLEN
_L_DN = 15 * _FRAME_SEQLEN


@wrap_nki
@nki.jit
def restore_layout(gathered: nl.ndarray, N: int = 2):
    L_full, dim = gathered.shape

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
