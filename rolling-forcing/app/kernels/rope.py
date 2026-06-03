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

import torch
import nki
import nki.language as nl

from torch_neuronx import wrap_nki


@nki.jit
def _causal_rope_rotation_nki(x, cos_sin, num_heads=12, head_dim=128):
    seq_len = x.shape[0]
    N = num_heads
    D = head_dim
    P = nl.tile_size.pmax

    assert seq_len % P == 0
    num_tiles = seq_len // P
    out = nl.ndarray((seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, D)]
        sin_tile = cs_sb[:, nl.ds(D, D)]
        x_sb = nl.load(x[nl.ds(ts, P), :, :])

        out_sb = nl.ndarray((P, N, D), dtype=x.dtype, buffer=nl.sbuf)
        for n in nl.affine_range(N):
            xh = x_sb[:, n, :]
            x_cos = nl.multiply(xh, cos_tile)

            x_swap = nl.ndarray((P, D), dtype=xh.dtype, buffer=nl.sbuf)
            x_swap[:, 0::2] = xh[:, 1::2]
            x_swap[:, 1::2] = xh[:, 0::2]

            x_sin = nl.multiply(x_swap, sin_tile)
            out_sb[:, n, :] = nl.add(x_cos, x_sin)

        nl.store(out[nl.ds(ts, P), :, :], out_sb)

    return out


causal_rope_rotation = wrap_nki(_causal_rope_rotation_nki)


def build_rope_grids(freqs_cos, freqs_sin, sign_pattern, start_frame,
                     F=15, H=30, W=52, head_dim=128):
    """Build 3D RoPE cos/sin grids in PyTorch (no NKI compilation needed)."""
    d = head_dim
    c = d // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    seq_len = F * H * W
    device = freqs_cos.device

    frame_idx = start_frame.flatten() + torch.arange(F, device=device)

    cos_half = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_cos[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_cos[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(seq_len, c)

    sin_half = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_sin[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_sin[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1),
    ], dim=-1).reshape(seq_len, c)

    cos_expanded = cos_half.repeat_interleave(2, dim=-1)
    sin_expanded = sin_half.repeat_interleave(2, dim=-1)
    sign = torch.ones(d, device=device, dtype=sin_expanded.dtype)
    sign[0::2] = -1.0
    sin_signed = sin_expanded * sign.unsqueeze(0)
    cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()

    P = 128
    pad = (P - seq_len % P) % P
    if pad > 0:
        cos_sin = torch.nn.functional.pad(cos_sin, (0, 0, 0, pad))
    return cos_sin
