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

import torch
import torch.nn as nn

from kernels.cross_attention import wan_cross_attn
from utils import _compile


ATTN_SEQLEN_MULTIPLE = 8192


@_compile
class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


@_compile
class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


@_compile
class WanPatchEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.patch_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C, F, H, W = x.shape
        pT, pH, pW = self.patch_size

        x = x.reshape(B, C, F // pT, pT, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, (F // pT) * (H // pH) * (W // pW), C * pT * pH * pW)

        return torch.matmul(x, self.weight.flatten(1).t()) + self.bias


def causal_head_modulate(x, e, modulation):
    e = modulation.unsqueeze(1) + e
    e_shift = e[:, :, 0:1]
    e_scale = e[:, :, 1:2]
    return x * (1 + e_scale) + e_shift


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = _compile(nn.Linear(dim, out_channels))

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self._modulate = _compile(causal_head_modulate)

    def forward(self, x, e):
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        x = self.norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self._modulate(x, e, self.modulation)
        return self.head(x)


def unpatchify(x, out_dim, patch_size, grid_sizes):
    f, h, w = grid_sizes
    pT, pH, pW = patch_size
    u = x.squeeze(0).view(f, h, w, pT, pH, pW, out_dim)
    u = u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
    return u.reshape(out_dim, f * pT, h * pH, w * pW)


def convert_flow_pred_to_x0(flow_pred, xt, sigma_t):
    dtype = flow_pred.dtype
    flow_pred = flow_pred.float()
    xt = xt.float()
    sigma_t = sigma_t.float().reshape(-1, 1, 1, 1)
    return (xt - sigma_t * flow_pred).to(dtype)


def modulated_norm_scale(norm_x, scale, ones, num_frames, frame_seqlen):
    y = norm_x.unflatten(1, (num_frames, frame_seqlen))
    return y * (ones + scale)


def modulated_norm_shift(y, shift):
    return (y + shift).flatten(1, 2)


def modulated_residual(x, y, scale, num_frames, frame_seqlen):
    return x + (y.unflatten(1, (num_frames, frame_seqlen)) * scale).flatten(1, 2)


def modulation_chunk(modulation, e):
    e = modulation.unsqueeze(1) + e
    return (e[:, :, 0:1], e[:, :, 1:2], e[:, :, 2:3],
            e[:, :, 3:4], e[:, :, 4:5], e[:, :, 5:6])


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.cos(freqs).float(), torch.sin(freqs).float()


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.float()
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class WanT2VCrossAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True,
                 eps=1e-6, layer_idx=0):
        assert dim % num_heads == 0
        assert qk_norm is True
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.layer_idx = layer_idx

        self.q = _compile(nn.Linear(dim, dim))
        self.k = _compile(nn.Linear(dim, dim))
        self.v = _compile(nn.Linear(dim, dim))
        self.o = _compile(nn.Linear(dim, dim))

        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        assert crossattn_cache is not None

        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]

        q = q[0].permute(1, 2, 0).contiguous()
        k = k[0].permute(1, 2, 0).contiguous()
        v = v[0].permute(1, 0, 2).contiguous()
        x = wan_cross_attn(q, k, v, softmax_scale=self.softmax_scale)
        x = x.unsqueeze(0).flatten(2)
        return self.o(x)
