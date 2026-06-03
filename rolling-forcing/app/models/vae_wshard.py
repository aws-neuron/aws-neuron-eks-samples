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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from utils import parallel_state as ps
from kernels.causal_conv3d_cache import (
    causal_conv3d_cache_update_shift,
    causal_conv3d_cache_update_copy,
)
from kernels.extract_w_edges import extract_w_edges


CACHE_T = 2
_GROUP = "vae-sp"


from utils import _compile


def init_vae_parallel_group():
    ps.register_group(_GROUP, dist.group.WORLD)


def destroy_vae_parallel_group():
    ps.destroy_group(_GROUP)




@_compile
class SiLU(nn.Module):
    def forward(self, x):
        ones = torch.full_like(x, 1.0)
        return x / (ones + torch.exp(-x))


@_compile
class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        norm_dim = 1 if self.channel_first else -1
        x_f = x.float()
        denom = (x_f * x_f).sum(dim=norm_dim, keepdim=True).sqrt().clamp(min=1e-12)
        return (x_f / denom * self.scale * self.gamma + self.bias).to(x.dtype)


@_compile
class Upsample(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        return x.reshape(B, C, H, 1, W, 1).expand(B, C, H, 2, W, 2).reshape(
            B, C, H * 2, W * 2)


@_compile
def vae_scaled_dot_product_attention(q, k, v):
    D = q.shape[-1]
    scale = D ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    max_scores = torch.amax(scores, dim=3, keepdim=True)
    exp_scores = torch.exp(scores - max_scores)
    attn = exp_scores / torch.sum(exp_scores, dim=3, keepdim=True)
    return torch.matmul(attn, v)


@_compile
def _causal_conv3d_core(x, cache, weight, bias, stride, dilation, groups, pad_tuple):
    x = torch.cat([cache, x], dim=2)
    x = F.pad(x, pad_tuple)
    return F.conv3d(x, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)


@_compile
def _temporal_interleave(x):
    b, c2, t, h, w = x.shape
    c = c2 // 2
    return x.reshape(b, 2, c, t, h, w).permute(0, 3, 1, 2, 4, 5).reshape(
        b * t * 2, c, h, w)


@_compile
def _split_qkv(qkv):
    c = qkv.shape[1] // 3
    bt = qkv.shape[0]
    qkv = qkv.reshape(bt, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
    return qkv[:, :, :, 0:c], qkv[:, :, :, c:2*c], qkv[:, :, :, 2*c:3*c]


@_compile
def _halo_cat_pad_conv2d(x, halo_left, halo_right,
                         weight, bias, pad_tuple,
                         stride, dilation, groups):
    parts = []
    if halo_left is not None:
        parts.append(halo_left)
    parts.append(x)
    if halo_right is not None:
        parts.append(halo_right)
    x = torch.cat(parts, dim=3) if len(parts) > 1 else parts[0]
    x = F.pad(x, pad_tuple)
    return F.conv2d(x, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)


@_compile
def _halo_cat_pad_conv3d(x_tc, halo_left, halo_right,
                         weight, bias, pad_tuple,
                         stride, dilation, groups):
    parts = []
    if halo_left is not None:
        parts.append(halo_left)
    parts.append(x_tc)
    if halo_right is not None:
        parts.append(halo_right)
    x = torch.cat(parts, dim=4) if len(parts) > 1 else parts[0]
    x = F.pad(x, pad_tuple)
    return F.conv3d(x, weight, bias, stride, padding=0,
                    dilation=dilation, groups=groups)




def _extract_edges(x, radius):
    orig_shape = x.shape
    W_local = orig_shape[-1]
    P = x.numel() // W_local
    x_t = x.reshape(P, W_local).transpose(0, 1).contiguous()
    edges_t = torch.empty(2 * radius, P, dtype=x.dtype, device=x.device)
    extract_w_edges(x_t, edges_t, W_local, radius)
    ndim = len(orig_shape)
    edges = edges_t.reshape(2, radius, *orig_shape[:-1])
    return edges.permute(0, *range(2, 2 + ndim - 1), 1).contiguous()


def _halo_exchange_w(x, radius, group_name=_GROUP):
    world = ps.get_world_size(group_name)
    rank = ps.get_rank(group_name)
    if world == 1:
        return None, None

    edges_local = _extract_edges(x, radius)
    edges_all = torch.empty(
        (world * edges_local.shape[0],) + edges_local.shape[1:],
        dtype=x.dtype, device=x.device,
    )
    ps.all_gather_into_tensor(edges_all, edges_local, group_name)
    edges_all = edges_all.reshape((world,) + edges_local.shape)

    halo_left = edges_all[rank - 1, 1] if rank > 0 else None
    halo_right = edges_all[rank + 1, 0] if rank < world - 1 else None
    return halo_left, halo_right


def _all_gather_w(x, group_name=_GROUP):
    world = ps.get_world_size(group_name)
    if world == 1:
        return x
    gathered = torch.empty(
        (world * x.shape[0],) + x.shape[1:], dtype=x.dtype, device=x.device,
    )
    ps.all_gather_into_tensor(gathered, x.contiguous(), group_name)
    gathered = gathered.reshape((world,) + x.shape)
    ndim = x.dim()
    perm = tuple(range(1, ndim)) + (0, ndim)
    gathered = gathered.permute(*perm).contiguous()
    return gathered.reshape(x.shape[:-1] + (x.shape[-1] * world,))




class CausalConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight,
                                 a=torch.nn.init.calculate_gain("linear"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.original_padding = padding
        self.spatial_temporal_padding = (
            padding[2], padding[2],
            padding[1], padding[1],
            2 * padding[0] - CACHE_T, 0,
        )
        self.cache = None

    def forward(self, x):
        world = ps.get_world_size(_GROUP)
        T = x.shape[2]
        kW = self.kernel_size[2]
        needs_halo = (kW > 1) and (world > 1)

        if self.cache is None:
            B, C, _, H, W_local = x.shape
            self.cache = torch.zeros(
                B, C, CACHE_T, H, W_local, dtype=x.dtype, device=x.device)

        if not needs_halo:
            output = _causal_conv3d_core(
                x, self.cache, self.weight, self.bias, self.stride,
                self.dilation, self.groups, self.spatial_temporal_padding,
            )
        else:
            x_tc = torch.cat([self.cache, x], dim=2)
            radius = kW // 2
            halo_left, halo_right = _halo_exchange_w(x_tc, radius)

            pad_W_l, pad_W_r, pad_H_l, pad_H_r, pad_T_l, pad_T_r = \
                self.spatial_temporal_padding
            new_pad_W_l = 0 if halo_left is not None else pad_W_l
            new_pad_W_r = 0 if halo_right is not None else pad_W_r

            output = _halo_cat_pad_conv3d(
                x_tc, halo_left, halo_right,
                self.weight, self.bias,
                (new_pad_W_l, new_pad_W_r, pad_H_l, pad_H_r, pad_T_l, pad_T_r),
                self.stride, self.dilation, self.groups,
            )

        C, H, W_local = x.shape[1], x.shape[3], x.shape[4]
        HW = H * W_local
        cache_2d = self.cache.view(C, CACHE_T * HW)
        if T < CACHE_T:
            causal_conv3d_cache_update_shift(cache_2d, x.view(C, T * HW), HW)
        else:
            causal_conv3d_cache_update_copy(cache_2d, x.view(C, T * HW))

        return output

    def clear_cache(self):
        self.cache = None


class Conv2d3x3(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias,
        )
        assert self.kernel_size == (3, 3) and self.padding == (1, 1)

    def forward(self, x):
        world = ps.get_world_size(_GROUP)
        if world == 1:
            return super().forward(x)

        halo_left, halo_right = _halo_exchange_w(x, radius=1)
        new_pad_W_l = 0 if halo_left is not None else 1
        new_pad_W_r = 0 if halo_right is not None else 1

        return _halo_cat_pad_conv2d(
            x, halo_left, halo_right,
            self.weight, self.bias,
            (new_pad_W_l, new_pad_W_r, 1, 1),
            self.stride, self.dilation, self.groups,
        )


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.first_video_frame = True

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(), Conv2d3x3(dim, dim // 2, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(), Conv2d3x3(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.size()

        if self.mode == "upsample3d":
            if self.first_video_frame:
                self.first_video_frame = False
            else:
                x = self.time_conv(x)
                x = _temporal_interleave(x)

        if x.dim() == 5:
            x = x.transpose(1, 2).reshape(b * t, c, h, w).contiguous()

        x = self.resample(x)

        t_out = x.shape[0] // b
        x = x.reshape(b, t_out, x.shape[1], x.shape[2], x.shape[3]).transpose(1, 2).contiguous()
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert dropout == 0.0

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), SiLU(), nn.Identity(),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (_compile(nn.Conv3d(in_dim, out_dim, 1))
                         if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        h = self.shortcut(x)
        x = self.residual(x)
        return x + h


@_compile
class AttentionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        world = ps.get_world_size(_GROUP)
        rank = ps.get_rank(_GROUP)
        W_local = x.shape[-1]

        if world > 1:
            w_start = rank * W_local
            w_end = w_start + W_local
            x_full = _all_gather_w(x)
        else:
            x_full = x

        identity = x_full
        b, c, t, h, w = x_full.size()
        x_work = x_full.transpose(1, 2).reshape(b * t, c, h, w)

        x_work = self.norm(x_work)
        qkv = self.to_qkv(x_work)
        q, k, v = _split_qkv(qkv)

        x_work = vae_scaled_dot_product_attention(q, k, v)
        x_work = x_work.squeeze(1).permute(0, 2, 1)
        x_work = x_work.reshape(b * t, c, h, w)

        x_work = self.proj(x_work)
        x_work = x_work.reshape(b, t, c, h, w).transpose(1, 2)
        y_full = x_work + identity

        if world > 1:
            return y_full[..., w_start:w_end].contiguous()
        return y_full




class Decoder3d(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2, attn_scales=[],
                 temperal_upsample=[False, True, True], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i in (1, 2, 3):
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        x = self.head(x)
        return x


class WanVAE_(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2, attn_scales=[],
                 temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.temperal_upsample = temperal_downsample[::-1]
        self.conv2 = _compile(nn.Conv3d(z_dim, z_dim, 1))
        self.decoder = Decoder3d(
            dim, z_dim, dim_mult, num_res_blocks, attn_scales,
            self.temperal_upsample, dropout,
        )
        self.clear_cache()

    def cached_decode(self, z, scale, chunk_idx=None, batch_frames=True):
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]

        x = self.conv2(z)
        T = z.shape[2]

        if batch_frames and chunk_idx is not None and chunk_idx > 0:
            return [self.decoder(x)]
        return [self.decoder(x[:, :, i:i + 1, :, :]) for i in range(T)]

    def clear_cache(self):
        for module in self.decoder.modules():
            if isinstance(module, CausalConv3d):
                module.clear_cache()
            elif isinstance(module, Resample):
                module.first_video_frame = True


class WanVAEWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        cfg = dict(
            dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
            attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0,
        )
        self.model = WanVAE_(**cfg)

        state_dict = torch.load(
            "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", map_location="cpu")
        decoder_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("encoder.") and not k.startswith("conv1.")
        }
        self._load_weights(decoder_state_dict)
        self.model.eval().requires_grad_(False)

    def _load_weights(self, ckpt_sd):
        model_keys = set(self.model.state_dict().keys())
        mapped_sd = {}
        for k, v in ckpt_sd.items():
            if k in model_keys:
                mapped_sd[k] = v
            else:
                parts = k.split(".")
                for i in range(len(parts)):
                    candidate = ".".join(parts[:i+1] + ["_orig_mod"] + parts[i+1:])
                    if candidate in model_keys:
                        mapped_sd[candidate] = v
                        break
                else:
                    mapped_sd[k] = v
        self.model.load_state_dict(mapped_sd, strict=False)

    def decode_to_pixel_device(self, latent, use_cache=False,
                               chunk_idx=None, batch_frames=True):
        assert latent.shape[0] == 1
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        if use_cache:
            return self.model.cached_decode(
                zs, scale, chunk_idx=chunk_idx, batch_frames=batch_frames)
        self.model.clear_cache()
        outputs = [self.model.decoder(
            self.model.conv2(
                zs / scale[1].view(1, 16, 1, 1, 1) + scale[0].view(1, 16, 1, 1, 1)
            )[:, :, i:i+1, :, :])
            for i in range(zs.shape[2])]
        self.model.clear_cache()
        return outputs

    @staticmethod
    def postprocess_pixels(frame_outputs):
        cpu_frames = [f.cpu() for f in frame_outputs]
        out = torch.cat(cpu_frames, dim=2)
        return out.float().clamp_(-1, 1).permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent, use_cache=False,
                        chunk_idx=None, batch_frames=True):
        return self.postprocess_pixels(
            self.decode_to_pixel_device(
                latent, use_cache, chunk_idx, batch_frames=batch_frames))


def build_vae(dtype=torch.bfloat16):
    return WanVAEWrapper().to(dtype=dtype).to("neuron")
