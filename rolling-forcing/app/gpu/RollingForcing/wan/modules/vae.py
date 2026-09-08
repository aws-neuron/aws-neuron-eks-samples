# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
]

# ─── NKI VAE Kernel Loading for Neuron ─────────────────────────────────
_USE_NKI_VAE = os.environ.get("USE_NKI_VAE", "1") == "1"
_nki_conv2d_k1 = None
_nki_conv2d_k3 = None
_nki_self_attn = None
_NKI_VAE_AVAILABLE = False

if _USE_NKI_VAE:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.vae_conv2d import vae_conv2d_k1, vae_conv2d_k3_shifted
        from kernels.vae_attention import vae_self_attention
        _nki_conv2d_k1 = wrap_nki(vae_conv2d_k1)
        _nki_conv2d_k3 = wrap_nki(vae_conv2d_k3_shifted)
        _nki_self_attn = wrap_nki(vae_self_attention)
        _NKI_VAE_AVAILABLE = True
        print("[vae.py] NKI VAE kernels: ✓ LOADED (conv2d_k1, conv2d_k3, self_attn)")
    except Exception as e:
        print(f"[vae.py] NKI VAE kernels: ✗ FAILED ({e}) — using PyTorch fallback")
else:
    print("[vae.py] NKI VAE kernels: — SKIPPED (USE_NKI_VAE=0)")


def _is_neuron_tensor(x):
    """Check if tensor is on a Neuron device."""
    return x.device.type == "neuron" or (hasattr(x.device, 'type') and 'xla' in str(x.device))


def _nki_conv2d_forward(weight, bias, x_2d, kernel_size, C_in, C_out, H, W, padding=0):
    """Run spatial conv2d via NKI kernel. x_2d is (BT, C, H, W)."""
    BT = x_2d.shape[0]
    device = x_2d.device
    P = 128
    SPATIAL_TILE = 512

    results = []
    for bt in range(BT):
        frame = x_2d[bt]  # (C_in, H, W)
        HW = H * W

        if kernel_size == 1:
            inp_flat = frame.reshape(C_in, HW).to(torch.bfloat16)
            C_in_p = ((C_in + P - 1) // P) * P
            C_out_p = ((C_out + P - 1) // P) * P
            HW_p = ((HW + SPATIAL_TILE - 1) // SPATIAL_TILE) * SPATIAL_TILE

            inp_padded = torch.zeros(C_in_p, HW_p, dtype=torch.bfloat16, device=device)
            inp_padded[:C_in, :HW] = inp_flat

            w = weight.reshape(C_out, C_in).to(torch.bfloat16)
            w_T = torch.zeros(C_in_p, C_out_p, dtype=torch.bfloat16, device=device)
            w_T[:C_in, :C_out] = w.T

            b_padded = torch.zeros(C_out_p, 1, dtype=torch.bfloat16, device=device)
            if bias is not None:
                b_padded[:C_out, 0] = bias.to(torch.bfloat16)

            out = _nki_conv2d_k1(inp_padded, w_T, b_padded, HW)
            results.append(out[:C_out, :HW].reshape(C_out, H, W))

        elif kernel_size == 3:
            inp_flat = frame.reshape(C_in, HW).to(torch.bfloat16)
            HW_p = ((HW + SPATIAL_TILE - 1) // SPATIAL_TILE) * SPATIAL_TILE
            C_in_p = ((C_in + P - 1) // P) * P
            C_out_p = ((C_out + P - 1) // P) * P

            frame_3d = frame.reshape(C_in, H, W)
            x_padded = F.pad(frame_3d.float(), (padding, padding, padding, padding))
            shifts = []
            for kh in range(3):
                for kw in range(3):
                    window = x_padded[:, kh:kh + H, kw:kw + W]
                    shifts.append(window.reshape(C_in, HW))
            shifted = torch.stack(shifts, dim=0).reshape(9 * C_in, HW).to(torch.bfloat16)

            shifted_padded = torch.zeros(9 * C_in_p, HW_p, dtype=torch.bfloat16, device=device)
            for k_idx in range(9):
                shifted_padded[k_idx * C_in_p:k_idx * C_in_p + C_in, :HW] = \
                    shifted[k_idx * C_in:k_idx * C_in + C_in, :HW]

            w_4d = weight.reshape(C_out, C_in, 3, 3)
            w_T_padded = torch.zeros(C_in_p * 9, C_out_p, dtype=torch.bfloat16, device=device)
            for k_idx in range(9):
                kh, kw = k_idx // 3, k_idx % 3
                w_slice = w_4d[:, :, kh, kw].T.to(torch.bfloat16)
                w_T_padded[k_idx * C_in_p:k_idx * C_in_p + C_in, :C_out] = w_slice

            b_padded = torch.zeros(C_out_p, 1, dtype=torch.bfloat16, device=device)
            if bias is not None:
                b_padded[:C_out, 0] = bias.to(torch.bfloat16)

            out = _nki_conv2d_k3(shifted_padded, w_T_padded, b_padded, HW)
            results.append(out[:C_out, :HW].reshape(C_out, H, W))

    return torch.stack(results, dim=0)  # (BT, C_out, H, W)

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x.contiguous(), dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    Uses standard PyTorch ops (Conv2d + SDPA) which work in Neuron eager mode.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)

        # Use NKI kernels if available and on Neuron
        if _NKI_VAE_AVAILABLE and _is_neuron_tensor(x):
            # NKI path: use vae_conv2d_k1 for QKV/proj, vae_self_attention for SDPA
            P = 128
            CHUNK = 512
            seq = h * w
            seq_padded = ((seq + CHUNK - 1) // CHUNK) * CHUNK

            # QKV via NKI conv2d_k1
            qkv = _nki_conv2d_forward(
                self.to_qkv.weight, self.to_qkv.bias,
                x, kernel_size=1, C_in=c, C_out=c * 3, H=h, W=w)
            # qkv: (BT, 3*c, h, w)

            results = []
            for bt_idx in range(b * t):
                qkv_frame = qkv[bt_idx]  # (3*c, h, w)
                qkv_flat = qkv_frame.reshape(3 * c, seq)
                q_flat, k_flat, v_flat = qkv_flat.chunk(3, dim=0)  # each (c, seq)

                # NKI attention expects: q(1,d,seq), k(1,d,seq), v(1,seq,d)
                q_nki = q_flat.unsqueeze(0).to(torch.bfloat16)  # (1, c, seq)
                k_nki = k_flat.unsqueeze(0).to(torch.bfloat16)  # (1, c, seq)
                v_nki = v_flat.T.unsqueeze(0).to(torch.bfloat16)  # (1, seq, c)

                # Pad seq to multiple of 512
                if seq < seq_padded:
                    q_nki = F.pad(q_nki, (0, seq_padded - seq))
                    k_nki = F.pad(k_nki, (0, seq_padded - seq))
                    v_nki = F.pad(v_nki, (0, 0, 0, seq_padded - seq))

                identity_mat = torch.eye(P, dtype=torch.bfloat16, device=x.device)
                scale = 1.0 / math.sqrt(c)

                out_nki = _nki_self_attn(q_nki, k_nki, v_nki, identity_mat, softmax_scale=scale)
                # out_nki: (seq_padded, 1, c) → trim to (seq, c)
                out_frame = out_nki[:seq, 0, :]  # (seq, c)
                out_frame = out_frame.T.reshape(c, h, w)  # (c, h, w)
                results.append(out_frame)

            x = torch.stack(results, dim=0)  # (BT, c, h, w)

            # Proj via NKI conv2d_k1
            x = _nki_conv2d_forward(
                self.proj.weight, self.proj.bias,
                x, kernel_size=1, C_in=c, C_out=c, H=h, W=w)
        else:
            # PyTorch fallback
            q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                             -1).permute(0, 1, 3,
                                                         2).contiguous().chunk(
                                                             3, dim=-1)
            x = F.scaled_dot_product_attention(q, k, v)
            x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
            x = self.proj(x)

        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)
        self.clear_cache()

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z.contiguous() / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z.contiguous() / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        return out

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
