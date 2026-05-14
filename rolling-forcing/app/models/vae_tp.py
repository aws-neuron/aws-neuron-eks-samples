"""VAE Decoder 2-way Channel Tensor Parallelism for Neuron.

Shards CausalConv3d and Conv2d layers in the VAE decoder across 2 NeuronCores
using channel parallelism. Each rank computes half the output channels.

TP Pattern (same as DiT linear TP but for convolutions):
  - Column-parallel conv: output channels sharded (no communication)
  - Row-parallel conv: input channels sharded, all-reduce after forward
  - ResidualBlock: first conv is column-parallel, second is row-parallel
  - AttentionBlock: QKV conv is column-parallel, proj conv is row-parallel
  - Shortcut convs: replicated (1x1, cheap)

Usage:
  from models.vae_tp import shard_vae_decoder_tp, create_vae_tp_group
  vae_tp_group = create_vae_tp_group(vae_ranks=[0, 1])
  shard_vae_decoder_tp(vae.model.decoder, vae_tp_rank, vae_tp_degree, vae_tp_group)

Requires: VAE_TP_DEGREE env var set to 2 (default 1 = no sharding).
NKI kernels are NOT used when VAE_TP_DEGREE > 1 (channel dims don't align to P=128).
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAE TP process group
# ---------------------------------------------------------------------------

_VAE_TP_GROUP: Optional[dist.ProcessGroup] = None
_VAE_TP_RANK: int = 0
_VAE_TP_WORLD_SIZE: int = 1


def create_vae_tp_group(vae_ranks: List[int]) -> Optional[dist.ProcessGroup]:
    """Create a process group for VAE tensor parallelism.

    Args:
        vae_ranks: List of global ranks that participate in VAE TP.
                   e.g. [0, 1] for 2-way TP on first 2 NeuronCores.

    Returns:
        The process group (or None if single rank).
    """
    global _VAE_TP_GROUP, _VAE_TP_RANK, _VAE_TP_WORLD_SIZE

    if len(vae_ranks) <= 1:
        _VAE_TP_RANK = 0
        _VAE_TP_WORLD_SIZE = 1
        _VAE_TP_GROUP = None
        return None

    group = dist.new_group(vae_ranks)
    global_rank = dist.get_rank()

    if global_rank in vae_ranks:
        _VAE_TP_GROUP = group
        _VAE_TP_RANK = vae_ranks.index(global_rank)
        _VAE_TP_WORLD_SIZE = len(vae_ranks)

    logger.info(f"[VAE-TP] Created group ranks={vae_ranks}, "
                f"global_rank={global_rank}, vae_tp_rank={_VAE_TP_RANK}")
    return group


def get_vae_tp_group():
    return _VAE_TP_GROUP


def get_vae_tp_rank():
    return _VAE_TP_RANK


def get_vae_tp_world_size():
    return _VAE_TP_WORLD_SIZE


@torch.compiler.disable
def vae_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce sum across VAE TP group."""
    if _VAE_TP_WORLD_SIZE <= 1:
        return x
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_VAE_TP_GROUP)
    return x


# ---------------------------------------------------------------------------
# Column-parallel Conv3d: output channels sharded
# ---------------------------------------------------------------------------

class ColumnParallelCausalConv3d(nn.Module):
    """CausalConv3d with output channels sharded across TP ranks.

    Each rank holds out_channels // tp_degree output channels.
    No communication in forward — output is a local channel shard.
    """

    def __init__(self, original_conv, tp_rank: int, tp_degree: int):
        super().__init__()
        out_ch = original_conv.weight.shape[0]
        in_ch = original_conv.weight.shape[1]
        chunk = out_ch // tp_degree
        start = tp_rank * chunk
        end = start + chunk

        # Shard weight: [out_ch, in_ch, kT, kH, kW] → [chunk, in_ch, kT, kH, kW]
        self.weight = nn.Parameter(original_conv.weight.data[start:end].contiguous())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data[start:end].contiguous())
        else:
            self.bias = None

        # Copy padding info from CausalConv3d
        self._padding = original_conv._padding
        self.stride = original_conv.stride
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return F.conv3d(x, self.weight, self.bias,
                        stride=self.stride, dilation=self.dilation,
                        groups=self.groups)


# ---------------------------------------------------------------------------
# Row-parallel Conv3d: input channels sharded, all-reduce output
# ---------------------------------------------------------------------------

class RowParallelCausalConv3d(nn.Module):
    """CausalConv3d with input channels sharded across TP ranks.

    Each rank holds weight [out_ch, in_ch // tp_degree, kT, kH, kW].
    Forward: local conv → all-reduce sum → add bias.
    """

    def __init__(self, original_conv, tp_rank: int, tp_degree: int):
        super().__init__()
        out_ch = original_conv.weight.shape[0]
        in_ch = original_conv.weight.shape[1]
        chunk = in_ch // tp_degree
        start = tp_rank * chunk
        end = start + chunk

        # Shard weight along input channel dim
        self.weight = nn.Parameter(original_conv.weight.data[:, start:end].contiguous())
        # Bias added after all-reduce (full size, only one copy needed)
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None

        self._padding = original_conv._padding
        self.stride = original_conv.stride
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        out = F.conv3d(x, self.weight, None,  # no bias yet
                       stride=self.stride, dilation=self.dilation,
                       groups=self.groups)
        # All-reduce across VAE TP ranks
        out = vae_all_reduce_sum(out)
        # Add bias after all-reduce
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Column-parallel Conv2d (for AttentionBlock QKV and Upsample convs)
# ---------------------------------------------------------------------------

class ColumnParallelConv2d(nn.Module):
    """Conv2d with output channels sharded."""

    def __init__(self, original_conv, tp_rank: int, tp_degree: int):
        super().__init__()
        out_ch = original_conv.weight.shape[0]
        chunk = out_ch // tp_degree
        start = tp_rank * chunk
        end = start + chunk

        self.weight = nn.Parameter(original_conv.weight.data[start:end].contiguous())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data[start:end].contiguous())
        else:
            self.bias = None
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation)


class RowParallelConv2d(nn.Module):
    """Conv2d with input channels sharded, all-reduce output."""

    def __init__(self, original_conv, tp_rank: int, tp_degree: int):
        super().__init__()
        in_ch = original_conv.weight.shape[1]
        chunk = in_ch // tp_degree
        start = tp_rank * chunk
        end = start + chunk

        self.weight = nn.Parameter(original_conv.weight.data[:, start:end].contiguous())
        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation

    def forward(self, x):
        out = F.conv2d(x, self.weight, None,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation)
        out = vae_all_reduce_sum(out)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


# ---------------------------------------------------------------------------
# Sharding functions
# ---------------------------------------------------------------------------

def _is_causal_conv3d(module):
    """Check if module is a CausalConv3d (by class name, avoid import)."""
    return type(module).__name__ == 'CausalConv3d'


def _shard_residual_block(block, tp_rank, tp_degree):
    """Shard a ResidualBlock's convolutions for channel TP.

    Pattern: first CausalConv3d → column-parallel, second → row-parallel.
    Shortcut conv stays replicated (identity or 1x1, cheap).
    """
    conv_indices = []
    for i, layer in enumerate(block.residual):
        if _is_causal_conv3d(layer):
            conv_indices.append(i)

    if len(conv_indices) >= 2:
        # First conv: column-parallel (shard output channels)
        idx0 = conv_indices[0]
        block.residual[idx0] = ColumnParallelCausalConv3d(
            block.residual[idx0], tp_rank, tp_degree)

        # Second conv: row-parallel (shard input channels, all-reduce)
        idx1 = conv_indices[1]
        block.residual[idx1] = RowParallelCausalConv3d(
            block.residual[idx1], tp_rank, tp_degree)

    # Data flow through residual path (Megatron-style column→row TP):
    #   Input: FULL channels (from previous row-parallel all-reduce)
    #   residual[0] RMS_norm(in_dim): FULL → keep full, DON'T shard
    #   residual[2] Conv(in→out) column-parallel: FULL input → LOCAL output
    #   residual[3] RMS_norm(out_dim): LOCAL → MUST shard gamma to out_dim/tp
    #   residual[6] Conv(out→out) row-parallel: LOCAL input → all-reduce → FULL output
    #   Shortcut: FULL→FULL (keep replicated)
    #   x + h: FULL + FULL ✓
    from wan.modules.vae import RMS_norm
    for i, layer in enumerate(block.residual):
        if isinstance(layer, RMS_norm) and len(conv_indices) >= 2 and i > conv_indices[0]:
            # Only shard the norm AFTER the first (column-parallel) conv
            full_ch = layer.gamma.shape[0]
            chunk = full_ch // tp_degree
            start = tp_rank * chunk
            end = start + chunk
            layer.gamma = nn.Parameter(layer.gamma.data[start:end].contiguous())
            if isinstance(layer.bias, nn.Parameter):
                layer.bias = nn.Parameter(layer.bias.data[start:end].contiguous())
            layer.scale = chunk ** 0.5

    # Shortcut: input is FULL, output is FULL → keep replicated (no sharding)

    return block


def _shard_attention_block(block, tp_rank, tp_degree):
    """Shard AttentionBlock for channel TP.

    QKV conv2d → column-parallel, proj conv2d → row-parallel.
    NKI kernels disabled (320 % 128 != 0), uses PyTorch SDPA fallback.
    """
    # Norm: input is FULL channels (AttentionBlock receives FULL from previous row-parallel)
    # Keep norm replicated — it operates on full channels.

    # to_qkv: Conv2d(dim, dim*3, 1) → column-parallel (input FULL, output LOCAL 3*dim/tp)
    block.to_qkv = ColumnParallelConv2d(block.to_qkv, tp_rank, tp_degree)

    # proj: Conv2d(dim, dim, 1) → row-parallel
    block.proj = RowParallelConv2d(block.proj, tp_rank, tp_degree)

    # Store local dim for attention reshape
    block._tp_local_dim = block.dim // tp_degree
    block._tp_degree = tp_degree

    # Override forward to use sharded attention (PyTorch SDPA, no NKI)
    original_forward = block.forward

    def tp_forward(x):
        from einops import rearrange
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")

        # RMS norm on local channels (already sharded)
        x_normed = block.norm(x)

        # QKV via column-parallel conv2d (local channels)
        local_dim = block._tp_local_dim
        qkv = block.to_qkv(x_normed)  # (BT, 3*local_dim, H, W)
        q, k, v = (
            qkv.reshape(b * t, 1, local_dim * 3, -1)
            .permute(0, 1, 3, 2).contiguous()
            .chunk(3, dim=-1))
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, local_dim, h, w)

        # Proj via row-parallel conv2d (all-reduce inside)
        x = block.proj(x)

        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity

    block.forward = tp_forward
    return block


def _shard_resample(resample_module, tp_rank, tp_degree):
    """Shard Resample module's Conv2d and CausalConv3d.

    Upsample: Conv2d(dim, dim, 3) in resample.resample[1]
    Time conv: CausalConv3d(dim, dim*2, (3,1,1)) — column-parallel
    Downsample: similar pattern.
    """
    mode = resample_module.mode

    if mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
        # The spatial conv2d in resample.resample is after upsample/before downsample
        # It's a full conv2d(dim, dim, 3, padding=1) — both input and output are full channels
        # Since the input to this comes from a row-parallel (all-reduced = full channels),
        # and output goes to next block's column-parallel, keep it replicated.
        pass

    if mode in ("upsample3d", "downsample3d"):
        if hasattr(resample_module, 'time_conv'):
            # time_conv: CausalConv3d(dim, dim*2) for upsample3d
            # or CausalConv3d(dim, dim) for downsample3d
            # Input is full channels (after all-reduce), output goes to reshape
            # Keep replicated — temporal conv is cheap and has complex reshape logic
            pass

    return resample_module


def shard_vae_decoder_tp(decoder, tp_rank: int, tp_degree: int):
    """Apply 2-way channel TP to the VAE Decoder3d in-place.

    Shards ResidualBlock conv3d layers and AttentionBlock conv2d layers.
    Resample and shortcut convs stay replicated for correctness.

    Args:
        decoder: Decoder3d instance with full weights loaded.
        tp_rank: VAE TP rank (0 or 1 for 2-way).
        tp_degree: VAE TP degree (2).
    """
    logger.info(f"[VAE-TP] Sharding decoder: tp_rank={tp_rank}, tp_degree={tp_degree}")

    # conv1: CausalConv3d(z_dim, dims[0], 3) — keep replicated (tiny: 48→640, ~830K params)
    # Output is FULL channels. All subsequent blocks expect FULL input at their entry.

    # Shard middle blocks
    for i, layer in enumerate(decoder.middle):
        if type(layer).__name__ == 'ResidualBlock':
            decoder.middle[i] = _shard_residual_block(layer, tp_rank, tp_degree)
        elif type(layer).__name__ == 'AttentionBlock':
            decoder.middle[i] = _shard_attention_block(layer, tp_rank, tp_degree)

    # Shard upsample blocks
    # Wan 2.1: decoder.upsamples is a flat nn.Sequential of ResidualBlock + Resample
    # (NOT nested Up_ResidualBlock like Wan 2.2)
    for i, layer in enumerate(decoder.upsamples):
        if type(layer).__name__ == 'ResidualBlock':
            decoder.upsamples[i] = _shard_residual_block(layer, tp_rank, tp_degree)
        elif type(layer).__name__ == 'AttentionBlock':
            decoder.upsamples[i] = _shard_attention_block(layer, tp_rank, tp_degree)
        elif type(layer).__name__ == 'Resample':
            decoder.upsamples[i] = _shard_resample(layer, tp_rank, tp_degree)

    # Head: [RMS_norm(out_dim), SiLU, CausalConv3d(out_dim, 12, 3)]
    # After last ResidualBlock row-parallel all-reduce → FULL channels in.
    # Keep head replicated (tiny: 128→12, ~50K params).

    total_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"[VAE-TP] Decoder sharded: {total_params / 1e6:.1f}M params (local, rank {tp_rank})")

    return decoder


def shard_vae_model_tp(vae_model, tp_rank: int, tp_degree: int):
    """Shard the full WanVAE_ model's decoder for TP.

    Only the decoder is sharded (decode path). Encoder stays replicated.
    conv2 (z_dim→z_dim, 1x1) stays replicated — tiny and feeds into sharded decoder.

    Args:
        vae_model: WanVAE_ instance (vae.model).
        tp_rank: VAE TP rank.
        tp_degree: VAE TP degree.
    """
    shard_vae_decoder_tp(vae_model.decoder, tp_rank, tp_degree)
    # Mark model as TP-sharded
    vae_model._vae_tp_degree = tp_degree
    vae_model._vae_tp_rank = tp_rank
    return vae_model
