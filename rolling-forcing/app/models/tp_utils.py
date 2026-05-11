"""Tensor Parallelism utilities for Wan2.1-T2V-14B on Trainium.

Implements column-parallel and row-parallel linear layers for sharding
the DiT model across 4 NeuronCores. All 4 cores run DiT in TP mode
while TE and VAE are replicated on each rank.

TP Strategy:
  - Q/K/V projections → ColumnParallelLinear (split output dim by heads)
  - O projection → RowParallelLinear (split input dim, all-reduce output)
  - FFN fc1/up → ColumnParallelLinear (split hidden dim)
  - FFN fc2/down → RowParallelLinear (split input dim, all-reduce output)
  - Norms, embeddings, modulation → Replicated (small, needed for correctness)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Process group management
# ---------------------------------------------------------------------------

_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1


def init_tp_group(tp_degree: int = 4):
    """Initialize tensor parallelism process group.

    Must be called after torch.distributed.init_process_group().
    On Trainium, the 4 NeuronCores within a chip form the TP group.

    Args:
        tp_degree: Number of ranks in the TP group (default 4 for trn2 chip).
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE

    if not dist.is_initialized():
        # Single-process fallback for testing
        _TP_RANK = 0
        _TP_WORLD_SIZE = 1
        _TP_GROUP = None
        print(f"[TP] Running in single-process mode (no distributed)")
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert world_size % tp_degree == 0, (
        f"World size {world_size} must be divisible by tp_degree {tp_degree}")

    # Create TP groups: ranks [0,1,2,3], [4,5,6,7], etc.
    num_groups = world_size // tp_degree
    for i in range(num_groups):
        ranks = list(range(i * tp_degree, (i + 1) * tp_degree))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group
            _TP_RANK = rank - i * tp_degree
            _TP_WORLD_SIZE = tp_degree

    print(f"[TP] Initialized: rank={_TP_RANK}/{_TP_WORLD_SIZE}, "
          f"global_rank={rank}/{world_size}")


def get_tp_group() -> Optional[dist.ProcessGroup]:
    """Get the tensor parallel process group."""
    return _TP_GROUP


def get_tp_rank() -> int:
    """Get the local TP rank (0 to tp_degree-1)."""
    return _TP_RANK


def get_tp_world_size() -> int:
    """Get the TP world size (tp_degree)."""
    return _TP_WORLD_SIZE


# ---------------------------------------------------------------------------
# All-reduce communication
# ---------------------------------------------------------------------------

# Maximum bytes per all-reduce call. The Neuron NRT rejects certain large
# payload sizes for multi-rank groups (e.g. 59,904,000 bytes fails for TP=8).
# Chunking to ≤8MB per call avoids hitting unsupported size/topology combos.
_MAX_ALLREDUCE_BYTES = int(os.environ.get("MAX_ALLREDUCE_BYTES", 8 * 1024 * 1024))


@torch.compiler.disable
def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce (sum) across TP group with chunking for NRT size limits.

    Decorated with @torch.compiler.disable to force a graph break — the
    compiled FFN NEFF handles only matmuls, then all-reduce runs in eager.

    The Neuron NRT rejects certain all-reduce payload sizes for TP=8.
    We chunk large tensors into ≤8MB pieces to stay within limits.
    """
    if _TP_WORLD_SIZE <= 1:
        return x

    elem_size = x.element_size()  # 2 for bf16, 4 for fp32
    total_bytes = x.numel() * elem_size

    if total_bytes <= _MAX_ALLREDUCE_BYTES:
        # Small tensor — single all-reduce
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    else:
        # Large tensor — chunk to stay within NRT limits
        max_elements = _MAX_ALLREDUCE_BYTES // elem_size
        flat = x.view(-1)
        numel = flat.numel()
        for start in range(0, numel, max_elements):
            end = min(start + max_elements, numel)
            chunk = flat[start:end]
            dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=_TP_GROUP)

    return x


# ---------------------------------------------------------------------------
# TP-aware RMSNorm (for QK norms that must compute global RMS across ranks)
# ---------------------------------------------------------------------------

class TPRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across all TP ranks.

    Standard RMSNorm computes: x * rsqrt(mean(x², dim=-1) + eps) * weight
    With TP, each rank only sees dim//tp_degree features. Computing mean(x²)
    locally gives a WRONG normalization factor because each rank's heads have
    different magnitudes.

    This version all-reduces sum(x²) across ranks before computing rsqrt,
    giving the mathematically correct global RMS normalization.
    """

    def __init__(self, local_dim: int, global_dim: int, eps: float = 1e-5):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        # Local sum of squares: [*, 1]
        local_sum_sq = x_float.pow(2).sum(dim=-1, keepdim=True)
        # All-reduce to get global sum of squares across all TP ranks
        global_sum_sq = all_reduce_sum(local_sum_sq.clone())
        # Global RMS: sqrt(sum_sq / global_dim)
        rms_inv = torch.rsqrt(global_sum_sq / self.global_dim + self.eps)
        return (x_float * rms_inv).type_as(x) * self.weight

    def extra_repr(self):
        return (f'local_dim={self.local_dim}, global_dim={self.global_dim}, '
                f'eps={self.eps}')


# ---------------------------------------------------------------------------
# Parallel Linear layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across TP ranks.

    Each rank holds weight of shape [out_features // tp_degree, in_features].
    No communication in forward pass — output is a local shard.

    Used for: Q, K, V projections (split by heads), FFN fc1/up projection.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 tp_degree: int = 4):
        super().__init__()
        assert out_features % tp_degree == 0, (
            f"out_features {out_features} must be divisible by tp_degree {tp_degree}")

        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // tp_degree
        self.tp_degree = tp_degree

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features} '
                f'(local={self.out_features_per_rank}), '
                f'bias={self.bias is not None}, tp={self.tp_degree}')


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across TP ranks.

    Each rank holds weight of shape [out_features, in_features // tp_degree].
    Forward pass performs matmul then all-reduce to get the full output.

    Used for: O projections, FFN fc2/down projection.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 tp_degree: int = 4):
        super().__init__()
        assert in_features % tp_degree == 0, (
            f"in_features {in_features} must be divisible by tp_degree {tp_degree}")

        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_rank = in_features // tp_degree
        self.tp_degree = tp_degree

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_rank))
        if bias:
            # Only rank 0 adds bias to avoid double-counting after all-reduce
            # Actually, we add bias on all ranks and scale — simpler: only add
            # bias after all-reduce. Store full bias but only apply on one rank.
            # Simplest correct approach: store bias, add after all-reduce.
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local matmul (no bias yet)
        out = nn.functional.linear(x, self.weight, None)
        # All-reduce across TP ranks
        out = all_reduce_sum(out)
        # Add bias after all-reduce (only one copy of bias needed)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return (f'in_features={self.in_features} '
                f'(local={self.in_features_per_rank}), '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, tp={self.tp_degree}')


# ---------------------------------------------------------------------------
# Weight sharding utility
# ---------------------------------------------------------------------------

def shard_linear_column(linear: nn.Linear, tp_rank: int, tp_degree: int
                        ) -> ColumnParallelLinear:
    """Convert nn.Linear to ColumnParallelLinear by slicing weights.

    Splits output dimension: each rank gets rows [rank*chunk : (rank+1)*chunk].

    Args:
        linear: Original full linear layer.
        tp_rank: This rank's index (0 to tp_degree-1).
        tp_degree: Total number of TP ranks.

    Returns:
        ColumnParallelLinear with sharded weights.
    """
    out_features = linear.out_features
    in_features = linear.in_features
    chunk_size = out_features // tp_degree

    col_linear = ColumnParallelLinear(
        in_features, out_features,
        bias=(linear.bias is not None),
        tp_degree=tp_degree)

    # Shard weight: [out_features, in_features] → [chunk_size, in_features]
    start = tp_rank * chunk_size
    end = start + chunk_size
    col_linear.weight = nn.Parameter(
        linear.weight.data[start:end].contiguous())

    if linear.bias is not None:
        col_linear.bias = nn.Parameter(
            linear.bias.data[start:end].contiguous())

    return col_linear


def shard_linear_row(linear: nn.Linear, tp_rank: int, tp_degree: int
                     ) -> RowParallelLinear:
    """Convert nn.Linear to RowParallelLinear by slicing weights.

    Splits input dimension: each rank gets columns [rank*chunk : (rank+1)*chunk].

    Args:
        linear: Original full linear layer.
        tp_rank: This rank's index (0 to tp_degree-1).
        tp_degree: Total number of TP ranks.

    Returns:
        RowParallelLinear with sharded weights.
    """
    out_features = linear.out_features
    in_features = linear.in_features
    chunk_size = in_features // tp_degree

    row_linear = RowParallelLinear(
        in_features, out_features,
        bias=(linear.bias is not None),
        tp_degree=tp_degree)

    # Shard weight: [out_features, in_features] → [out_features, chunk_size]
    start = tp_rank * chunk_size
    end = start + chunk_size
    row_linear.weight = nn.Parameter(
        linear.weight.data[:, start:end].contiguous())

    if linear.bias is not None:
        # Bias is full-sized, applied after all-reduce
        row_linear.bias = nn.Parameter(linear.bias.data.clone())

    return row_linear


def shard_qkv_norm(norm: nn.Module, tp_rank: int, tp_degree: int) -> nn.Module:
    """Shard RMSNorm weight for Q/K norms using TP-aware global RMS.

    QK norms in Wan have weight shape [dim]. After column-parallel split of Q/K,
    each rank holds [dim // tp_degree] features. The RMS must still be computed
    over the FULL dim features (requiring all-reduce of sum-of-squares) to match
    the non-TP reference model's normalization behavior.

    Uses TPRMSNorm which all-reduces sum(x²) before computing rsqrt.
    """
    if not hasattr(norm, 'weight'):
        return norm

    global_dim = norm.weight.shape[0]
    local_dim = global_dim // tp_degree
    start = tp_rank * local_dim
    end = start + local_dim

    # Create TP-aware norm that computes global RMS via all-reduce
    new_norm = TPRMSNorm(local_dim, global_dim, eps=norm.eps)
    new_norm.weight = nn.Parameter(norm.weight.data[start:end].contiguous())
    return new_norm


def shard_model_tp(model, tp_rank: int, tp_degree: int):
    """Apply tensor parallelism sharding to a CausalWanModel in-place.

    Shards:
      - Self-attention Q/K/V → column-parallel (split heads)
      - Self-attention O → row-parallel
      - Self-attention QK norms → sharded to match local head count
      - Cross-attention Q/K/V → column-parallel (split heads)
      - Cross-attention O → row-parallel
      - Cross-attention QK norms → sharded to match local head count
      - FFN fc1 → column-parallel
      - FFN fc2 → row-parallel

    Args:
        model: CausalWanModel instance with full (unsharded) weights loaded.
        tp_rank: This rank's local TP index (0 to tp_degree-1).
        tp_degree: Number of TP ranks.
    """
    num_heads = model.num_heads
    assert num_heads % tp_degree == 0, (
        f"num_heads {num_heads} must be divisible by tp_degree {tp_degree}")
    heads_per_rank = num_heads // tp_degree

    for block_idx, block in enumerate(model.blocks):
        # --- Self-Attention ---
        self_attn = block.self_attn

        # Q, K, V: column-parallel (split output dim = split heads)
        self_attn.q = shard_linear_column(self_attn.q, tp_rank, tp_degree)
        self_attn.k = shard_linear_column(self_attn.k, tp_rank, tp_degree)
        self_attn.v = shard_linear_column(self_attn.v, tp_rank, tp_degree)

        # O: row-parallel (split input dim = each rank has local heads)
        self_attn.o = shard_linear_row(self_attn.o, tp_rank, tp_degree)

        # QK norms: shard to match local head count
        self_attn.norm_q = shard_qkv_norm(self_attn.norm_q, tp_rank, tp_degree)
        self_attn.norm_k = shard_qkv_norm(self_attn.norm_k, tp_rank, tp_degree)

        # Update num_heads to local count
        self_attn.num_heads = heads_per_rank

        # --- Cross-Attention ---
        cross_attn = block.cross_attn

        cross_attn.q = shard_linear_column(cross_attn.q, tp_rank, tp_degree)
        cross_attn.k = shard_linear_column(cross_attn.k, tp_rank, tp_degree)
        cross_attn.v = shard_linear_column(cross_attn.v, tp_rank, tp_degree)
        cross_attn.o = shard_linear_row(cross_attn.o, tp_rank, tp_degree)

        cross_attn.norm_q = shard_qkv_norm(cross_attn.norm_q, tp_rank, tp_degree)
        cross_attn.norm_k = shard_qkv_norm(cross_attn.norm_k, tp_rank, tp_degree)

        cross_attn.num_heads = heads_per_rank

        # --- FFN ---
        # FFN is nn.Sequential(Linear(dim, ffn_dim), GELU(), Linear(ffn_dim, dim))
        # or WanFFN with .fc1 and .fc2
        ffn = block.ffn
        if hasattr(ffn, 'fc1'):
            # WanFFN class
            ffn.fc1 = shard_linear_column(ffn.fc1, tp_rank, tp_degree)
            ffn.fc2 = shard_linear_row(ffn.fc2, tp_rank, tp_degree)
        elif isinstance(ffn, nn.Sequential):
            # nn.Sequential(Linear, GELU, Linear)
            ffn[0] = shard_linear_column(ffn[0], tp_rank, tp_degree)
            ffn[2] = shard_linear_row(ffn[2], tp_rank, tp_degree)
        else:
            raise ValueError(f"Unknown FFN type: {type(ffn)}")

    # Update model-level num_heads for KV cache sizing
    model.num_heads_per_rank = heads_per_rank
    model.tp_degree = tp_degree
    model.tp_rank = tp_rank

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TP] Sharded model on rank {tp_rank}/{tp_degree}: "
          f"{heads_per_rank} heads/rank, "
          f"{total_params / 1e9:.2f}B params (local)")

    return model
