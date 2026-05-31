"""T5 Tensor Parallelism — shard T5 encoder across all ranks.

T5-XXL: dim=4096, num_heads=64, dim_ffn=10240, 24 layers
With TP=8: 8 heads/rank, dim_attn/rank=512, dim_ffn/rank=1280

Shards Q/K/V/O in attention and fc layers in FFN using the same
ColumnParallel/RowParallel pattern as DiT.
"""
import torch
import torch.nn as nn
import torch.distributed as dist

from models.tp_utils import (
    ColumnParallelLinear,
    RowParallelLinear,
    shard_linear_column,
    shard_linear_row,
)


def shard_t5_encoder(encoder, tp_rank: int, tp_degree: int):
    """Shard T5Encoder across TP ranks (in-place).

    Shards attention Q/K/V (ColumnParallel) and O (RowParallel),
    and FFN fc (ColumnParallel) and out (RowParallel) in each block.
    """
    for block in encoder.blocks:
        # T5SelfAttention has .attn (T5Attention) and .ffn (T5FeedForward)
        attn = block.attn
        attn.q = shard_linear_column(attn.q, tp_rank, tp_degree)
        attn.k = shard_linear_column(attn.k, tp_rank, tp_degree)
        attn.v = shard_linear_column(attn.v, tp_rank, tp_degree)
        attn.o = shard_linear_row(attn.o, tp_rank, tp_degree)
        attn.num_heads = attn.num_heads // tp_degree
        attn.dim_attn = attn.dim_attn // tp_degree

        ffn = block.ffn
        # T5FeedForward: gate[0] (dim→dim_ffn), fc1 (dim→dim_ffn), fc2 (dim_ffn→dim)
        ffn.gate[0] = shard_linear_column(ffn.gate[0], tp_rank, tp_degree)
        ffn.fc1 = shard_linear_column(ffn.fc1, tp_rank, tp_degree)
        ffn.fc2 = shard_linear_row(ffn.fc2, tp_rank, tp_degree)

    return encoder
