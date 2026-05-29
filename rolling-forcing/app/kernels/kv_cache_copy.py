"""NKI KV cache copy kernel using nki_op (supports in-place mutation).

Replaces tensor.copy_() with NKI DMA for KV cache operations.
Uses nki_op with mutates_args to write directly to destination tensors.
"""
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx import nki_op


@nki.jit
def _cache_copy_kernel(dst, src):
    seqlen = src.shape[0]
    payload_per_row = src.shape[1] * src.shape[2]
    tile_rows = 128 if payload_per_row > 1024 else 1024

    num_tiles = (seqlen + tile_rows - 1) // tile_rows
    for tile_i in range(num_tiles):
        tile_start = tile_i * tile_rows
        current_size = min(tile_rows, seqlen - tile_start)
        nisa.dma_copy(
            dst=dst[nl.ds(tile_start, current_size), :, :],
            src=src[nl.ds(tile_start, current_size), :, :],
        )
    return dst


@nki.jit
def _kv_cache_copy_kernel(k_dst, k_src, v_dst, v_src):
    seqlen = k_src.shape[0]
    payload_per_row = k_src.shape[1] * k_src.shape[2]
    tile_rows = 128 if payload_per_row > 1024 else 1024

    num_tiles = (seqlen + tile_rows - 1) // tile_rows
    for tile_i in range(num_tiles):
        tile_start = tile_i * tile_rows
        current_size = min(tile_rows, seqlen - tile_start)
        nisa.dma_copy(
            dst=k_dst[nl.ds(tile_start, current_size), :, :],
            src=k_src[nl.ds(tile_start, current_size), :, :],
        )
        nisa.dma_copy(
            dst=v_dst[nl.ds(tile_start, current_size), :, :],
            src=v_src[nl.ds(tile_start, current_size), :, :],
        )
    return k_dst, v_dst


@nki_op("rf::cache_copy", mutates_args={"dst"})
def cache_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    _cache_copy_kernel(dst, src)


@nki_op("rf::kv_cache_copy", mutates_args={"k_dst", "v_dst"})
def kv_cache_copy(
    k_dst: torch.Tensor,
    k_src: torch.Tensor,
    v_dst: torch.Tensor,
    v_src: torch.Tensor,
) -> None:
    _kv_cache_copy_kernel(k_dst, k_src, v_dst, v_src)
