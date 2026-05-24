"""KV cache copy NKI kernels using bundled neuronxcc.nki API.

Simple HBM-to-HBM DMA copy for KV cache tensors.
Requires seqlen to be a multiple of 128 (NKI tile size).
Caller should fall back to tensor.copy_() for non-aligned sizes.

Adapted from kernel_builder API to standard neuronxcc.nki:
- nb.ndarray → nl.ndarray
- nb.ds → nl.ds
- nisa.dma_copy(dst=, src=) keyword-only
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def cache_copy(dst, src):
    """Copy a single cache tensor via DMA.

    Both tensors have shape [seqlen, num_heads, head_size].
    seqlen must be a multiple of 128.
    """
    seqlen = src.shape[0]
    P = nl.tile_size.pmax  # 128

    assert seqlen % P == 0, f"seqlen ({seqlen}) must be a multiple of {P}"
    num_tiles = seqlen // P

    for tile_i in range(num_tiles):
        tile_start = tile_i * P
        nisa.dma_copy(
            dst=dst[nl.ds(tile_start, P), :, :],
            src=src[nl.ds(tile_start, P), :, :],
        )


@nki.jit
def kv_cache_copy(k_dst, k_src, v_dst, v_src):
    """Copy K and V cache tensors via DMA in a single kernel.

    All tensors have shape [seqlen, num_heads, head_size].
    seqlen must be a multiple of 128.
    """
    seqlen = k_src.shape[0]
    P = nl.tile_size.pmax  # 128

    assert seqlen % P == 0, f"seqlen ({seqlen}) must be a multiple of {P}"
    num_tiles = seqlen // P

    for tile_i in range(num_tiles):
        tile_start = tile_i * P
        nisa.dma_copy(
            dst=k_dst[nl.ds(tile_start, P), :, :],
            src=k_src[nl.ds(tile_start, P), :, :],
        )
        nisa.dma_copy(
            dst=v_dst[nl.ds(tile_start, P), :, :],
            src=v_src[nl.ds(tile_start, P), :, :],
        )
