import neuronxcc.nki as nki
import nki.compiler.kernel_builder as nb
from nki.compiler.kernel_builder import Tensor
from nki.compiler.kernel_builder import isa as nisa


@nki.jit
def cache_copy(dst: Tensor, src: Tensor):
    """Copy a single cache tensor via direct HBM-to-HBM DMA.

    Both tensors have shape [seqlen, num_heads, head_size] with the same dtype.
    Loops over the first dimension (seqlen), copying 128 elements per iteration
    along the partition dimension. The last iteration handles any remainder.
    """
    seqlen = src.shape[0]

    TILE_P = 128
    num_tiles = (seqlen + TILE_P - 1) // TILE_P

    for tile_i in range(num_tiles):
        tile_start = tile_i * TILE_P
        current_size = min(TILE_P, seqlen - tile_start)

        nisa.dma_copy(
            dst=dst[nb.ds(tile_start, current_size), :, :],
            src=src[nb.ds(tile_start, current_size), :, :],
            name=f"copy[tile={tile_i}]",
        )


@nki.jit
def kv_cache_copy(k_dst: Tensor, k_src: Tensor, v_dst: Tensor, v_src: Tensor):
    """Copy K and V cache tensors via direct HBM-to-HBM DMA in a single kernel.

    All tensors have shape [seqlen, num_heads, head_size] with the same dtype.
    Loops over the first dimension (seqlen), copying 128 elements per iteration
    along the partition dimension. The last iteration handles any remainder.
    """
    seqlen = k_src.shape[0]

    TILE_P = 128
    num_tiles = (seqlen + TILE_P - 1) // TILE_P

    for tile_i in range(num_tiles):
        tile_start = tile_i * TILE_P
        current_size = min(TILE_P, seqlen - tile_start)

        nisa.dma_copy(
            dst=k_dst[nb.ds(tile_start, current_size), :, :],
            src=k_src[nb.ds(tile_start, current_size), :, :],
            name=f"k_copy[tile={tile_i}]",
        )
        nisa.dma_copy(
            dst=v_dst[nb.ds(tile_start, current_size), :, :],
            src=v_src[nb.ds(tile_start, current_size), :, :],
            name=f"v_copy[tile={tile_i}]",
        )
