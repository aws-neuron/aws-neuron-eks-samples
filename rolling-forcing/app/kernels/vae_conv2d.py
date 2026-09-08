"""NKI kernels for spatial Conv2d — core building blocks for VAE on Neuron.

Two kernels:
  1. vae_conv2d_k1: pointwise 1×1 convolution (weight matmul + bias)
  2. vae_conv2d_k3_shifted: 3×3 convolution via 9 shifted matmuls

nc_matmul semantics: nc_matmul(stationary, moving) = stationary.T @ moving
  So for output = W @ input, we need stationary = W.T (i.e. weight_T).

Input layout: (C, H*W) flattened spatial — caller handles the 5D→2D reshape.
Weight layout: TRANSPOSED — weight_T is (C_in, C_out) so nc_matmul gives W @ input.

NKI API: uses only nisa.dma_copy and nisa.nc_matmul from ISA;
all other ops use nl.* (nl.zeros, nl.copy, nl.add, etc.) — new SDK compatible.
"""
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def vae_conv2d_k1(input_2d, weight_T, bias, HW):
    """Pointwise Conv2d (kernel_size=1): weight @ input + bias.

    nc_matmul(stationary, moving) = stationary.T @ moving
    We pass weight_T = W.T, so nc_matmul(weight_T_chunk, input) = W_chunk @ input  ✓

    Args:
        input_2d: (C_in, HW_padded) bf16 — input flattened spatially, padded to multiple of 512
        weight_T: (C_in, C_out) bf16 — TRANSPOSED weight: weight_T[c_in, c_out] = W[c_out, c_in]
        bias:     (C_out, 1) bf16 — bias reshaped to (C_out, 1) for broadcasting
        HW:       int — actual H*W (before padding)

    Returns:
        output: (C_out, HW_padded) bf16
    """
    C_in = input_2d.shape[0]
    HW_padded = input_2d.shape[1]
    C_out = weight_T.shape[1]
    P = nl.tile_size.pmax  # 128

    SPATIAL_TILE = 512
    num_co_tiles = C_out // P  # C_out must be multiple of 128
    num_ci_tiles = C_in // P   # C_in must be multiple of 128
    num_sp_tiles = HW_padded // SPATIAL_TILE

    output = nl.ndarray((C_out, HW_padded), dtype=input_2d.dtype, buffer=nl.shared_hbm)

    for co_t in nl.sequential_range(num_co_tiles):
        co = co_t * P

        # Load bias tile: (P, 1)
        bias_sbuf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        b_load = nl.ndarray((P, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_load, src=bias[nl.ds(co, P), 0:1])
        bias_sbuf[:, 0:1] = nl.copy(b_load, dtype=nl.float32)

        for sp_t in nl.sequential_range(num_sp_tiles):
            sp = sp_t * SPATIAL_TILE

            # Accumulator in float32
            acc = nl.zeros((P, SPATIAL_TILE), dtype=nl.float32)

            for ci_t in range(num_ci_tiles):
                ci = ci_t * P

                # Load weight_T chunk: (P_ci, P_co) from weight_T[ci:ci+P, co:co+P]
                w_chunk = nl.ndarray((P, P), dtype=weight_T.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=w_chunk, src=weight_T[nl.ds(ci, P), nl.ds(co, P)])

                # Load input chunk: (P_ci, SPATIAL_TILE)
                inp_chunk = nl.ndarray((P, SPATIAL_TILE), dtype=input_2d.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=inp_chunk, src=input_2d[nl.ds(ci, P), nl.ds(sp, SPATIAL_TILE)])

                # nc_matmul: w_chunk.T @ inp_chunk = W[co:co+P, ci:ci+P] @ input[ci:ci+P, sp:sp+512]
                mm_psum = nl.ndarray((P, SPATIAL_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(mm_psum, w_chunk, inp_chunk)
                mm_sbuf = nl.copy(mm_psum, dtype=nl.float32)
                acc = nl.add(acc, mm_sbuf)

            # Add bias (broadcast along spatial dim)
            acc = nl.add(acc, bias_sbuf)

            # Store
            out_tile = nl.copy(acc, dtype=input_2d.dtype)
            nisa.dma_copy(dst=output[nl.ds(co, P), nl.ds(sp, SPATIAL_TILE)], src=out_tile)

    return output


@nki.jit
def vae_conv2d_k3_shifted(shifted_inputs, weight_slices_T, bias, num_positions):
    """3×3 Conv2d using 9 pre-shifted input tensors (prepared by caller).

    nc_matmul(stationary, moving) = stationary.T @ moving
    We pass weight_slices_T so nc_matmul gives W_slice @ shifted_input  ✓

    Args:
        shifted_inputs:  (9 * C_in, HW_out_padded) bf16 — 9 shifts stacked along channel dim
        weight_slices_T: (C_in * 9, C_out) bf16 — TRANSPOSED weight slices in blocked layout
                         row = k_idx * C_in + c_in, col = c_out
        bias:            (C_out, 1) bf16
        num_positions:   int — actual H_out * W_out

    Returns:
        output: (C_out, HW_out_padded) bf16
    """
    total_cin = shifted_inputs.shape[0]  # 9 * C_in
    HW_padded = shifted_inputs.shape[1]
    C_out = weight_slices_T.shape[1]
    C_in = total_cin // 9
    P = nl.tile_size.pmax  # 128

    SPATIAL_TILE = 512
    num_co_tiles = C_out // P
    num_ci_tiles = C_in // P
    num_sp_tiles = HW_padded // SPATIAL_TILE

    output = nl.ndarray((C_out, HW_padded), dtype=shifted_inputs.dtype, buffer=nl.shared_hbm)

    for co_t in nl.sequential_range(num_co_tiles):
        co = co_t * P

        # Load bias
        bias_sbuf = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        b_load = nl.ndarray((P, 1), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_load, src=bias[nl.ds(co, P), 0:1])
        bias_sbuf[:, 0:1] = nl.copy(b_load, dtype=nl.float32)

        for sp_t in nl.sequential_range(num_sp_tiles):
            sp = sp_t * SPATIAL_TILE

            acc = nl.zeros((P, SPATIAL_TILE), dtype=nl.float32)

            # 9 kernel positions × C_in/P contraction tiles
            for k_idx in range(9):
                for ci_t in range(num_ci_tiles):
                    ci = ci_t * P

                    # Weight_T for this (k_idx, ci_tile): (P_ci, P_co)
                    w_row = k_idx * C_in + ci
                    w_chunk = nl.ndarray((P, P), dtype=weight_slices_T.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=w_chunk,
                                  src=weight_slices_T[nl.ds(w_row, P), nl.ds(co, P)])

                    # Input for this shift and channel tile
                    inp_row = k_idx * C_in + ci
                    inp_chunk = nl.ndarray((P, SPATIAL_TILE), dtype=shifted_inputs.dtype,
                                           buffer=nl.sbuf)
                    nisa.dma_copy(dst=inp_chunk,
                                  src=shifted_inputs[nl.ds(inp_row, P), nl.ds(sp, SPATIAL_TILE)])

                    # nc_matmul: w_chunk.T @ inp_chunk = W_slice[co:co+P, ci:ci+P] @ shifted_input
                    mm_psum = nl.ndarray((P, SPATIAL_TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(mm_psum, w_chunk, inp_chunk)
                    mm_sbuf = nl.copy(mm_psum, dtype=nl.float32)
                    acc = nl.add(acc, mm_sbuf)

            # Add bias
            acc = nl.add(acc, bias_sbuf)

            out_tile = nl.copy(acc, dtype=shifted_inputs.dtype)
            nisa.dma_copy(dst=output[nl.ds(co, P), nl.ds(sp, SPATIAL_TILE)], src=out_tile)

    return output
