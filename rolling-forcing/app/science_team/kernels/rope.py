"""RoPE kernel — uses our proven causal_rope_rotation kernel.

build_rope_grids is implemented in PyTorch (the alpha NKI version requires
reshape_dim/expand_dim/broadcast APIs not available in our SDK).
"""
import math
import torch
import nki
import nki.language as nl
import nki.isa as nisa
from torch_neuronx.nki_hop import wrap_nki


_P = 128


@wrap_nki
@nki.jit
def causal_rope_rotation(x, cos_sin, head_start=0, head_end=12, head_dim=128):
    """Apply RoPE rotation to x using pre-built cos_sin grids.

    Args:
        x:       [seq_len, num_heads, head_dim] bfloat16 (must be padded to 128)
        cos_sin: [seq_len, 2*head_dim] float32
        head_start: first head index to process
        head_end:   last head index (exclusive)
        head_dim:   dimension per head
    Returns:
        out: [seq_len, num_heads, head_dim] bfloat16
    """
    seq_len = x.shape[0]
    N = head_end - head_start
    D = head_dim
    P = nl.tile_size.pmax

    assert seq_len % P == 0
    num_tiles = seq_len // P

    out = nl.ndarray((seq_len, N, D), dtype=x.dtype, buffer=nl.shared_hbm)

    for tile_i in nl.sequential_range(num_tiles):
        ts = tile_i * P
        cs_sb = nl.load(cos_sin[nl.ds(ts, P), :])
        cos_tile = cs_sb[:, nl.ds(0, D)]
        sin_tile = cs_sb[:, nl.ds(D, D)]
        x_sb = nl.load(x[nl.ds(ts, P), :, :])

        out_sb = nl.ndarray((P, N, D), dtype=x.dtype, buffer=nl.sbuf)
        for n in nl.affine_range(N):
            xh = x_sb[:, n, :]
            x_cos = nl.multiply(xh, cos_tile)

            x_swap = nl.ndarray((P, D), dtype=xh.dtype, buffer=nl.sbuf)
            x_swap[:, 0::2] = xh[:, 1::2]
            x_swap[:, 1::2] = xh[:, 0::2]

            x_sin = nl.multiply(x_swap, sin_tile)
            out_sb[:, n, :] = nl.add(x_cos, x_sin)

        nl.store(out[nl.ds(ts, P), :, :], out_sb)

    return out


def build_rope_grids(freqs_cos, freqs_sin, sign_pattern, start_frame,
                     F=15, H=44, W=78, head_dim=128):
    """Build 3D RoPE cos/sin grids in PyTorch.

    This replaces the alpha NKI kernel that uses reshape_dim/expand_dim/broadcast.
    Builds the same [seq_len_padded, 2*head_dim] output as the NKI version.

    Args:
        freqs_cos: [max_seq, head_dim//2] float32
        freqs_sin: [max_seq, head_dim//2] float32
        sign_pattern: [128, head_dim] float32 (sign pattern for sin)
        start_frame: tensor [1, 1] int32
        F, H, W: grid dimensions
        head_dim: head dimension (128)
    Returns:
        cos_sin: [seq_len_padded, 2*head_dim] float32
    """
    d = head_dim
    c = d // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    seq_len = F * H * W
    device = freqs_cos.device

    frame_idx = start_frame.flatten() + torch.arange(F, device=device)

    cos_half = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_cos[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_cos[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1)
    ], dim=-1).reshape(seq_len, c)

    sin_half = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(F, 1, 1, -1).expand(F, H, W, -1),
        freqs_sin[:H, s0:s0 + s1].view(1, H, 1, -1).expand(F, H, W, -1),
        freqs_sin[:W, s0 + s1:].view(1, 1, W, -1).expand(F, H, W, -1)
    ], dim=-1).reshape(seq_len, c)

    cos_expanded = cos_half.repeat_interleave(2, dim=-1)
    sin_expanded = sin_half.repeat_interleave(2, dim=-1)
    sign = torch.ones(d, device=device, dtype=sin_expanded.dtype)
    sign[0::2] = -1.0
    sin_signed = sin_expanded * sign.unsqueeze(0)

    cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()
    return cos_sin
