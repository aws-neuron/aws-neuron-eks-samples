import torch
import pytest

from models.layers import causal_rope_apply, rope_params
from kernels.rope import causal_rope_rotation, build_rope_grids


def _make_freqs(head_dim):
    """Precompute RoPE frequencies matching CausalWanModel.__init__."""
    d = head_dim
    cos_f, sin_f = rope_params(1024, d - 4 * (d // 6))
    cos_h, sin_h = rope_params(1024, 2 * (d // 6))
    cos_w, sin_w = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_f, cos_h, cos_w], dim=1), torch.cat([sin_f, sin_h, sin_w], dim=1)


def _build_expanded_grids(grid_sizes, freqs_cos, freqs_sin, start_frame_t, head_dim):
    """Build cos_expanded and sin_signed for the NKI RoPE kernel.

    Takes the same raw freqs as causal_rope_apply and produces the
    rotate_half-style expanded grids:
        cos_expanded[seq, 2j]   = cos_expanded[seq, 2j+1] = cos[seq, j]
        sin_signed[seq, 2j]     = -sin[seq, j]
        sin_signed[seq, 2j+1]   =  sin[seq, j]
    """
    c = head_dim // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    f, h, w = grid_sizes
    seq_len = f * h * w

    frame_idx = start_frame_t + torch.arange(f)

    cos = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, -1)   # [seq_len, c]

    sin = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(seq_len, -1)   # [seq_len, c]

    # Expand: repeat each value for the interleaved pair
    cos_expanded = cos.repeat_interleave(2, dim=-1)        # [seq_len, D]

    # Signed sin for rotate_half: [-sin, sin, -sin, sin, ...]
    sin_expanded = sin.repeat_interleave(2, dim=-1)        # [seq_len, D]
    sign = torch.ones(head_dim)
    sign[0::2] = -1.0
    sin_signed = sin_expanded * sign                       # [seq_len, D]

    return cos_expanded, sin_signed


def _rotate_half_ref(x, cos_expanded, sin_signed, num_heads, head_dim):
    """CPU reference for causal_rope_rotation only (rotate_half formula).

    Applies: out[n] = x[n] * cos + swap_pairs(x[n]) * sin  per head.

    Args:
        x: [seq_len, num_heads, head_dim] bfloat16
        cos_expanded: [seq_len, head_dim] float32
        sin_signed:   [seq_len, head_dim] float32

    Returns: [seq_len, num_heads, head_dim] float32
    """
    x_f32 = x.float()
    parts = []
    for n in range(num_heads):
        xh = x_f32[:, n, :]                        # [seq_len, D]
        # swap adjacent pairs: [x1, x0, x3, x2, ...]
        xh_swap = xh.clone()
        xh_swap[:, 0::2] = xh[:, 1::2]
        xh_swap[:, 1::2] = xh[:, 0::2]
        parts.append(xh * cos_expanded + xh_swap * sin_signed)
    return torch.stack(parts, dim=1).to(x.dtype)


@pytest.mark.parametrize("grid_sizes,start_frame", [
    ((15, 30, 52), 0),   # full block, first window
    ((15, 30, 52), 3),   # full block, later window
    ((3, 30, 52), 0),    # anchor block, updating_cache
    ((3, 30, 52), 7),    # anchor block, normal denoising
])
def test_causal_rope_rotation(grid_sizes, start_frame):
    """Test causal_rope_rotation NKI kernel against rotate_half CPU reference."""
    dtype = torch.bfloat16
    num_heads, head_dim = 12, 128
    f, h, w = grid_sizes
    seq_len = f * h * w

    x = torch.randn(seq_len, num_heads, head_dim, dtype=dtype)

    # Build grids on CPU (these are just inputs to the kernel under test)
    freqs_cos, freqs_sin = _make_freqs(head_dim)
    start_frame_t = torch.tensor(start_frame)
    cos_expanded, sin_signed = _build_expanded_grids(
        grid_sizes, freqs_cos, freqs_sin, start_frame_t, head_dim
    )
    # ── CPU reference (rotate_half only) ──
    expected = _rotate_half_ref(x, cos_expanded, sin_signed, num_heads, head_dim)

    # ── Pack into combined [seq_len, 2*D] tensor ──
    cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).to(torch.float32)

    # ── Move to Neuron ──
    x_n = x.to("neuron")
    cos_sin_n = cos_sin.to("neuron")

    # ── Call NKI kernel ──
    result = causal_rope_rotation(
        x_n, cos_sin_n,
        num_heads=num_heads, head_dim=head_dim,
    )

    # ── Compare ──
    result_cpu = result.cpu()
    assert torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-3), \
        f"max diff: {(result_cpu - expected).abs().max().item()}"


@pytest.mark.parametrize("grid_sizes,start_frame", [
    ((15, 30, 52), 0),   # full block, first window
    ((15, 30, 52), 3),   # full block, later window
    ((3, 30, 52), 0),    # anchor block, updating_cache
    ((3, 30, 52), 7),    # anchor block, normal denoising
])
def test_causal_rope_e2e(grid_sizes, start_frame):
    """End-to-end test: build_rope_grids + causal_rope_rotation on Neuron
    vs causal_rope_apply on CPU."""
    dtype = torch.bfloat16
    B, num_heads, head_dim = 1, 12, 128
    f, h, w = grid_sizes
    seq_len = f * h * w

    x = torch.randn(B, seq_len, num_heads, head_dim, dtype=dtype)

    freqs_cos, freqs_sin = _make_freqs(head_dim)
    start_frame_t = torch.tensor(start_frame)

    # ── CPU reference (causal_rope_apply) ──
    expected = causal_rope_apply(
        x, grid_sizes, freqs_cos, freqs_sin, start_frame=start_frame_t
    )

    # ── Build helper inputs ──
    sign_pat = _build_sign_pattern(head_dim)

    # ── Move to Neuron ──
    freqs_cos_n = freqs_cos.to("neuron")
    freqs_sin_n = freqs_sin.to("neuron")
    sign_n = sign_pat.to("neuron")
    sf_n = torch.tensor([[start_frame]], dtype=torch.int32).to("neuron")
    x_n = x[0].to("neuron")  # [seq_len, num_heads, head_dim]

    # ── Call NKI kernels sequentially ──
    # Step 1: build grids -> [F*H, W*2*D], reshape to [seq_len, 2*D]
    combined = build_rope_grids(
        freqs_cos_n, freqs_sin_n, sign_n, sf_n,
        F=f, H=h, W=w, head_dim=head_dim,
    ).view(seq_len, 2 * head_dim)

    # Step 2: apply rotation (combined passed directly, no split)
    result = causal_rope_rotation(
        x_n, combined,
        num_heads=num_heads, head_dim=head_dim,
    )

    # ── Compare ──
    result_cpu = result.cpu().reshape(1, seq_len, num_heads, head_dim)
    assert torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-3), \
        f"max diff: {(result_cpu - expected).abs().max().item()}"


def _build_sign_pattern(head_dim):
    """Build [128, head_dim] float32 sign pattern.

    sign[:, 2j] = -1.0, sign[:, 2j+1] = 1.0
    """
    sign = torch.ones(head_dim, dtype=torch.float32)
    sign[0::2] = -1.0
    return sign.unsqueeze(0).expand(128, -1).contiguous()


@pytest.mark.parametrize("grid_sizes,start_frame", [
    ((15, 30, 52), 0),   # full block, first window
    ((15, 30, 52), 3),   # full block, later window
    ((3, 30, 52), 0),    # anchor block, updating_cache
    ((3, 30, 52), 7),    # anchor block, normal denoising
])
def test_build_rope_grids_kernel(grid_sizes, start_frame):
    """Test NKI build_rope_grids kernel against CPU reference."""
    head_dim = 128
    f, h, w = grid_sizes
    seq_len = f * h * w
    D = head_dim

    freqs_cos, freqs_sin = _make_freqs(head_dim)
    start_frame_t = torch.tensor(start_frame)

    # ── CPU reference ──
    cos_expected, sin_expected = _build_expanded_grids(
        grid_sizes, freqs_cos, freqs_sin, start_frame_t, head_dim
    )

    # ── Build kernel helper inputs ──
    sign_pat = _build_sign_pattern(head_dim)

    # ── Move to Neuron ──
    freqs_cos_n = freqs_cos.to("neuron")
    freqs_sin_n = freqs_sin.to("neuron")
    sign_n = sign_pat.to("neuron")
    sf_n = torch.tensor([[start_frame]], dtype=torch.int32).to("neuron")

    # ── Call NKI kernel ──
    combined = build_rope_grids(
        freqs_cos_n, freqs_sin_n, sign_n, sf_n,
        F=f, H=h, W=w, head_dim=head_dim,
    )

    # ── Compare ──
    # Output is [F*H, W*2*D], reshape to [seq_len, 2*D] (same physical layout)
    combined_cpu = combined.cpu().reshape(seq_len, 2 * D)
    cos_result = combined_cpu[:, :D]
    sin_result = combined_cpu[:, D:]

    assert torch.allclose(cos_result, cos_expected, rtol=1e-2, atol=1e-3), \
        f"cos max diff: {(cos_result - cos_expected).abs().max().item()}"
    assert torch.allclose(sin_result, sin_expected, rtol=1e-2, atol=1e-3), \
        f"sin max diff: {(sin_result - sin_expected).abs().max().item()}"
