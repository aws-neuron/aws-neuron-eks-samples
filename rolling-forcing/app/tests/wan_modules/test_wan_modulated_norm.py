import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import modulated_norm_scale, modulated_norm_shift


@pytest.mark.parametrize("num_frames", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_modulated_norm(num_frames):
    """Test modulated norm: unflatten → scale+shift → flatten."""
    dtype = torch.bfloat16
    B, frame_seqlen, dim = 1, 1560, 1536
    L = num_frames * frame_seqlen  # 23400

    norm_x = torch.randn(B, L, dim, dtype=dtype)
    shift = torch.randn(B, num_frames, 1, dim, dtype=dtype)
    scale = torch.randn(B, num_frames, 1, dim, dtype=dtype)
    ones = torch.ones_like(scale)

    # CPU reference
    expected = modulated_norm_shift(
        modulated_norm_scale(norm_x, scale, ones, num_frames, frame_seqlen),
        shift,
    )

    # JIT + neuron
    modulated_norm_scale_neuron = jit(modulated_norm_scale)
    modulated_norm_shift_neuron = jit(modulated_norm_shift)
    norm_x_neuron = norm_x.to("neuron")
    shift_neuron = shift.to("neuron")
    scale_neuron = scale.to("neuron")
    ones_neuron = ones.to("neuron")
    result = modulated_norm_shift_neuron(
        modulated_norm_scale_neuron(
            norm_x_neuron,
            scale_neuron,
            ones_neuron,
            num_frames,
            frame_seqlen,
        ),
        shift_neuron,
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
