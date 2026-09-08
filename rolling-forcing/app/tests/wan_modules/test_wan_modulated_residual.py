import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import modulated_residual


@pytest.mark.parametrize("num_frames", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_modulated_residual(num_frames):
    """Test scaled residual: x + unflatten(y) * scale → flatten."""
    dtype = torch.bfloat16
    B, frame_seqlen, dim = 1, 1560, 1536
    L = num_frames * frame_seqlen  # 23400

    x = torch.randn(B, L, dim, dtype=dtype)
    y = torch.randn(B, L, dim, dtype=dtype)
    scale = torch.randn(B, num_frames, 1, dim, dtype=dtype)

    # CPU reference
    expected = modulated_residual(x, y, scale, num_frames, frame_seqlen)

    # JIT + neuron
    modulated_residual_neuron = jit(modulated_residual)
    result = modulated_residual_neuron(
        x.to("neuron"), y.to("neuron"), scale.to("neuron"),
        num_frames, frame_seqlen)

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
