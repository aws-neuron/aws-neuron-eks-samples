import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import modulation_chunk


@pytest.mark.parametrize("num_frames", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_modulation_chunk(num_frames):
    """Test modulation bias add + chunk into 6 per-frame vectors."""
    dtype = torch.bfloat16
    B, dim = 1, 1536

    modulation = torch.randn(1, 6, dim, dtype=dtype)
    e = torch.randn(B, num_frames, 6, dim, dtype=dtype)

    # CPU reference
    expected = modulation_chunk(modulation, e)

    # JIT + neuron
    modulation_chunk_neuron = jit(modulation_chunk)
    result = modulation_chunk_neuron(
        modulation.to("neuron"), e.to("neuron"))

    assert len(result) == 6
    for i in range(6):
        assert result[i].shape == (B, num_frames, 1, dim)
        torch.testing.assert_close(
            result[i].cpu(), expected[i], rtol=1e-2, atol=1e-2)
