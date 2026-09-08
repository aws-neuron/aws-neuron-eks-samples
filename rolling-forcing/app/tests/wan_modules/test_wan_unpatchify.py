import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import unpatchify


@pytest.mark.parametrize("f", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_unpatchify(f):
    """unpatchify: [1, F*H*W, out_C] → [c, f*pT, h*pH, w*pW]"""
    dtype = torch.bfloat16
    out_dim = 16
    patch_size = (1, 2, 2)
    grid_sizes = (f, 30, 52)
    f, h, w = grid_sizes
    out_C = out_dim * patch_size[0] * patch_size[1] * patch_size[2]  # 64

    x = torch.randn(1, f * h * w, out_C, dtype=dtype)

    # CPU reference
    expected = unpatchify(x, out_dim, patch_size, grid_sizes)

    # Neuron
    unpatchify_neuron = jit(unpatchify)
    result_neuron = unpatchify_neuron(x.to("neuron"), out_dim, patch_size, grid_sizes)

    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)
