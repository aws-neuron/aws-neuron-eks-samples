import math

import pytest
import torch
import torch.nn as nn

from models.layers import CausalHead, WanLayerNorm


class RefCausalHead(nn.Module):
    """GPU reference from causal_model_opt.py. Uses .chunk() for modulation."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e[1]) + e[0]))
        return x


@pytest.mark.parametrize("num_frames", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_causal_head(num_frames):
    """CausalHead: [1, L, 2048] + [1, F, 1, 2048] -> [1, F, 1560, 64]

    dim=2048, out_dim=16, patch_size=(1,2,2),
    frame_seqlen=30*52=1560, out_channels=prod(1,2,2)*16=64.
    """
    dtype = torch.bfloat16
    dim = 2048
    out_dim = 16
    patch_size = (1, 2, 2)
    frame_seqlen = 30 * 52  # 1560
    L = num_frames * frame_seqlen

    x = torch.randn(1, L, dim, dtype=dtype)
    e = torch.randn(1, num_frames, 1, dim, dtype=dtype)

    gpu_head = RefCausalHead(dim, out_dim, patch_size).to(dtype)

    causal_head = CausalHead(dim, out_dim, patch_size).to(dtype)
    causal_head.load_state_dict(gpu_head.state_dict())

    expected = gpu_head(x, e)

    # CPU: bitwise identical (slicing vs .chunk() produces same values)
    result_cpu = causal_head(x, e)
    assert torch.equal(result_cpu, expected)

    # Neuron: bf16 tolerance across 3 kernels (norm, modulate, linear)
    causal_head_neuron = causal_head.to("neuron")
    result_neuron = causal_head_neuron(x.to("neuron"), e.to("neuron"))
    # FIXME here we use rtol=2e-2, atol=2e-2
    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=2e-2, atol=2e-2)
