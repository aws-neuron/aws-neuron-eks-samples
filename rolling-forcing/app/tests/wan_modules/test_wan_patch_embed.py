import pytest
import torch
import torch.nn as nn

from models.layers import WanPatchEmbed


@pytest.mark.parametrize("F", [15, 3], ids=["denoise-15f", "cache-update-3f"])
def test_patch_embed(F):
    dtype = torch.bfloat16
    B, C, H, W = 1, 16, 60, 104
    in_channels, out_channels = 16, 2048
    kernel_size = (1, 2, 2)

    x = torch.randn(B, C, F, H, W, dtype=dtype)

    # 1. nn.Conv3d on CPU (ground truth)
    conv3d = nn.Conv3d(in_channels, out_channels,
                       kernel_size=kernel_size, stride=kernel_size).to(dtype)
    expected = conv3d(x)

    # 2. WanPatchEmbed on CPU — load weights from Conv3d
    patch_cpu = WanPatchEmbed(in_channels, out_channels, kernel_size).to(dtype)
    patch_cpu.load_state_dict(conv3d.state_dict())
    result_cpu = patch_cpu(x)

    # Conv3d vs WanPatchEmbed CPU: bitwise identical only when using fp32
    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # 3. WanPatchEmbed on Neuron device
    patch_neuron = patch_cpu.to("neuron")
    x_neuron = x.to("neuron")
    result_neuron = patch_neuron(x_neuron)

    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)
