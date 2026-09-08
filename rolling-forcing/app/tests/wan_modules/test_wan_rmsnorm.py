import torch

from models.layers import WanRMSNorm


def test_wan_rmsnorm():
    """Test WanRMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    dtype = torch.bfloat16
    B, T, dim, eps = 1, 23400, 1536, 1e-5
    x_cpu = torch.randn(B, T, dim, dtype=dtype)

    module_cpu = WanRMSNorm(dim, eps).to(dtype)
    expected = module_cpu(x_cpu)

    module_neuron = module_cpu.to("neuron")
    x_neuron = x_cpu.to("neuron")
    result = module_neuron(x_neuron)

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
