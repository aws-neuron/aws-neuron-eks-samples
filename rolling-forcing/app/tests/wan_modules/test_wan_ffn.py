import torch
import torch.nn as nn

from models.layers import GELU, WanFFN


def test_wan_ffn_sequential():
    """Test nn.Sequential with a 2-layer MLP (linear + relu + linear)."""
    dtype = torch.bfloat16
    B, T, dim, ffn_dim = 1, 23400, 1536, 8960
    x_cpu = torch.randn(B, T, dim, dtype=dtype)

    module_cpu = nn.Sequential(
        nn.Linear(dim, ffn_dim),
        GELU(),
        nn.Linear(ffn_dim, dim),
    ).to(dtype)
    expected = module_cpu(x_cpu)

    from torch_neuronx.jit import jit
    module_neuron = jit(module_cpu).to("neuron")

    x_neuron = x_cpu.to("neuron")
    result = module_neuron(x_neuron)

    torch.testing.assert_close(result.cpu(), expected, rtol=5e-3, atol=5e-3)


def test_wan_ffn():
    """Test a 2-layer FFN (linear + GELU + linear)."""
    dtype = torch.bfloat16
    B, T, dim, ffn_dim = 1, 23400, 1536, 8960
    x_cpu = torch.randn(B, T, dim, dtype=dtype)

    module_cpu = WanFFN(dim, ffn_dim).to(dtype)
    expected = module_cpu(x_cpu)

    module_neuron = module_cpu.to("neuron")
    x_neuron = x_cpu.to("neuron")
    result = module_neuron(x_neuron)

    torch.testing.assert_close(result.cpu(), expected, rtol=5e-3, atol=5e-3)
