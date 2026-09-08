import torch
import pytest

from models.layers import WanLayerNorm


@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_wan_layer_norm(elementwise_affine):
    """Ref CPU (torch.nn.LayerNorm) vs Model CPU vs Model Neuron."""
    dtype = torch.bfloat16
    B, T, dim, eps = 1, 23400, 1536, 1e-6
    x_cpu = torch.randn(B, T, dim, dtype=dtype)

    # Ref CPU: torch.nn.LayerNorm
    ref_ln = torch.nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine).to(dtype)
    module_cpu = WanLayerNorm(dim, eps, elementwise_affine=elementwise_affine).to(dtype)
    if elementwise_affine:
        module_cpu.weight.data.copy_(ref_ln.weight.data)
        module_cpu.bias.data.copy_(ref_ln.bias.data)
    ref_out = ref_ln(x_cpu)

    # Model CPU
    cpu_out = module_cpu(x_cpu)
    torch.testing.assert_close(cpu_out, ref_out, rtol=1e-2, atol=1e-2)

    # Model Neuron
    module_neuron = module_cpu.to("neuron")
    neuron_out = module_neuron(x_cpu.to("neuron")).cpu()
    torch.testing.assert_close(neuron_out, ref_out, rtol=1e-2, atol=1e-2)
