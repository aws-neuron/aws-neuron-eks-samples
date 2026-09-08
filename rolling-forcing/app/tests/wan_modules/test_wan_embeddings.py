import torch
import torch.nn as nn

from torch_neuronx.jit import jit

from models.layers import GELU, SiLU


def test_text_embedding():
    """text_embedding: Linear(4096, 2048) → GELU → Linear(2048, 2048)"""
    dtype = torch.bfloat16
    text_dim, dim = 4096, 2048
    x = torch.randn(1, 512, text_dim, dtype=dtype)

    # CPU reference (nn.GELU)
    ref = nn.Sequential(
        nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
        nn.Linear(dim, dim)).to(dtype)
    expected = ref(x)

    # CPU custom (our GELU)
    custom = nn.Sequential(
        nn.Linear(text_dim, dim), GELU(),
        nn.Linear(dim, dim)).to(dtype)
    custom.load_state_dict(ref.state_dict(), strict=False)
    result_cpu = custom(x)

    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # Neuron
    neuron = jit(custom).to("neuron")
    result_neuron = neuron(x.to("neuron"))

    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_time_embedding():
    """time_embedding: Linear(256, 2048) → SiLU → Linear(2048, 2048)"""
    dtype = torch.bfloat16
    freq_dim, dim = 256, 2048
    x = torch.randn(15, freq_dim, dtype=dtype)

    # CPU reference (nn.SiLU)
    ref = nn.Sequential(
        nn.Linear(freq_dim, dim), nn.SiLU(),
        nn.Linear(dim, dim)).to(dtype)
    expected = ref(x)

    # CPU custom (our SiLU)
    custom = nn.Sequential(
        nn.Linear(freq_dim, dim), SiLU(),
        nn.Linear(dim, dim)).to(dtype)
    custom.load_state_dict(ref.state_dict(), strict=False)
    result_cpu = custom(x)

    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # Neuron
    neuron = jit(custom).to("neuron")
    result_neuron = neuron(x.to("neuron"))

    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_time_projection():
    """time_projection: SiLU → Linear(2048, 12288)"""
    dtype = torch.bfloat16
    dim = 2048
    x = torch.randn(15, dim, dtype=dtype)

    # CPU reference (nn.SiLU)
    ref = nn.Sequential(
        nn.SiLU(), nn.Linear(dim, dim * 6)).to(dtype)
    expected = ref(x)

    # CPU custom (our SiLU)
    custom = nn.Sequential(
        SiLU(), nn.Linear(dim, dim * 6)).to(dtype)
    custom.load_state_dict(ref.state_dict(), strict=False)
    result_cpu = custom(x)

    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # Neuron
    neuron = jit(custom).to("neuron")
    result_neuron = neuron(x.to("neuron"))

    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)
