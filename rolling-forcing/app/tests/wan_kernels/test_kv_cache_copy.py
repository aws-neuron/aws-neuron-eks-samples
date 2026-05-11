import torch
import pytest

from kernels.kv_cache_copy import cache_copy, kv_cache_copy


@pytest.mark.parametrize(
    "seqlen,num_heads,head_size",
    [
        (4680, 12, 128),
        (23400, 12, 128),
        (28080, 12, 128),
        (32760, 12, 128),
    ],
)
def test_kv_cache_copy(seqlen, num_heads, head_size):
    dtype = torch.bfloat16
    shape = (seqlen, num_heads, head_size)

    # Generate random source data for K and V
    k_src = torch.randn(shape, dtype=dtype)
    v_src = torch.randn(shape, dtype=dtype)
    k_dst = torch.zeros(shape, dtype=dtype)
    v_dst = torch.zeros(shape, dtype=dtype)

    # Move to Neuron
    k_src_neuron = k_src.to("neuron")
    v_src_neuron = v_src.to("neuron")
    k_dst_neuron = k_dst.to("neuron")
    v_dst_neuron = v_dst.to("neuron")

    # Call the @jit-decorated kernel directly (compilation is automatic)
    kv_cache_copy(k_dst_neuron, k_src_neuron, v_dst_neuron, v_src_neuron)

    # Verify exact match (pure copy, no tolerance)
    assert torch.allclose(
        k_dst_neuron.cpu(), k_src, rtol=0, atol=0,
    ), f"K copy mismatch for shape {shape}"
    assert torch.allclose(
        v_dst_neuron.cpu(), v_src, rtol=0, atol=0,
    ), f"V copy mismatch for shape {shape}"


@pytest.mark.parametrize(
    "seqlen,num_heads,head_size",
    [
        (4680, 12, 128),
        (23400, 12, 128),
        (28080, 12, 128),
        (32760, 12, 128),
    ],
)
def test_cache_copy(seqlen, num_heads, head_size):
    dtype = torch.bfloat16
    shape = (seqlen, num_heads, head_size)

    src = torch.randn(shape, dtype=dtype)
    dst = torch.zeros(shape, dtype=dtype)

    src_neuron = src.to("neuron")
    dst_neuron = dst.to("neuron")

    # Call the @jit-decorated kernel directly (compilation is automatic)
    cache_copy(dst_neuron, src_neuron)

    # Verify exact match (pure copy, no tolerance)
    assert torch.allclose(
        dst_neuron.cpu(), src, rtol=0, atol=0,
    ), f"Copy mismatch for shape {shape}"
