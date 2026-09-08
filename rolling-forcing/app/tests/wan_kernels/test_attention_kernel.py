from typing import Tuple

import math
import torch
import pytest

from kernels.self_attention import wan_flash_self_attn
from kernels.cross_attention import wan_cross_attn
from tests.neuron_profiler import profile_kernel


def ref_attention(*, q, k, v, softmax_scale, kernel_dtype, accum_dtype):
    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q.to(accum_dtype), k.to(accum_dtype)) * softmax_scale  # (batch, seqlen_q, seqlen_k)

    # Apply softmax
    exp_scores = torch.exp(scores - torch.max(scores, dim=-1, keepdims=True)[0]).to(kernel_dtype)

    # Apply attention to values: attention_weights @ V
    expected = torch.matmul(exp_scores.to(accum_dtype), v.to(accum_dtype))  # (batch, seqlen_q, d_head)
    expected = (expected / exp_scores.to(accum_dtype).sum(axis=-1, keepdims=True)).to(kernel_dtype)
    return expected


def gen_test_inputs(
    batch_size: int,
    seqlen_q: int,
    seqlen_k: int,
    d_head: int,
    dtype: torch.dtype,
    is_cross_attn: bool = False,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:

    accum_dtype = torch.float32

    # Generate input tensors in the format expected by the kernel
    q = ((torch.rand(batch_size, d_head, seqlen_q, dtype=dtype) - 0.5) * 2)
    k = ((torch.rand(batch_size, d_head, seqlen_k, dtype=dtype) - 0.5) * 2)
    v = ((torch.rand(batch_size, seqlen_k, d_head, dtype=dtype) - 0.5) * 2)

    # Identity matrix for transpose operations
    identity = torch.eye(128, dtype=dtype)

    # Compute expected output using reference implementation
    softmax_scale = 1.0 / math.sqrt(d_head)

    # Convert to reference format for expected computation
    expected = ref_attention(
        q=q.transpose(1, 2),
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        kernel_dtype=dtype,
        accum_dtype=accum_dtype,
    ).transpose(0, 1)

    # Pad K and V to next multiple of section_len (8192) for self attention kernel
    padded_seqlen_k = math.ceil(seqlen_k / 8192) * 8192
    if not is_cross_attn and padded_seqlen_k != seqlen_k:
        k_padded = torch.zeros(batch_size, d_head, padded_seqlen_k, dtype=dtype)
        k_padded[:, :, :seqlen_k] = k
        k = k_padded
        v_padded = torch.zeros(batch_size, padded_seqlen_k, d_head, dtype=dtype)
        v_padded[:, :seqlen_k, :] = v
        v = v_padded

    return q, k, v, identity, expected


@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,d_head,use_dynamic_loop",
    [
        (1, 4680, 4680, 128, False),
        (1, 18720, 18720, 128, False),
        (1, 23400, 9360, 128, False),
        (12, 4680, 4680, 128, True),
        (12, 18720, 18720, 128, True),
        (12, 23400, 9360, 128, True),
    ],
)
def test_wan_self_attention(
    batch_size,
    seqlen_q,
    seqlen_k,
    d_head,
    use_dynamic_loop,
):
    dtype = torch.bfloat16
    q_data, k_data, v_data, identity_data, expected = gen_test_inputs(
        batch_size, seqlen_q, seqlen_k, d_head, dtype
    )

    softmax_scale = 1.0 / math.sqrt(d_head)
    q_data = q_data.to("neuron")
    k_data = k_data.to("neuron")
    v_data = v_data.to("neuron")
    identity_data = identity_data.to("neuron")

    if use_dynamic_loop:
        from nki.compiler.ncc_driver import CompileOptions
        from torch_neuronx.jit import jit
        from torch_neuronx.utils import get_platform_target
        def _make_compile_opts():
            """Create CompileOptions with instruction scheduling disabled. Required by dynamic loop"""
            target = get_platform_target()
            opts = CompileOptions(target=target, verbose=False)
            return opts.set_pipeline_options("enable-instruction-scheduling=false")

        kernel = jit(
            wan_flash_self_attn,
            is_nki_kb=True,
            compiler_args=_make_compile_opts(),
        )
        out = kernel(
            q_data,
            k_data,
            v_data,
            identity_data,
            softmax_scale=softmax_scale,
            actual_seqlen_k=seqlen_k,
            use_dynamic_loop=True,
        ).cpu()
    else:
        out = wan_flash_self_attn(
            q_data,
            k_data,
            v_data,
            identity_data,
            softmax_scale=softmax_scale,
            actual_seqlen_k=seqlen_k,
        ).cpu()

    assert out.shape == expected.shape, (
        f"Output shape mismatch: {out.shape} vs {expected.shape}"
    )

    assert torch.allclose(out, expected.to(out.dtype), rtol=1e-2, atol=1e-3), (
        f"compile_and_execute output does not match expected. "
        f"Max absolute error: {torch.max(torch.abs(out - expected)):.6f}"
    )

    # Profile using neuron-profile
    kernel_obj = kernel if use_dynamic_loop else wan_flash_self_attn
    summary = profile_kernel(kernel_obj)
    total_us = summary["total_active_time"] * 1e6
    print(
        f"\n[self_attn] batch={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
        f"d_head={d_head}, dynamic_loop={use_dynamic_loop}\n"
        f"time: {total_us:.2f} us"
    )


@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,d_head",
    [
        (1, 23400, 512, 128),
        (12, 23400, 512, 128),
    ],
)
def test_wan_cross_attention(
    batch_size,
    seqlen_q,
    seqlen_k,
    d_head,
):
    dtype = torch.bfloat16
    q_data, k_data, v_data, identity_data, expected = gen_test_inputs(
        batch_size, seqlen_q, seqlen_k, d_head, dtype, is_cross_attn=True
    )

    softmax_scale = 1.0 / math.sqrt(d_head)
    q_data = q_data.to("neuron")
    k_data = k_data.to("neuron")
    v_data = v_data.to("neuron")
    identity_data = identity_data.to("neuron")

    # accuracy validation + warmup
    out = wan_cross_attn(
        q_data, k_data, v_data, identity_data,
        softmax_scale=softmax_scale,
    ).cpu()

    assert out.shape == expected.shape, (
        f"Output shape mismatch: {out.shape} vs {expected.shape}"
    )

    assert torch.allclose(out, expected.to(out.dtype), rtol=1e-2, atol=1e-3), (
        f"compile_and_execute output does not match expected. "
        f"Max absolute error: {torch.max(torch.abs(out - expected)):.6f}"
    )

    # Profile using neuron-profile
    summary = profile_kernel(wan_cross_attn)
    total_us = summary["total_active_time"] * 1e6
    print(
        f"\n[cross_attn] batch={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, "
        f"d_head={d_head}\n"
        f"time: {total_us:.2f} us"
    )
