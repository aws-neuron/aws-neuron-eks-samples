# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Optimized for batch_size=1: static input/output shapes, no packing/unpacking.
# Uses fake-batch-2 trick for K-side masking via cu_seqlens.
import torch
import flash_attn

__all__ = ['flash_attn_varlen_b1']


def flash_attn_varlen_b1(q, k, v, valid_q=None, valid_k=None, dtype=torch.bfloat16):
    """
    Static-shape flash attention for batch_size=1 using flash_attn_varlen_func.

    Full padded tensors go directly to the kernel. K-side masking is achieved
    via a fake batch=2 split in cu_seqlens: seq 0 = valid tokens (real
    computation), seq 1 = garbage tokens (wasted but harmless). When all K
    tokens are valid, a simple batch=1 call is used instead.

    q: [1, Lq, N, D]
    k: [1, Lk, N, D]
    v: [1, Lk, N, D]
    valid_q: int, optional. Number of valid Q tokens. Required when valid_k < Lk.
    valid_k: int, optional. Number of valid K tokens. If None, all K valid.

    Returns: [1, Lq, N, D] — same static shape as input q.
             Valid Q positions have correct attention output.
             Padding Q positions contain garbage (caller slices valid frames).
    """
    assert q.size(0) == 1, f"batch_size must be 1, got {q.size(0)}"
    half_dtypes = (torch.float16, torch.bfloat16)
    lq, lk = q.size(1), k.size(1)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q.squeeze(0))  # [Lq, N, D] — static
    k = half(k.squeeze(0))  # [Lk, N, D] — static
    v = half(v.squeeze(0))  # [Lk, N, D] — static
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if valid_k is not None and valid_k < lk:
        # K needs masking: fake batch=2 (valid seq + garbage seq)
        assert valid_q is not None
        cu_q = torch.tensor([0, valid_q, lq], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, valid_k, lk], dtype=torch.int32, device=k.device)
        x = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=max(valid_q, lq - valid_q),
            max_seqlen_k=max(valid_k, lk - valid_k))
    else:
        # All K valid: batch=1, all tokens processed
        cu_q = torch.tensor([0, lq], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, lk], dtype=torch.int32, device=k.device)
        x = flash_attn.flash_attn_varlen_func(
            q, k, v, cu_q, cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk)

    return x.unsqueeze(0).to(out_dtype)  # [1, Lq, N, D] — static
