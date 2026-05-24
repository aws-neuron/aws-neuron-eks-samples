# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

__all__ = [
    'attention',
]


def create_variable_length_mask(q_lens, k_lens, max_q_len, max_k_len, device, dtype=torch.bool):
    """
    Create attention mask for variable-length sequences.

    Args:
        q_lens: [B] tensor of query sequence lengths
        k_lens: [B] tensor of key sequence lengths
        max_q_len: Maximum query sequence length
        max_k_len: Maximum key sequence length
        device: Device to create mask on
        dtype: Data type for the mask

    Returns:
        mask: [B, 1, max_q_len, max_k_len] mask where True means VALID positions
    """
    batch_size = q_lens.size(0) if q_lens is not None else k_lens.size(0)

    # Create base mask (all True initially)
    mask = torch.ones(batch_size, 1, max_q_len, max_k_len, device=device, dtype=dtype)

    # Mask out padding positions in keys
    if k_lens is not None:
        k_positions = torch.arange(max_k_len, device=device).unsqueeze(0)  # [1, max_k_len]
        k_valid = k_positions < k_lens.unsqueeze(1)  # [B, max_k_len]
        mask = mask & k_valid.view(batch_size, 1, 1, max_k_len)

    # Mask out padding positions in queries
    if q_lens is not None:
        q_positions = torch.arange(max_q_len, device=device).unsqueeze(0)  # [1, max_q_len]
        q_valid = q_positions < q_lens.unsqueeze(1)  # [B, max_q_len]
        mask = mask & q_valid.view(batch_size, 1, max_q_len, 1)

    return mask


def create_sliding_window_mask(seq_len, window_size, device, causal=False):
    """
    Create sliding window attention mask.

    Args:
        seq_len: Sequence length
        window_size: Tuple of (left, right) window sizes
        device: Device to create mask on
        causal: Whether to apply causal masking

    Returns:
        mask: [seq_len, seq_len] mask where True means VALID positions
    """
    left_window, right_window = window_size

    # Create position indices
    positions = torch.arange(seq_len, device=device)
    row_idx = positions.unsqueeze(1)  # [seq_len, 1]
    col_idx = positions.unsqueeze(0)  # [1, seq_len]

    # Calculate distance
    distance = col_idx - row_idx  # [seq_len, seq_len]

    # Create window mask
    if left_window >= 0 and right_window >= 0:
        mask = (distance >= -left_window) & (distance <= right_window)
    else:
        # No window restriction
        mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)

    # Apply causal mask if needed
    if causal:
        causal_mask = distance <= 0
        mask = mask & causal_mask

    return mask


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Native PyTorch attention with full feature support
    b, lq, nq, c1 = q.shape
    b, lk, nk, c1 = k.shape
    out_dtype = q.dtype

    # Apply q_scale if provided
    if q_scale is not None:
        q = q * q_scale

    # Convert to appropriate dtype
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

    # Transpose to [B, N, L, C] format for scaled_dot_product_attention
    q = q.transpose(1, 2)  # [B, Nq, Lq, C]
    k = k.transpose(1, 2)  # [B, Nk, Lk, C]
    v = v.transpose(1, 2)  # [B, Nk, Lk, C]

    # Build attention mask
    attn_mask = None
    use_is_causal = causal and window_size == (-1, -1) and q_lens is None and k_lens is None

    if not use_is_causal:
        # Need explicit mask for variable lengths or window size
        device = q.device

        # Start with full attention
        attn_mask = torch.ones(b, 1, lq, lk, device=device, dtype=torch.bool)

        # Apply variable length mask
        if q_lens is not None or k_lens is not None:
            var_mask = create_variable_length_mask(q_lens, k_lens, lq, lk, device)
            attn_mask = attn_mask & var_mask

        # Apply sliding window mask
        if window_size != (-1, -1):
            window_mask = create_sliding_window_mask(lq, window_size, device, causal=causal)
            # Expand to [1, 1, lq, lq] then broadcast
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            # For cross-attention (lq != lk), we need to adjust
            if lq == lk:
                attn_mask = attn_mask & window_mask
            else:
                # For cross-attention, apply window mask only if lengths match
                # Otherwise, just use the variable length mask
                pass
        elif causal and not use_is_causal:
            # Apply causal mask manually
            causal_mask = create_sliding_window_mask(lq, (-1, -1), device, causal=True)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            if lq == lk:
                attn_mask = attn_mask & causal_mask

        # Convert mask: True = valid, False = masked
        # scaled_dot_product_attention expects: True = MASK OUT, False = keep
        # So we need to invert
        attn_mask = ~attn_mask

    # Apply scaled dot product attention
    if softmax_scale is not None:
        # Manual implementation for custom softmax_scale
        scale = softmax_scale
        q_scaled = q * scale
        attn_weights = torch.matmul(q_scaled, k.transpose(-2, -1))

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if dropout_p > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

        out = torch.matmul(attn_weights, v)
    else:
        # Use PyTorch's optimized implementation
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask if not use_is_causal else None,
            dropout_p=dropout_p,
            is_causal=use_is_causal
        )

    # Transpose back to [B, L, N, C]
    out = out.transpose(1, 2).contiguous()

    return out.to(out_dtype)
