#!/usr/bin/env python3
"""Self-attention diagnostic: PyTorch SDPA reference at production shapes.

Run on the dummy pod:
    NEURON_RT_NUM_CORES=4 python /workspace/self_attn_diag.py

Phase 1: Establishes PyTorch SDPA baseline — validates shapes, produces
reference outputs for each production shape.

Phase 2 (after porting): Tests NKI kernel against SDPA reference.

Production shapes from layers.py CausalWanSelfAttention:
    - num_heads (bs) = 12
    - head_dim (d) = 128
    - frame_length = 1560 (h=30, w=52)
    - section_len = 8192 (kernel K/V processing chunk)
    - ATTN_SEQLEN_MULTIPLE = 8192

    q: (12, 128, seq_q)    k: (12, 128, seq_k)    v: (12, seq_k, 128)
    out: (seq_q, 12, 128)

Key: seq_k is always padded to multiple of 8192, but actual_seqlen_k
tells the kernel how many K tokens are valid. The kernel must mask
positions beyond actual_seqlen_k to -inf before softmax.
"""

import os
# Auto-configure Neuron core count if not already set
# Pod has 8 physical cores with NEURON_LOGICAL_NC_CONFIG=2 → 4 logical cores
if "NEURON_RT_NUM_CORES" not in os.environ:
    os.environ["NEURON_RT_NUM_CORES"] = "4"

import torch
import torch.nn.functional as F
import math
import time


def sdpa_reference(q, k, v, identity, softmax_scale, actual_seqlen_k=None):
    """PyTorch SDPA reference matching the NKI kernel's IO convention.

    Args:
        q: (bs, d, seq_q) bfloat16
        k: (bs, d, seq_k) bfloat16
        v: (bs, seq_k, d) bfloat16
        identity: (128, 128) — unused in reference
        softmax_scale: float
        actual_seqlen_k: int or None — if set, mask k[:, :, actual_seqlen_k:] to -inf

    Returns:
        out: (seq_q, bs, d) bfloat16 — matches NKI kernel output layout
    """
    bs, d, seq_q = q.shape
    seq_k = k.shape[2]

    # Reshape to standard attention layout: (bs, seq, d)
    q_attn = q.permute(0, 2, 1)   # (bs, seq_q, d)
    k_attn = k.permute(0, 2, 1)   # (bs, seq_k, d)
    v_attn = v                      # (bs, seq_k, d)

    # QK^T scores: (bs, seq_q, seq_k)
    scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-1, -2)) * softmax_scale

    # Mask padded K positions
    if actual_seqlen_k is not None and actual_seqlen_k < seq_k:
        scores[:, :, actual_seqlen_k:] = float('-inf')

    # Softmax + PV
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_attn.float())  # (bs, seq_q, d)

    # Permute to kernel output layout: (seq_q, bs, d)
    out = out.permute(1, 0, 2).to(q.dtype)
    return out


def test_shape(name, bs, d, seq_q, seq_k, actual_seqlen_k, softmax_scale, device="cpu"):
    """Test one shape configuration."""
    torch.manual_seed(42)

    q = torch.randn(bs, d, seq_q, dtype=torch.bfloat16, device=device)
    k = torch.randn(bs, d, seq_k, dtype=torch.bfloat16, device=device)
    v = torch.randn(bs, seq_k, d, dtype=torch.bfloat16, device=device)
    identity = torch.eye(d, dtype=torch.bfloat16, device=device)

    # Reference
    t0 = time.perf_counter()
    out_ref = sdpa_reference(q, k, v, identity, softmax_scale, actual_seqlen_k)
    t_ref = (time.perf_counter() - t0) * 1000

    # Verify output shape
    assert out_ref.shape == (seq_q, bs, d), f"Shape mismatch: {out_ref.shape} vs expected ({seq_q}, {bs}, {d})"

    # Check for NaN/Inf
    has_nan = torch.isnan(out_ref).any().item()
    has_inf = torch.isinf(out_ref).any().item()

    # Also test with F.scaled_dot_product_attention for cross-validation
    q_sdpa = q.float().permute(0, 2, 1).unsqueeze(0)  # (1, bs, seq_q, d)
    k_sdpa = k.float().permute(0, 2, 1)[:, :actual_seqlen_k].unsqueeze(0)  # (1, bs, actual_k, d)
    v_sdpa = v.float()[:, :actual_seqlen_k].unsqueeze(0)  # (1, bs, actual_k, d)
    out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, scale=softmax_scale)
    out_sdpa = out_sdpa[0].permute(1, 0, 2).to(torch.bfloat16)  # (seq_q, bs, d)

    diff = (out_ref.float() - out_sdpa.float()).abs()
    max_diff = diff.max().item()

    status = "✅" if max_diff < 1.0 and not has_nan and not has_inf else "❌"
    print(f"  {name:>35s}: q=({bs},{d},{seq_q}) k=({bs},{d},{seq_k}) actual_k={actual_seqlen_k}"
          f"  ref_time={t_ref:.1f}ms  ref_vs_sdpa_diff={max_diff:.6f}"
          f"  nan={has_nan} inf={has_inf} {status}")

    return out_ref


def test_nki_kernel(name, bs, d, seq_q, seq_k, actual_seqlen_k, softmax_scale, out_ref, device):
    """Test NKI kernel against reference (Phase 2 — after porting)."""
    torch.manual_seed(42)

    # Create on CPU with same seed as Phase 1
    q = torch.randn(bs, d, seq_q, dtype=torch.bfloat16)
    k = torch.randn(bs, d, seq_k, dtype=torch.bfloat16)
    v = torch.randn(bs, seq_k, d, dtype=torch.bfloat16)
    identity = torch.eye(d, dtype=torch.bfloat16)

    # Pad Q to multiple of 128 (NKI requires compile-time-constant slice sizes)
    P = 128
    pad_q = (-seq_q) % P  # e.g. 4680 → pad 72 → 4736
    if pad_q > 0:
        q = torch.nn.functional.pad(q, (0, pad_q))  # zero-pad last dim

    # Build mask tensor: (128, seq_k) bf16, 0 for valid, -inf for masked
    mask = torch.zeros(P, seq_k, dtype=torch.bfloat16)
    if actual_seqlen_k < seq_k:
        mask[:, actual_seqlen_k:] = float('-inf')

    # Move to Neuron device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    identity = identity.to(device)
    mask = mask.to(device)

    try:
        from kernels.self_attention import wan_flash_self_attn
        from torch_neuronx.nki_hop import wrap_nki
        kernel = wrap_nki(wan_flash_self_attn)

        num_sections = seq_k // 8192
        t0 = time.perf_counter()
        out_nki = kernel(q, k, v, identity, mask,
                         softmax_scale=softmax_scale,
                         num_sections=num_sections,
                         use_dynamic_loop=False)
        t_nki = (time.perf_counter() - t0) * 1000

        # Truncate padded output and move to CPU for comparison
        out_nki_cpu = out_nki[:seq_q].cpu()
        out_ref_cpu = out_ref.cpu()
        diff = (out_nki_cpu.float() - out_ref_cpu.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        status = "✅" if max_diff < 2.0 else "❌"
        print(f"  {name:>35s}: NKI max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}"
              f"  nki_time={t_nki:.1f}ms  pad_q={pad_q} {status}")
        return max_diff
    except Exception as e:
        import traceback
        print(f"  {name:>35s}: NKI FAILED — {e}")
        traceback.print_exc()
        return None


def main():
    print("=" * 80)
    print("Self-Attention Diagnostic")
    print("=" * 80)

    bs = 12       # num_heads
    d = 128       # head_dim
    softmax_scale = 1.0 / math.sqrt(d)  # 0.08838834764831843
    frame_length = 1560
    section_len = 8192

    # Production shapes from CausalWanSelfAttention.forward():
    # seq_q: always block_length (3 * frame_length) or full denoising length
    # seq_k: buffer size, padded to multiple of 8192
    # actual_seqlen_k: real K length (varies)

    shapes = [
        # (name, seq_q, seq_k, actual_seqlen_k)
        # First block (anchor): q=4680, k=buffer, actual_k=4680
        ("anchor_block", 4680, 8192, 4680),

        # After 2 blocks: actual_k=9360
        ("2_blocks", 4680, 16384, 9360),

        # After 5 blocks: actual_k=23400
        ("5_blocks", 4680, 24576, 23400),

        # Full cache: actual_k=32760
        ("full_cache", 4680, 32768, 32760),

        # Cache update: q=4680, k=max_attention_size
        ("cache_update_full", 4680, 32768, 32760),

        # Large seq_q (5-frame denoising): 5*1560=7800
        ("5frame_denoise", 7800, 32768, 32760),

        # Minimal: single section
        ("minimal_1section", 4680, 8192, 8192),

        # Exact section boundary
        ("exact_2sections", 4680, 16384, 16384),

        # Edge: actual_k just past section boundary
        ("past_section_edge", 4680, 16384, 8193),
    ]

    print("\n--- Phase 1: PyTorch SDPA Reference Validation ---")
    print(f"  bs={bs}, d={d}, softmax_scale={softmax_scale:.10f}")
    print(f"  section_len={section_len}, frame_length={frame_length}")
    print()

    refs = {}
    for name, seq_q, seq_k, actual_k in shapes:
        refs[name] = test_shape(name, bs, d, seq_q, seq_k, actual_k, softmax_scale)

    # Phase 2: NKI kernel test (only on Neuron device)
    print("\n--- Phase 2: NKI Kernel vs Reference ---")
    device = None
    try:
        import torch_neuronx
        device = torch.device("neuron")
        # Force a small allocation to verify device is functional
        _test = torch.zeros(1, device=device)
        del _test
        print(f"  Neuron device: {device}")
    except Exception as e:
        print(f"  Not on Neuron device — skipping NKI kernel tests.")
        print(f"    Error: {e}")

    if device is not None:
        print()
        all_pass = True
        for name, seq_q, seq_k, actual_k in shapes:
            result = test_nki_kernel(name, bs, d, seq_q, seq_k, actual_k,
                                     softmax_scale, refs[name], device)
            if result is None or result >= 2.0:
                all_pass = False

        print()
        if all_pass:
            print("🎉 ALL SHAPES PASS — NKI self-attention kernel is correct!")
        else:
            print("⚠️  Some shapes failed — see details above.")

    print()
    print("--- Kernel Architecture Notes ---")
    print(f"  section_len = {section_len}")
    print(f"  For seq_k=32768: num_sections = {32768 // section_len}")
    print(f"  For seq_k=57344: num_sections = {57344 // section_len}")
    print(f"  Each section: {section_len // 2048} x 2048-tiles, {section_len // 512} x 512-tiles")
    print(f"  ATTN_SEQLEN_MULTIPLE = {section_len}")
    print(f"  The kernel processes K/V in {section_len}-token sections with online softmax")
    print(f"  (running max + running sum across sections for numerical stability)")


if __name__ == "__main__":
    main()
