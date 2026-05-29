"""Test the neuron-science-team self-attention kernel against PyTorch SDPA.

Run on Neuron:
    NEURON_RT_NUM_CORES=4 python test_nst_self_attn.py

Tests the kernel at production shapes for 1.3B model (TP=4):
  - num_heads_per_rank=3, head_dim=128
  - seq_q: 2574 (3 frames × 858), 12870 (15 frames × 858)
  - seq_k: 8192, 16384, 24576 (padded to ATTN_SEQLEN_MULTIPLE)

Compares against PyTorch scaled_dot_product_attention on CPU.
"""
import os
import sys
import math

if "NEURON_RT_NUM_CORES" not in os.environ:
    os.environ["NEURON_RT_NUM_CORES"] = "4"

import torch
import torch.nn.functional as F

# Add science_team kernels to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "science_team"))

DEVICE = torch.device("neuron")
TOL = 0.05  # bf16 tolerance


def pytorch_reference(q, k, v, softmax_scale, actual_seqlen_k):
    """PyTorch SDPA reference.

    q: [N, D, seq_q]
    k: [N, D, seq_k]
    v: [N, seq_k, D]
    Returns: [seq_q, N, D]
    """
    N, D, seq_q = q.shape
    seq_k = k.shape[2]

    # QK^T: [N, seq_q, seq_k]
    scores = torch.matmul(q.transpose(1, 2), k) * softmax_scale

    # Mask: positions beyond actual_seqlen_k are -inf
    if actual_seqlen_k < seq_k:
        mask = torch.zeros(seq_q, seq_k, dtype=scores.dtype, device=scores.device)
        mask[:, actual_seqlen_k:] = float('-inf')
        scores = scores + mask.unsqueeze(0)

    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    # attn @ v: [N, seq_q, D]
    out = torch.matmul(attn, v)
    # Return: [seq_q, N, D]
    return out.permute(1, 0, 2).contiguous()


def test_nst_self_attn(N, seq_q, seq_k, D, actual_seqlen_k, dtype=torch.bfloat16):
    """Test science team self-attention kernel."""
    name = f"nst_self_attn N={N} seq_q={seq_q} seq_k={seq_k} actual_k={actual_seqlen_k}"
    print(f"  Testing: {name}")

    torch.manual_seed(42)
    softmax_scale = 1.0 / math.sqrt(D)

    # Generate inputs
    q = torch.randn(N, D, seq_q, dtype=dtype)
    k = torch.randn(N, D, seq_k, dtype=dtype)
    v = torch.randn(N, seq_k, D, dtype=dtype)

    # CPU reference
    expected = pytorch_reference(q, k, v, softmax_scale, actual_seqlen_k)

    # Move to Neuron
    q_n = q.to(DEVICE)
    k_n = k.to(DEVICE)
    v_n = v.to(DEVICE)

    try:
        from science_team.kernels.self_attention import wan_flash_self_attn
        out = wan_flash_self_attn(
            q_n, k_n, v_n,
            softmax_scale=softmax_scale,
            actual_seqlen_k=actual_seqlen_k,
        )
        out_cpu = out.cpu()

        max_diff = (out_cpu.float() - expected.float()).abs().max().item()
        ok = max_diff < TOL
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] max_diff = {max_diff:.6f}")
        return ok, max_diff
    except Exception as e:
        print(f"    [FAIL] ERROR: {e}")
        return False, None


def test_our_self_attn(N, seq_q, seq_k, D, actual_seqlen_k, dtype=torch.bfloat16):
    """Test our existing self-attention kernel for comparison."""
    name = f"our_self_attn N={N} seq_q={seq_q} seq_k={seq_k} actual_k={actual_seqlen_k}"
    print(f"  Testing: {name}")

    torch.manual_seed(42)
    softmax_scale = 1.0 / math.sqrt(D)

    # Generate inputs (our kernel needs padded seq_q to 128, identity matrix, mask)
    P = 128
    pad_q = (P - seq_q % P) % P
    q = torch.randn(N, D, seq_q + pad_q, dtype=dtype)
    k = torch.randn(N, D, seq_k, dtype=dtype)
    v = torch.randn(N, seq_k, D, dtype=dtype)
    identity = torch.eye(D, dtype=dtype)
    mask = torch.zeros(P, seq_k, dtype=dtype)
    if actual_seqlen_k < seq_k:
        mask[:, actual_seqlen_k:] = float('-inf')

    # CPU reference (use unpadded q)
    expected = pytorch_reference(q[:, :, :seq_q], k, v, softmax_scale, actual_seqlen_k)

    # Move to Neuron
    q_n = q.to(DEVICE)
    k_n = k.to(DEVICE)
    v_n = v.to(DEVICE)
    identity_n = identity.to(DEVICE)
    mask_n = mask.to(DEVICE)

    num_sections = seq_k // 8192

    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.self_attention import wan_flash_self_attn
        kernel = wrap_nki(wan_flash_self_attn)
        out = kernel(
            q_n, k_n, v_n, identity_n, mask_n,
            softmax_scale=softmax_scale,
            num_sections=num_sections,
        )
        out_cpu = out[:seq_q].cpu()

        max_diff = (out_cpu.float() - expected.float()).abs().max().item()
        ok = max_diff < TOL
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] max_diff = {max_diff:.6f}")
        return ok, max_diff
    except Exception as e:
        print(f"    [FAIL] ERROR: {e}")
        return False, None


if __name__ == "__main__":
    print("=" * 60)
    print("  Self-Attention Kernel Test: NST vs Our vs PyTorch Reference")
    print("=" * 60)

    # Production shapes for 1.3B model TP=4
    N = 3        # heads per rank
    D = 128      # head_dim
    ATTN_MULTIPLE = 8192

    test_cases = [
        # (seq_q, seq_k, actual_seqlen_k) — seq_k padded to ATTN_MULTIPLE
        (2574, 8192, 2574),    # cache-update: 3 frames, small cache
        (2574, 8192, 7722),    # streaming: 3 frames, medium cache
        (12870, 16384, 12870), # initial window: 15 frames
        (2574, 24576, 20592),  # streaming: 3 frames, large cache (steady state)
    ]

    results = []
    print("\n--- NST Science Team Kernel ---")
    for seq_q, seq_k, actual_k in test_cases:
        ok, diff = test_nst_self_attn(N, seq_q, seq_k, D, actual_k)
        results.append(("nst", seq_q, seq_k, actual_k, ok, diff))

    print("\n--- Our Existing Kernel ---")
    for seq_q, seq_k, actual_k in test_cases:
        ok, diff = test_our_self_attn(N, seq_q, seq_k, D, actual_k)
        results.append(("ours", seq_q, seq_k, actual_k, ok, diff))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for kernel, seq_q, seq_k, actual_k, ok, diff in results:
        status = "PASS" if ok else "FAIL"
        diff_str = f"{diff:.6f}" if diff is not None else "ERROR"
        print(f"  [{status}] {kernel:4s} q={seq_q:5d} k={seq_k:5d} actual={actual_k:5d}  diff={diff_str}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    sys.exit(0 if all_pass else 1)
