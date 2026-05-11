"""Single test runner for all three NKI kernels: rope, cross-attn, self-attn.

Run:
    NEURON_RT_NUM_CORES=4 python /workspace/video-streaming-develop/test_all_kernels.py

Prints one-line pass/fail per test with max_diff. Exit code 0 if all pass.

Production shapes from layers.py CausalWanSelfAttention:
  1.3B model:
    Small config (30x52 latent): frame_length=1560, block_length=4680
    Medium config (44x78 latent): frame_length=858, block_length=2574
    num_heads=12, head_dim=128, T5_seq_k=512, section_len=8192

  14B model (TP=8):
    Small config (60x104 latent): frame_length=1560, block_length=4680
    num_heads_per_rank=5 (40 total / 8 TP), head_dim=128
    T5_seq_k=512, section_len=8192
    Full 15-frame attention: seq_q=23424 (padded from 15*1560=23400)
"""
import os
import sys
import math
import traceback

if "NEURON_RT_NUM_CORES" not in os.environ:
    os.environ["NEURON_RT_NUM_CORES"] = "4"

import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/video-streaming-develop")
sys.path.insert(0, "/workspace/video-streaming-develop/kernels")

DEVICE = torch.device("neuron")
TOL = 0.05  # bf16 tolerance

results = []


def record(name, max_diff, err=None):
    ok = err is None and max_diff is not None and max_diff < TOL
    status = "PASS" if ok else "FAIL"
    if err:
        msg = f"  [{status}] {name:<55s}  ERROR: {err[:120]}"
    else:
        msg = f"  [{status}] {name:<55s}  max_diff = {max_diff:.6f}"
    print(msg)
    results.append((name, ok, max_diff, err))


# ════════════════════════════════════════════════════════════════════════
# Helper: rope_params (from layers.py)
# ════════════════════════════════════════════════════════════════════════
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.cos(freqs).float(), torch.sin(freqs).float()


# ════════════════════════════════════════════════════════════════════════
# Helper: causal_rope_apply (CPU reference, from layers.py)
# ════════════════════════════════════════════════════════════════════════
def causal_rope_apply(x, grid_sizes, freqs_cos, freqs_sin, start_frame=torch.tensor(0)):
    n, c = x.size(2), x.size(3) // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    f, h, w = grid_sizes
    seq_len = f * h * w
    frame_idx = start_frame + torch.arange(f)
    cos = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    sin = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    x_0 = x[:, :seq_len].to(torch.float32)
    x_pairs = x_0.reshape(1, seq_len, n, c, 2)
    x_re = x_pairs[:, :, :, :, 0:1].reshape(1, seq_len, n, c)
    x_im = x_pairs[:, :, :, :, 1:2].reshape(1, seq_len, n, c)
    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos
    x_0 = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], dim=-1)
    x_0 = x_0.reshape(1, seq_len, n, c * 2)
    return x_0.type_as(x)


# ════════════════════════════════════════════════════════════════════════
# Helper: build cos_sin tensor for NKI rope kernel
# ════════════════════════════════════════════════════════════════════════
def build_cos_sin_for_nki(grid_sizes, freqs_cos, freqs_sin, start_frame, D):
    f, h, w = grid_sizes
    seq_len = f * h * w
    c = D // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    frame_idx = start_frame + torch.arange(f)
    cos_half = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, -1)
    sin_half = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, -1)
    cos_expanded = cos_half.repeat_interleave(2, dim=-1)
    sin_expanded = sin_half.repeat_interleave(2, dim=-1)
    sign = torch.ones(D)
    sign[0::2] = -1.0
    sin_signed = sin_expanded * sign.unsqueeze(0)
    return torch.cat([cos_expanded, sin_signed], dim=-1).to(torch.float32)


# ════════════════════════════════════════════════════════════════════════
# ROPE
# ════════════════════════════════════════════════════════════════════════
def test_rope():
    print("\n── RoPE (causal_rope_rotation) ──────────────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from rope import causal_rope_rotation
    except Exception as e:
        record("rope: import", None, f"{type(e).__name__}: {e}")
        return

    D = 128

    def _make_freqs(d):
        c = d // 2
        s0 = c - 2 * (c // 3)
        s1 = c // 3
        cos_f, sin_f = rope_params(1024, 2 * s0)
        cos_h, sin_h = rope_params(1024, 2 * s1)
        cos_w, sin_w = rope_params(1024, 2 * s1)
        return (torch.cat([cos_f, cos_h, cos_w], dim=1),
                torch.cat([sin_f, sin_h, sin_w], dim=1))

    fc, fs = _make_freqs(D)

    # Test configs: (name, grid, start_frame, num_heads)
    configs = [
        # 1.3B model: h=30, w=52, frame_length=1560, N=12
        ("1.3B_small_anchor_sf0",  (3, 30, 52),  0, 12),
        ("1.3B_small_anchor_sf3",  (3, 30, 52),  3, 12),
        ("1.3B_small_5frame_sf0",  (5, 30, 52),  0, 12),
        ("1.3B_small_full_sf0",    (15, 30, 52), 0, 12),
        ("1.3B_small_full_sf3",    (15, 30, 52), 3, 12),
        # 1.3B: h=22, w=39, frame_length=858
        ("1.3B_med_anchor_sf0",    (3, 22, 39),  0, 12),
        ("1.3B_med_anchor_sf7",    (3, 22, 39),  7, 12),
        ("1.3B_med_5frame_sf0",    (5, 22, 39),  0, 12),
        # 14B TP=8: h=60, w=104 (but same spatial after patchify → h=30,w=52)
        # Actually frame_length=1560 same as 1.3B, but N=5 heads per rank
        ("14B_TP8_anchor_sf0",     (3, 30, 52),  0, 5),
        ("14B_TP8_anchor_sf3",     (3, 30, 52),  3, 5),
        ("14B_TP8_5frame_sf0",     (5, 30, 52),  0, 5),
        ("14B_TP8_full_sf0",       (15, 30, 52), 0, 5),
    ]

    wrapped = wrap_nki(causal_rope_rotation)

    for name, grid, sf, N in configs:
        try:
            torch.manual_seed(0)
            f, h, w = grid
            S = f * h * w
            x = torch.randn(1, S, N, D, dtype=torch.bfloat16)
            sf_t = torch.tensor(sf)

            # CPU reference
            ref = causal_rope_apply(x.clone(), grid, fc, fs, sf_t)[0]  # [seq, N, D]

            # NKI kernel: needs [seq, N, D] input, [seq, 2D] cos_sin
            cos_sin = build_cos_sin_for_nki(grid, fc, fs, sf_t, D)
            P = 128
            pad = (P - S % P) % P
            x_nki = x[0]  # [S, N, D]
            if pad > 0:
                x_nki = F.pad(x_nki, (0, 0, 0, 0, 0, pad))
                cos_sin = F.pad(cos_sin, (0, 0, 0, pad))

            out = wrapped(x_nki.to(DEVICE), cos_sin.to(DEVICE), N, D).cpu()
            out = out[:S]  # trim padding

            diff = (out.float() - ref.float()).abs().max().item()
            record(f"rope: {name}", diff)
        except Exception as e:
            record(f"rope: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION
# ════════════════════════════════════════════════════════════════════════
def test_cross_attn():
    print("\n── Cross-Attention (wan_cross_attn) ─────────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from cross_attention import wan_cross_attn
    except Exception as e:
        record("cross_attn: import", None, f"{type(e).__name__}: {e}")
        return

    def _sdpa_ref(q, k, v, scale):
        """q: (bs,d,Sq), k: (bs,d,Sk), v: (bs,Sk,d) → out (Sq,bs,d)"""
        qa = q.permute(0, 2, 1).float()
        ka = k.permute(0, 2, 1).float()
        va = v.float()
        scores = torch.matmul(qa, ka.transpose(-1, -2)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, va)
        return out.permute(1, 0, 2).to(q.dtype)

    D, Sk = 128, 512
    P = 128

    # (name, seq_q_raw, num_heads)
    configs = [
        # 1.3B (N=12): Small frame_length=1560
        ("1.3B_small_1frame",    1560, 12),
        ("1.3B_small_3frame",    4680, 12),
        ("1.3B_small_5frame",    7800, 12),
        # 1.3B: Medium frame_length=858
        ("1.3B_med_1frame",      858,  12),
        ("1.3B_med_3frame",      2574, 12),
        ("1.3B_med_5frame",      4290, 12),
        # 14B TP=8 (N=5): frame_length=1560
        ("14B_TP8_1frame",       1560,  5),
        ("14B_TP8_3frame",       4680,  5),
        ("14B_TP8_5frame",       7800,  5),
        # 14B TP=8: full 15 frames (production shape)
        ("14B_TP8_15frame",     23400,  5),
    ]

    wrapped = wrap_nki(wan_cross_attn)

    for name, Sq_raw, N in configs:
        try:
            torch.manual_seed(0)
            # Pad Sq to multiple of 128
            pad = (P - Sq_raw % P) % P
            Sq = Sq_raw + pad

            q = torch.randn(N, D, Sq_raw, dtype=torch.bfloat16)
            k = torch.randn(N, D, Sk, dtype=torch.bfloat16)
            v = torch.randn(N, Sk, D, dtype=torch.bfloat16)
            identity = torch.eye(128, dtype=torch.bfloat16)
            scale = 1.0 / math.sqrt(D)

            ref = _sdpa_ref(q, k, v, scale)  # (Sq_raw, N, D)

            # Pad q for kernel
            if pad > 0:
                q_padded = F.pad(q, (0, pad))
            else:
                q_padded = q

            out = wrapped(q_padded.to(DEVICE), k.to(DEVICE), v.to(DEVICE),
                          identity.to(DEVICE), softmax_scale=scale).cpu()
            out = out[:Sq_raw]  # trim

            diff = (out.float() - ref.float()).abs().max().item()
            record(f"cross_attn: {name} (Sq={Sq_raw}→{Sq})", diff)
        except Exception as e:
            record(f"cross_attn: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# SELF-ATTENTION
# ════════════════════════════════════════════════════════════════════════
def test_self_attn():
    print("\n── Self-Attention (wan_flash_self_attn) ─────────────────────────")
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from self_attention import wan_flash_self_attn
    except Exception as e:
        record("self_attn: import", None, f"{type(e).__name__}: {e}")
        return

    def _sdpa_ref(q, k, v, k_valid, scale):
        """q: (N,D,Sq), k: (N,D,Sk), v: (N,Sk,D) → out (Sq,N,D)"""
        qa = q.permute(0, 2, 1).unsqueeze(0).float()
        ka = k[:, :, :k_valid].permute(0, 2, 1).unsqueeze(0).float()
        va = v[:, :k_valid, :].unsqueeze(0).float()
        out = F.scaled_dot_product_attention(qa, ka, va, scale=scale)
        return out[0].permute(1, 0, 2).to(q.dtype)

    D, P = 128, 128
    SECTION = 8192
    scale = 1.0 / math.sqrt(D)

    # Production shapes from CausalWanSelfAttention
    # 1.3B: frame_length=1560, block_length=4680, max_attn=32760, N=12
    # 14B TP=8: frame_length=1560, block_length=4680, N=5
    #   Full 15-frame: seq_q=4680, but KV cache has all 23400 padded to 24576
    configs = [
        # (name, seq_q, seq_k, k_valid, num_heads)
        # 1.3B (N=12): Small config shapes
        ("1.3B_small_anchor",         4680,  8192,  4680, 12),
        ("1.3B_small_2blocks",        4680, 16384,  9360, 12),
        ("1.3B_small_5blocks",        4680, 24576, 23400, 12),
        ("1.3B_small_full_cache",     4680, 32768, 32760, 12),
        ("1.3B_small_exact_1sec",     4680,  8192,  8192, 12),
        ("1.3B_small_exact_2sec",     4680, 16384, 16384, 12),
        ("1.3B_small_past_sec_edge",  4680, 16384,  8193, 12),
        # 1.3B: Medium config shapes
        ("1.3B_med_anchor",           2574,  8192,  2574, 12),
        ("1.3B_med_2blocks",          2574,  8192,  5148, 12),
        ("1.3B_med_3blocks",          2574,  8192,  7722, 12),
        ("1.3B_med_full_cache",       2574, 24576, 18018, 12),
        # 14B TP=8 (N=5): production shapes
        ("14B_TP8_anchor",            4680,  8192,  4680,  5),
        ("14B_TP8_2blocks",           4680, 16384,  9360,  5),
        ("14B_TP8_5blocks",           4680, 24576, 23400,  5),
        ("14B_TP8_full_cache",        4680, 32768, 32760,  5),
        # 14B: large seq_q (15 frames = 23400 → padded to 23424)
        ("14B_TP8_full_15f_anchor",  23424, 24576, 23400,  5),
        ("14B_TP8_full_15f_cache",   23424, 32768, 32760,  5),
    ]

    wrapped = wrap_nki(wan_flash_self_attn)

    for name, Sq, Sk, k_valid, N in configs:
        try:
            torch.manual_seed(0)
            q = torch.randn(N, D, Sq, dtype=torch.bfloat16)
            k = torch.randn(N, D, Sk, dtype=torch.bfloat16)
            v = torch.randn(N, Sk, D, dtype=torch.bfloat16)
            identity = torch.eye(D, dtype=torch.bfloat16)

            ref = _sdpa_ref(q, k, v, k_valid, scale)

            # Pad q to multiple of 128
            pad_q = (P - Sq % P) % P
            q_p = F.pad(q, (0, pad_q)) if pad_q else q

            # Build mask: (128, Sk), 0 for valid, -inf for masked
            mask = torch.zeros(P, Sk, dtype=torch.bfloat16)
            if k_valid < Sk:
                mask[:, k_valid:] = float('-inf')

            num_sections = Sk // SECTION

            out = wrapped(
                q_p.to(DEVICE), k.to(DEVICE), v.to(DEVICE),
                identity.to(DEVICE), mask.to(DEVICE),
                softmax_scale=scale, num_sections=num_sections,
            )
            out = out[:Sq].cpu()

            diff = (out.float() - ref.float()).abs().max().item()
            record(f"self_attn: {name} (Sq={Sq},Sk={Sk},kv={k_valid})", diff)
        except Exception as e:
            record(f"self_attn: {name}", None, f"{type(e).__name__}: {e}")
            traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 72)
    print("NKI Kernel Test Suite — All Three Kernels")
    print(f"Device: {DEVICE}  Tolerance: max_diff < {TOL}")
    print("=" * 72)

    test_rope()
    test_cross_attn()
    test_self_attn()

    print("\n" + "=" * 72)
    n_pass = sum(1 for _, ok, _, _ in results if ok)
    n_total = len(results)
    all_ok = n_pass == n_total
    print(f"SUMMARY: {n_pass}/{n_total} passed  {'✅ ALL PASS' if all_ok else '❌ FAILURES'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)
