import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_neuronx.jit import jit

from models.layers import WanRMSNorm, rope_params, causal_rope_apply, ATTN_SEQLEN_MULTIPLE
from models.layers import CausalWanSelfAttention


def test_self_attn_qkv_proj():
    """Test q = norm_q(q_proj(x)).view(...), k = norm_k(k_proj(x)).view(...), v = v_proj(x).view(...)"""
    dtype = torch.bfloat16
    B, L, dim, num_heads = 1, 23400, 1536, 12
    head_dim = dim // num_heads  # 128

    x = torch.randn(B, L, dim, dtype=dtype)
    q_proj = nn.Linear(dim, dim).to(dtype)
    k_proj = nn.Linear(dim, dim).to(dtype)
    v_proj = nn.Linear(dim, dim).to(dtype)
    norm_q = WanRMSNorm(dim).to(dtype)
    norm_k = WanRMSNorm(dim).to(dtype)

    # CPU reference
    expected_q = norm_q(q_proj(x)).view(B, -1, num_heads, head_dim)
    expected_k = norm_k(k_proj(x)).view(B, -1, num_heads, head_dim)
    expected_v = v_proj(x).view(B, -1, num_heads, head_dim)

    # Neuron: chain JIT modules
    q_proj_neuron = jit(q_proj).to("neuron")
    k_proj_neuron = jit(k_proj).to("neuron")
    v_proj_neuron = jit(v_proj).to("neuron")
    norm_q_neuron = norm_q.to("neuron")
    norm_k_neuron = norm_k.to("neuron")
    x_neuron = x.to("neuron")

    result_q = norm_q_neuron(q_proj_neuron(x_neuron)).view(B, -1, num_heads, head_dim)
    result_k = norm_k_neuron(k_proj_neuron(x_neuron)).view(B, -1, num_heads, head_dim)
    result_v = v_proj_neuron(x_neuron).view(B, -1, num_heads, head_dim)

    torch.testing.assert_close(result_q.cpu(), expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_k.cpu(), expected_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_v.cpu(), expected_v, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("grid_sizes,start_frame", [
    ((15, 30, 52), 0),   # full block, first window (current_start_frame=0)
    ((15, 30, 52), 3),   # full block, later window (current_start_frame>0)
    ((3, 30, 52), 0),    # anchor block, updating_cache path
    ((3, 30, 52), 7),    # anchor block, normal denoising (rope_start_frame)
])
def test_causal_rope_apply(grid_sizes, start_frame):
    """Test causal_rope_apply: 3D RoPE with real arithmetic."""
    dtype = torch.bfloat16
    B, num_heads, head_dim = 1, 12, 128
    f, h, w = grid_sizes
    L = f * h * w

    x = torch.randn(B, L, num_heads, head_dim, dtype=dtype)

    d = head_dim
    cos_f, sin_f = rope_params(1024, d - 4 * (d // 6))
    cos_h, sin_h = rope_params(1024, 2 * (d // 6))
    cos_w, sin_w = rope_params(1024, 2 * (d // 6))
    freqs_cos = torch.cat([cos_f, cos_h, cos_w], dim=1)
    freqs_sin = torch.cat([sin_f, sin_h, sin_w], dim=1)

    start_frame_t = torch.tensor(start_frame)

    # CPU reference
    expected = causal_rope_apply(x, grid_sizes, freqs_cos, freqs_sin, start_frame=start_frame_t)

    # JIT + Neuron (start_frame is a scalar tensor — values stay out of IR)
    causal_rope_apply_neuron = jit(causal_rope_apply)
    result = causal_rope_apply_neuron(
        x.to("neuron"), grid_sizes,
        freqs_cos.to("neuron"), freqs_sin.to("neuron"),
        start_frame=start_frame_t.to("neuron"))

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Tensor read/write ops from CausalWanSelfAttention.forward
# (causal_model_opt.py lines 120-269)
#
# These tests verify that .copy_() and slice assignment work correctly on
# Neuron device with the exact production shapes from the GPU baseline.
# No JIT — tensors are moved to "neuron" and ops run directly.
# All tensors are [1, L, 12, 128] in bfloat16.
#
# Production constants:
#   frame_length=1560, block_length=4680, s=23400, max_attention_size=32760,
#   kv_cache_logical=37440, kv_cache_alloc=37440, evict_rolled=28080
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,dest_off,src_off,copy_len,src_total", [
    # ── Phase 2: Eviction (causal_model_opt.py) ──
    # When KV cache overflows, old entries are left-shifted via buffer.
    # Tests that .copy_() works with the large evict_rolled=28080 slice size.
    # src_start = sink_tokens + num_evicted = 4680 + 4680 = 9360 (typical full-cache scenario).

    # Read rolled entries from cache middle into buffer start
    ("evict_read",           0,     9360,  28080, 37440),
    # Write shifted entries back to cache (after anchor region)
    ("evict_writeback",      4680,  0,     28080, 37440),

    # ── Phase 3: updating_cache path (causal_model_opt.py) ──
    # Cache-update call: copies cache_len tokens (dynamic, ≤ max_attention_size=32760).
    # Tests the largest single copy in the forward method.

    # cache_start_pos=0: anchor is visible, copy from cache start
    ("upd_cache_read_pos0",  0,     0,     32760, 37440),
    # cache_start_pos=4680: cache grew beyond max_attention_size, read with offset
    ("upd_cache_read_pos4k", 0,     4680,  32760, 37440),
    # Overwrite anchor in buffer with RoPE'd version (src is block_length-sized tensor)
    ("anchor_from_small",    0,     0,     4680,  4680),

    # ── Phase 3: Normal denoising path (causal_model_opt.py) ──
    # Assembles attention KV from: anchor (4680) + working cache + current.
    # Each copied at exact valid length (dynamic).

    # Copy anchor v from cache start (both src and dest start at 0)
    ("anchor_v_from_cache",  0,     0,     4680,  37440),
    # Read working cache: src and dest offset by block_length=4680, wc_len=4680 (steady-state)
    ("wc_read",              4680,  4680,  4680,  37440),
    # Current tokens for non-first block: dest_off = block_length + wc_len = 9360
    # copy_len = valid_tokens = 23400, src is the roped_key tensor
    ("current_nonfirst",     9360,  0,     23400, 23400),
    # Current tokens for first block: offset=0, no anchor or wc prefix
    ("current_first",        0,     0,     23400, 23400),
])
def test_self_attn_tensor_ops(name, dest_off, src_off, copy_len, src_total):
    """Test .copy_() ops from CausalWanSelfAttention.forward on Neuron device.

    Each case mirrors a specific buffer/cache copy in the forward method with
    production shapes (B=1, num_heads=12, head_dim=128, bf16). Since .copy_()
    is pure data movement (no arithmetic), results must be bitwise identical.
    """
    dtype = torch.bfloat16
    dest_total = 37440  # kv_cache_alloc_size = 24 × 1560
    shape_suffix = (12, 128)  # (num_heads, head_dim)

    src = torch.randn(1, src_total, *shape_suffix, dtype=dtype)
    dest_cpu = torch.randn(1, dest_total, *shape_suffix, dtype=dtype)
    dest_neuron = dest_cpu.clone()

    # CPU reference
    dest_cpu[0, dest_off:dest_off + copy_len].copy_(
        src[0, src_off:src_off + copy_len])

    # Neuron: same op on device tensors
    dest_neuron = dest_neuron.to("neuron")
    src_neuron = src.to("neuron")
    dest_neuron[0, dest_off:dest_off + copy_len].copy_(
        src_neuron[0, src_off:src_off + copy_len])

    # Bitwise match — no arithmetic, pure copy
    torch.testing.assert_close(dest_neuron.cpu(), dest_cpu, rtol=0, atol=0)


@pytest.mark.parametrize("name,dest_off,copy_len,src_total", [
    # ── Phase 2: Cache write (causal_model_opt.py) ──
    # New block is written to KV cache via slice assignment (not .copy_()).
    # Tests that dest[0, a:b] = src[0, :n] works on Neuron with different offsets.

    # Anchor block: write un-roped k to cache start (local_start_index=0)
    ("assign_anchor_k",     0,    4680,  23400),
    # Non-anchor block: write roped k to cache middle (local_start_index=4680)
    ("assign_non_anchor_k", 4680, 4680,  23400),
    # Value write at cache start (anchor path)
    ("assign_v_start",      0,    4680,  23400),
    # Value write at cache middle (non-anchor path)
    ("assign_v_mid",        4680, 4680,  23400),
])
def test_self_attn_slice_assign(name, dest_off, copy_len, src_total):
    """Test slice assignment ops from CausalWanSelfAttention.forward on Neuron device.

    The forward method writes new blocks to KV cache using slice assignment:
      kv_cache["k"][0, start:end] = k[0, :block_length]
    This tests that the assignment form (vs .copy_()) works correctly on Neuron.
    """
    dtype = torch.bfloat16
    dest_total = 37440  # kv_cache_alloc_size
    shape_suffix = (12, 128)  # (num_heads, head_dim)

    src = torch.randn(1, src_total, *shape_suffix, dtype=dtype)
    dest_cpu = torch.randn(1, dest_total, *shape_suffix, dtype=dtype)
    dest_neuron = dest_cpu.clone()

    # CPU reference
    dest_cpu[0, dest_off:dest_off + copy_len] = src[0, :copy_len]

    # Neuron: same slice assignment on device tensors
    dest_neuron = dest_neuron.to("neuron")
    src_neuron = src.to("neuron")
    dest_neuron[0, dest_off:dest_off + copy_len] = src_neuron[0, :copy_len]

    # Bitwise match — no arithmetic, pure copy
    torch.testing.assert_close(dest_neuron.cpu(), dest_cpu, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# End-to-end test: RefCausalSelfAttention (CPU) vs CausalWanSelfAttention (Neuron)
#
# Simulates rolling forcing windows 0-12 and 42-45 to cover all branches:
#   - Phase 2: eviction vs no-eviction, anchor vs non-anchor write
#   - Phase 3: updating_cache (cache_start_pos=0 and >0), normal (local_start=0 and >0)
#   - Phase 4: attention kernel with varying k_len_int and valid_tokens
#   - Ramp-down: decreasing num_valid_frames (12→9→6→3)
# ---------------------------------------------------------------------------


class RefCausalSelfAttention(nn.Module):
    """CPU reference: identical cache logic, F.scaled_dot_product_attention for Phase 4."""

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=1,
                 qk_norm=True, eps=1e-6, layer_idx=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_length = 1560
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.kv_cache_logical_size = 24 * self.frame_length
        self.layer_idx = layer_idx

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs_cos, freqs_sin,
                kv_cache=None, current_start=0, cache_start=None,
                updating_cache=False, num_valid_frames=None, shared_buffers=None):
        assert kv_cache is not None
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1
        if cache_start is None:
            cache_start = current_start

        # ── Phase 1: QKV projection + RoPE ──
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen
        current_start_frame_t = torch.tensor(current_start_frame, device=x.device)
        roped_query = causal_rope_apply(
            q, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t).type_as(v)
        roped_key = causal_rope_apply(
            k, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t).type_as(v)

        grid_sizes_one_block = (3, h, w)

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        # ── Phase 2: Cache management (write + eviction) ──
        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        kv_cache_size = self.kv_cache_logical_size
        sink_tokens = self.block_length

        buffer_k, buffer_v = shared_buffers

        num_evicted = 0
        if (num_new_tokens > 0) and (
                num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens
            src_start = sink_tokens + num_evicted
            buffer_k[0, :evict_rolled].copy_(kv_cache["k"][0, src_start:src_start + evict_rolled])
            buffer_v[0, :evict_rolled].copy_(kv_cache["v"][0, src_start:src_start + evict_rolled])
            kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled].copy_(buffer_k[0, :evict_rolled])
            kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled].copy_(buffer_v[0, :evict_rolled])

        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        if local_start_index == 0:
            kv_cache["k"][0, :self.block_length] = k[0, :self.block_length]
        else:
            kv_cache["k"][0, local_start_index:local_end_index] = roped_key[0, :self.block_length]
        kv_cache["v"][0, local_start_index:local_end_index] = v[0, :self.block_length]

        if num_new_tokens > 0:
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        # ── Phase 3: Assemble KV into buffers ──
        if updating_cache:
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)

            buffer_k[0, :cache_len].copy_(
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len])
            buffer_v[0, :cache_len].copy_(
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            if cache_start_pos == 0:
                anchor_roped = causal_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(0, device=v.device)).type_as(v)
                buffer_k[0, :self.block_length].copy_(anchor_roped[0])

            k_len_int = cache_len

        else:
            offset = 0
            if local_start_index > 0:
                wc_max = self.max_attention_size - valid_tokens - self.block_length
                wc_end = local_start_index
                wc_start = max(self.block_length, wc_end - wc_max)
                wc_len = wc_end - wc_start

                wc_frame_length = wc_len // self.frame_length
                rope_start_frame = current_start_frame - wc_frame_length - 3
                anchor_roped = causal_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(rope_start_frame, device=v.device)).type_as(v)
                buffer_k[0, :self.block_length].copy_(anchor_roped[0])
                buffer_v[0, :self.block_length].copy_(kv_cache["v"][0, :self.block_length])
                offset = self.block_length

                buffer_k[0, offset:offset + wc_len].copy_(kv_cache["k"][0, wc_start:wc_start + wc_len])
                buffer_v[0, offset:offset + wc_len].copy_(kv_cache["v"][0, wc_start:wc_start + wc_len])
                offset += wc_len

            buffer_k[0, offset:offset + valid_tokens].copy_(roped_key[0, :valid_tokens])
            buffer_v[0, offset:offset + valid_tokens].copy_(v[0, :valid_tokens])
            k_len_int = offset + valid_tokens

        # ── Phase 4: F.scaled_dot_product_attention ──
        # Neuron kernel processes ALL Q positions (garbage Q beyond valid_tokens
        # still attends to valid K/V). Match that behaviour here.
        q_attn = roped_query.permute(0, 2, 1, 3)
        k_attn = buffer_k[:, :k_len_int].permute(0, 2, 1, 3)
        v_attn = buffer_v[:, :k_len_int].permute(0, 2, 1, 3)
        attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
        x = attn_out.permute(0, 2, 1, 3).flatten(2)

        # ── Phase 5: Output projection ──
        x = self.o(x)
        return x


def _make_freqs(head_dim):
    """Precompute RoPE frequencies matching CausalWanModel.__init__."""
    d = head_dim
    cos_f, sin_f = rope_params(1024, d - 4 * (d // 6))
    cos_h, sin_h = rope_params(1024, 2 * (d // 6))
    cos_w, sin_w = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_f, cos_h, cos_w], dim=1), torch.cat([sin_f, sin_h, sin_w], dim=1)


def _make_kv_cache(alloc_size, num_heads, head_dim, dtype, device):
    """Allocate a KV cache dict matching the pipeline."""
    return {
        "k": torch.zeros(1, alloc_size, num_heads, head_dim, dtype=dtype, device=device),
        "v": torch.zeros(1, alloc_size, num_heads, head_dim, dtype=dtype, device=device),
        "global_end_index": 0,
        "local_end_index": 0,
    }


def _make_shared_buffers(buf_size, num_heads, head_dim, dtype, device):
    """Allocate shared scratch/assembly buffers."""
    return (
        torch.zeros(1, buf_size, num_heads, head_dim, dtype=dtype, device=device),
        torch.zeros(1, buf_size, num_heads, head_dim, dtype=dtype, device=device),
    )


def _build_test_cases(nds=5, nfpb=3, num_blocks=42):
    """Build independent test cases with pre-computed cache index states.

    Simulates cache index evolution through all 46 windows but only emits
    test cases for target windows (W0-W12 + W42-W45). Each case includes
    the "before" cache state (global_end_index, local_end_index) so tests
    can run independently with random cache data.

    Returns list of (current_start, updating_cache, nvf,
                     global_end_index, local_end_index, description).
    """
    frame_length = 1560
    block_length = nfpb * frame_length  # 4680
    kv_cache_logical = 24 * frame_length  # 37440
    cases = []
    window_num = num_blocks + nds - 1  # 46

    target_windows = set(range(13))  # W0-W12
    target_windows |= set(range(window_num - nds + 1, window_num))  # W42-W45

    global_end = 0
    local_end = 0

    for window_index in range(window_num):
        start_block = max(0, window_index - nds + 1)
        end_block = min(num_blocks - 1, window_index)
        current_start_frame = start_block * nfpb
        current_num_frames = (end_block + 1 - start_block) * nfpb
        current_start = current_start_frame * frame_length
        cache_end = current_start + block_length

        if window_index in target_windows:
            cases.append((
                current_start, False, current_num_frames,
                global_end, local_end,
                f"W{window_index} denoise (blks={start_block}-{end_block}, nvf={current_num_frames})"
            ))

        # Simulate denoise effect on cache indices
        num_new = cache_end - global_end
        num_evicted = 0
        if num_new > 0 and num_new + local_end > kv_cache_logical:
            num_evicted = num_new + local_end - kv_cache_logical
        new_local = local_end + num_new - num_evicted
        if num_new > 0:
            global_end, local_end = cache_end, new_local

        if window_index in target_windows:
            # Cache-update call uses post-denoise indices
            cases.append((
                current_start, True, nfpb,
                global_end, local_end,
                f"W{window_index} cache-update"
            ))
        # Cache-update has num_new=0 (same current_start), no index change

    return cases


_TEST_CASES = _build_test_cases()


@pytest.fixture(scope="session")
def causal_self_attn_modules():
    """Create CPU ref and Neuron modules once per xdist worker.

    Module creation + NKI kernel compilation is expensive; shared across
    all e2e test cases via session scope.
    """
    dtype = torch.bfloat16
    dim, num_heads = 1536, 12
    head_dim = dim // num_heads

    cpu_module = RefCausalSelfAttention(dim, num_heads).to(dtype)
    neuron_module = CausalWanSelfAttention(dim, num_heads).to(dtype)

    # Share weights
    cpu_sd = cpu_module.state_dict()
    neuron_module.load_state_dict(cpu_sd, strict=False)
    neuron_module = neuron_module.to("neuron")

    freqs_cos, freqs_sin = _make_freqs(head_dim)
    return cpu_module, neuron_module, freqs_cos, freqs_sin


@pytest.mark.parametrize(
    "case_idx",
    range(len(_TEST_CASES)),
    ids=[c[-1] for c in _TEST_CASES],
)
def test_causal_self_attn_e2e(causal_self_attn_modules, case_idx):
    """E2E: RefCausalSelfAttention (CPU) vs CausalWanSelfAttention (Neuron).

    Each test is independent: creates its own random cache/buffers/input with
    the pre-computed cache index state, runs ONE forward call on both CPU and
    Neuron, and compares outputs. Compatible with pytest-xdist (-n auto).
    """
    cpu_module, neuron_module, freqs_cos, freqs_sin = causal_self_attn_modules
    current_start, updating_cache, nvf, ge, le, desc = _TEST_CASES[case_idx]

    dtype = torch.bfloat16
    dim, num_heads = 1536, 12
    head_dim = dim // num_heads
    H, W = 30, 52
    s = nvf * H * W  # tokens for this call
    kv_cache_alloc = 1560 * 24  # 37440
    buf_size = 1560 * 21  # 32760: max_attention_size
    buf_size = (buf_size + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE
    grid_sizes = (nvf, H, W)

    # Deterministic random state per test case
    torch.manual_seed(case_idx)

    # Random cache with specified index state
    cache_k = torch.randn(1, kv_cache_alloc, num_heads, head_dim, dtype=dtype)
    cache_v = torch.randn(1, kv_cache_alloc, num_heads, head_dim, dtype=dtype)

    cpu_kv = _make_kv_cache(kv_cache_alloc, num_heads, head_dim, dtype, "cpu")
    cpu_kv["k"].copy_(cache_k)
    cpu_kv["v"].copy_(cache_v)
    cpu_kv["global_end_index"] = ge
    cpu_kv["local_end_index"] = le

    neuron_kv = _make_kv_cache(kv_cache_alloc, num_heads, head_dim, dtype, "neuron")
    neuron_kv["k"].copy_(cache_k.to("neuron"))
    neuron_kv["v"].copy_(cache_v.to("neuron"))
    neuron_kv["global_end_index"] = ge
    neuron_kv["local_end_index"] = le

    cpu_bufs = _make_shared_buffers(buf_size, num_heads, head_dim, dtype, "cpu")
    neuron_bufs = _make_shared_buffers(buf_size, num_heads, head_dim, dtype, "neuron")

    x = torch.randn(1, s, dim, dtype=dtype)
    freqs_cos_n = freqs_cos.to("neuron")
    freqs_sin_n = freqs_sin.to("neuron")

    expected = cpu_module(
        x, grid_sizes=grid_sizes, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
        kv_cache=cpu_kv,
        current_start=current_start, updating_cache=updating_cache,
        num_valid_frames=nvf, shared_buffers=cpu_bufs)

    result = neuron_module(
        x.to("neuron"),
        grid_sizes=grid_sizes,
        freqs_cos=freqs_cos_n,
        freqs_sin=freqs_sin_n,
        kv_cache=neuron_kv,
        current_start=current_start,
        updating_cache=updating_cache,
        num_valid_frames=nvf,
        shared_buffers=neuron_bufs,
    )

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
