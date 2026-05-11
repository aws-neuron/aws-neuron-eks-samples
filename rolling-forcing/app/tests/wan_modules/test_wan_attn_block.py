import copy
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import WanLayerNorm, rope_params, ATTN_SEQLEN_MULTIPLE
from models.layers import CausalWanAttentionBlock

from tests.wan_modules.test_wan_self_attn import RefCausalSelfAttention
from tests.wan_modules.test_wan_cross_attn import RefCrossAttention


class RefCausalWanAttentionBlock(nn.Module):
    """CPU reference: mirrors GPU CausalWanAttentionBlock with plain PyTorch ops."""

    def __init__(self, dim, ffn_dim, num_heads, eps=1e-6):
        super().__init__()
        self.dim = dim

        # norms (WanLayerNorm is @jit but works on CPU)
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)

        # sub-modules
        self.self_attn = RefCausalSelfAttention(dim, num_heads, eps=eps)
        self.cross_attn = RefCrossAttention(dim, num_heads, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, grid_sizes, freqs_cos, freqs_sin,
                context, context_lens,
                updating_cache=False, kv_cache=None, crossattn_cache=None,
                current_start=0, cache_start=None,
                num_valid_frames=None, shared_buffers=None):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid_sizes, freqs_cos, freqs_sin,
            kv_cache, current_start, cache_start,
            updating_cache=updating_cache, num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers)
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention
        x = x + self.cross_attn(
            self.norm3(x), context, context_lens,
            crossattn_cache=crossattn_cache)

        # ffn
        y = self.ffn(
            (self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)

        return x


# ---------------------------------------------------------------------------
# Helpers (copied from test_wan_self_attn — same test infrastructure)
# ---------------------------------------------------------------------------

def _make_freqs(head_dim):
    d = head_dim
    cos_f, sin_f = rope_params(1024, d - 4 * (d // 6))
    cos_h, sin_h = rope_params(1024, 2 * (d // 6))
    cos_w, sin_w = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_f, cos_h, cos_w], dim=1), torch.cat([sin_f, sin_h, sin_w], dim=1)


def _make_kv_cache(alloc_size, num_heads, head_dim, dtype, device):
    return {
        "k": torch.zeros(1, alloc_size, num_heads, head_dim, dtype=dtype, device=device),
        "v": torch.zeros(1, alloc_size, num_heads, head_dim, dtype=dtype, device=device),
        "global_end_index": 0,
        "local_end_index": 0,
    }


def _make_shared_buffers(buf_size, num_heads, head_dim, dtype, device):
    return (
        torch.zeros(1, buf_size, num_heads, head_dim, dtype=dtype, device=device),
        torch.zeros(1, buf_size, num_heads, head_dim, dtype=dtype, device=device),
    )


def _build_test_cases(nds=5, nfpb=3, num_blocks=42):
    """Build independent test cases with pre-computed cache index states."""
    frame_length = 1560
    block_length = nfpb * frame_length
    kv_cache_logical = 24 * frame_length
    cases = []
    window_num = num_blocks + nds - 1

    target_windows = set(range(13))
    target_windows |= set(range(window_num - nds + 1, window_num))

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

        num_new = cache_end - global_end
        num_evicted = 0
        if num_new > 0 and num_new + local_end > kv_cache_logical:
            num_evicted = num_new + local_end - kv_cache_logical
        new_local = local_end + num_new - num_evicted
        if num_new > 0:
            global_end, local_end = cache_end, new_local

        if window_index in target_windows:
            cases.append((
                current_start, True, nfpb,
                global_end, local_end,
                f"W{window_index} cache-update"
            ))

    return cases


_TEST_CASES = _build_test_cases()


# ---------------------------------------------------------------------------
# Session fixture and parametrized e2e test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def causal_attn_block_modules():
    """Create CPU ref and Neuron attention block modules once per xdist worker."""
    dtype = torch.bfloat16
    dim, ffn_dim, num_heads = 1536, 6144, 12
    head_dim = dim // num_heads

    cpu_ref = RefCausalWanAttentionBlock(dim, ffn_dim, num_heads).to(dtype)
    module_cpu = CausalWanAttentionBlock(
        't2v_cross_attn', dim, ffn_dim, num_heads,
        cross_attn_norm=True).to(dtype)

    sd = cpu_ref.state_dict()
    module_cpu.load_state_dict(sd, strict=False)
    module_neuron = copy.deepcopy(module_cpu).to("neuron")

    freqs_cos, freqs_sin = _make_freqs(head_dim)
    return cpu_ref, module_cpu, module_neuron, freqs_cos, freqs_sin


@pytest.mark.parametrize(
    "case_idx",
    range(len(_TEST_CASES)),
    ids=[c[-1] for c in _TEST_CASES],
)
def test_causal_attn_block_e2e(causal_attn_block_modules, case_idx):
    """E2E: RefCausalWanAttentionBlock (CPU) vs CausalWanAttentionBlock (Neuron).

    Each test is independent: creates its own random cache/buffers/input with
    the pre-computed cache index state, runs ONE forward call on both CPU and
    Neuron, and compares outputs. Compatible with pytest-xdist (-n auto).
    """
    cpu_ref, module_cpu, module_neuron, freqs_cos, freqs_sin = causal_attn_block_modules
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
    text_len = 512

    # Deterministic random state per test case
    torch.manual_seed(case_idx)

    # Random KV cache with specified index state
    cache_k = torch.randn(1, kv_cache_alloc, num_heads, head_dim, dtype=dtype)
    cache_v = torch.randn(1, kv_cache_alloc, num_heads, head_dim, dtype=dtype)

    cpu_kv = _make_kv_cache(kv_cache_alloc, num_heads, head_dim, dtype, "cpu")
    cpu_kv["k"].copy_(cache_k)
    cpu_kv["v"].copy_(cache_v)
    cpu_kv["global_end_index"] = ge
    cpu_kv["local_end_index"] = le

    cpu_bufs = _make_shared_buffers(buf_size, num_heads, head_dim, dtype, "cpu")

    # Input tensor
    x = torch.randn(1, s, dim, dtype=dtype)

    # Modulation embeddings: [B, nvf, 6, C]
    e = torch.randn(1, nvf, 6, dim, dtype=dtype)

    # Cross-attention context: [B, text_len, C]
    context = torch.randn(1, text_len, dim, dtype=dtype)

    # Cross-attention cache (fresh per test — is_init=False)
    # Use separate caches so each module computes its own K/V from context
    cpu_crossattn_cache = {"is_init": False, "k": None, "v": None}

    # Reference model CPU execution
    expected = cpu_ref(
        x, e, grid_sizes=grid_sizes, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
        context=context, context_lens=None,
        updating_cache=updating_cache, kv_cache=cpu_kv,
        crossattn_cache=cpu_crossattn_cache,
        current_start=current_start, num_valid_frames=nvf,
        shared_buffers=cpu_bufs)

    # Model CPU execution
    cpu_crossattn_cache = {"is_init": False, "k": None, "v": None}  # reset
    result_cpu = module_cpu(
        x, e, grid_sizes=grid_sizes, freqs_cos=freqs_cos, freqs_sin=freqs_sin,
        context=context, context_lens=None,
        updating_cache=updating_cache, kv_cache=cpu_kv,
        crossattn_cache=cpu_crossattn_cache,
        current_start=current_start, num_valid_frames=nvf,
        shared_buffers=cpu_bufs)

    # CPU execution against reference
    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # Model Neuron execution
    neuron_kv = _make_kv_cache(kv_cache_alloc, num_heads, head_dim, dtype, "neuron")
    neuron_kv["k"].copy_(cache_k.to("neuron"))
    neuron_kv["v"].copy_(cache_v.to("neuron"))
    neuron_kv["global_end_index"] = ge
    neuron_kv["local_end_index"] = le
    neuron_bufs = _make_shared_buffers(buf_size, num_heads, head_dim, dtype, "neuron")

    # RoPE frequencies
    freqs_cos_n = freqs_cos.to("neuron")
    freqs_sin_n = freqs_sin.to("neuron")

    neuron_crossattn_cache = {"is_init": False, "k": None, "v": None}
    x_neuron, e_neuron, context_neuron = x.to("neuron"), e.to("neuron"), context.to("neuron")
    t_neuron_start = time.perf_counter()
    result_neuron = module_neuron(
        x_neuron,
        e_neuron,
        grid_sizes=grid_sizes,
        freqs_cos=freqs_cos_n,
        freqs_sin=freqs_sin_n,
        context=context_neuron,
        context_lens=None,
        updating_cache=updating_cache,
        kv_cache=neuron_kv,
        crossattn_cache=neuron_crossattn_cache,
        current_start=current_start,
        num_valid_frames=nvf,
        shared_buffers=neuron_bufs,
    )
    t_neuron_end = time.perf_counter()
    print(f"\n[WALL] Neuron forward (1st call): {(t_neuron_end - t_neuron_start)*1000:.1f} ms")

    # Second call to measure cached execution time (no trace/compile/loading)
    neuron_kv2 = _make_kv_cache(kv_cache_alloc, num_heads, head_dim, dtype, "neuron")
    neuron_kv2["k"].copy_(cache_k.to("neuron"))
    neuron_kv2["v"].copy_(cache_v.to("neuron"))
    neuron_kv2["global_end_index"] = ge
    neuron_kv2["local_end_index"] = le
    neuron_bufs2 = _make_shared_buffers(buf_size, num_heads, head_dim, dtype, "neuron")
    neuron_crossattn_cache2 = {"is_init": False, "k": None, "v": None}
    t_neuron2_start = time.perf_counter()
    module_neuron(
        x_neuron,
        e_neuron,
        grid_sizes=grid_sizes,
        freqs_cos=freqs_cos_n,
        freqs_sin=freqs_sin_n,
        context=context_neuron,
        context_lens=None,
        updating_cache=updating_cache,
        kv_cache=neuron_kv2,
        crossattn_cache=neuron_crossattn_cache2,
        current_start=current_start,
        num_valid_frames=nvf,
        shared_buffers=neuron_bufs2,
    )
    t_neuron2_end = time.perf_counter()
    print(f"[WALL] Neuron forward (2nd call): {(t_neuron2_end - t_neuron2_start)*1000:.1f} ms")

    # CPU execution against Neuron execution
    torch.testing.assert_close(result_cpu, result_neuron.cpu(), rtol=1e-2, atol=1e-1)
