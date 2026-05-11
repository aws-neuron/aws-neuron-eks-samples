"""End-to-end test for Neuron CausalWanModel (1 layer).

Three-way comparison: RefCausalWanModel (pure CPU baseline) vs
CausalWanModel on CPU vs CausalWanModel on Neuron.

Uses the same spatial dimensions as test_wan_attn_block (H=30, W=52 after
patching) so that RoPE NEFFs are reused from existing compilations:
- x: [1, 16, 15, 60, 104] -> after patch_size=(1,2,2) -> grid_sizes=(15,30,52)
- t: [1, 15] timesteps (padded, real denoising step patterns)
- context: [1, 512, 4096]
- num_valid_frames marks how many of the 15 frames are valid (3-15)

Reuses _build_test_cases from test_wan_attn_block which simulates a real
42-block pipeline with correct cache index evolution (global_end, local_end).
"""
import copy

import pytest
import torch
import torch.nn as nn

from models.causal_model import CausalWanModel
from models.layers import sinusoidal_embedding_1d, rope_params, unpatchify, ATTN_SEQLEN_MULTIPLE

from tests.wan_modules.test_wan_attn_block import (
    RefCausalWanAttentionBlock,
    _build_test_cases as _build_attn_test_cases,
)
from tests.wan_modules.test_wan_causal_head import RefCausalHead


# ---------------------------------------------------------------------------
# RefCausalWanModel — pure CPU baseline (mirrors GPU causal_model_opt.py)
# ---------------------------------------------------------------------------

class RefCausalWanModel(nn.Module):
    """Pure CPU baseline matching GPU causal_model_opt.py.

    Uses nn.Conv3d, nn.GELU, nn.SiLU, and RefCausalWanAttentionBlock.
    """

    def __init__(self, patch_size, text_len, in_dim, dim, ffn_dim,
                 freq_dim, text_dim, out_dim, num_heads, num_layers, eps=1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.freq_dim = freq_dim
        self.dim = dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            RefCausalWanAttentionBlock(dim, ffn_dim, num_heads, eps)
            for _ in range(num_layers)
        ])

        self.head = RefCausalHead(dim, out_dim, patch_size, eps)

        # RoPE: same as GPU (float64 computation, stored as cos/sin)
        d = dim // num_heads
        cos_0, sin_0 = rope_params(1024, d - 4 * (d // 6))
        cos_1, sin_1 = rope_params(1024, 2 * (d // 6))
        cos_2, sin_2 = rope_params(1024, 2 * (d // 6))
        self.freqs_cos = torch.cat([cos_0, cos_1, cos_2], dim=1)
        self.freqs_sin = torch.cat([sin_0, sin_1, sin_2], dim=1)

    def forward(self, x, t, context,
                updating_cache=False, kv_cache=None, crossattn_cache=None,
                current_start=0, cache_start=0,
                num_valid_frames=None, shared_buffers=None):
        assert x.shape[0] == 1

        x = self.patch_embedding(x)
        grid_sizes = tuple(int(d) for d in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        assert context.size(1) == self.text_len
        context = self.text_embedding(context)

        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos,
            freqs_sin=self.freqs_sin,
            context=context,
            context_lens=None,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
        )

        for block_index, block in enumerate(self.blocks):
            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
            })
            x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = x.flatten(1, 2)
        return unpatchify(x, self.out_dim, self.patch_size, grid_sizes).unsqueeze(0)


# ---------------------------------------------------------------------------
# Constants — spatial dims match test_wan_attn_block (H=30, W=52 after patching)
# ---------------------------------------------------------------------------
DIM = 1536
FFN_DIM = 6144
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS
IN_DIM = 16
OUT_DIM = 16
FREQ_DIM = 256
TEXT_DIM = 4096
TEXT_LEN = 512
PATCH_SIZE = (1, 2, 2)

# H_IN=60, W_IN=104 -> after patch_size=(1,2,2) -> H=30, W=52 (matches attn_block test)
H_IN, W_IN = 60, 104
H, W = H_IN // PATCH_SIZE[1], W_IN // PATCH_SIZE[2]  # 30, 52
FRAME_LENGTH = H * W  # 1560

MAX_FRAMES = 15
NFPB = 3
NDS = 5
BLOCK_LENGTH = NFPB * FRAME_LENGTH  # 4680

KV_CACHE_ALLOC = FRAME_LENGTH * 24  # 37440
BUF_SIZE = FRAME_LENGTH * 21  # 32760: max_attention_size
BUF_SIZE = (BUF_SIZE + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE  # 32768

DENOISING_STEPS = [999, 893, 786, 680, 573]

# Steady-state: [573]*3 + [680]*3 + [786]*3 + [893]*3 + [999]*3
STEADY_PATTERN = []
for _step in reversed(DENOISING_STEPS):
    STEADY_PATTERN.extend([_step] * NFPB)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_cache(num_heads, head_dim, dtype, device):
    return {
        "k": torch.zeros(1, KV_CACHE_ALLOC, num_heads, head_dim, dtype=dtype, device=device),
        "v": torch.zeros(1, KV_CACHE_ALLOC, num_heads, head_dim, dtype=dtype, device=device),
        "global_end_index": 0,
        "local_end_index": 0,
    }


def _make_shared_buffers(num_heads, head_dim, dtype, device):
    return (
        torch.zeros(1, BUF_SIZE, num_heads, head_dim, dtype=dtype, device=device),
        torch.zeros(1, BUF_SIZE, num_heads, head_dim, dtype=dtype, device=device),
    )


def _build_timestep(current_start, nvf, updating_cache):
    """Build timestep pattern matching the pipeline for a given window state.

    Uses the same logic as rolling_forcing_inference_opt._build_timestep_patterns:
    - Ramp-up (current_start==0, nvf<15): tail of steady pattern
    - Steady-state (nvf==15): full steady pattern
    - Ramp-down (current_start>0, nvf<15): head of steady pattern
    - Cache-update: context_noise=0 for nfpb frames
    """
    if updating_cache:
        return [0] * nvf
    num_blocks = nvf // NFPB
    if nvf == MAX_FRAMES:
        return list(STEADY_PATTERN)
    elif current_start == 0:
        cnf = num_blocks * NFPB
        return STEADY_PATTERN[-cnf:] + [0] * (MAX_FRAMES - cnf)
    else:
        cnf = num_blocks * NFPB
        return STEADY_PATTERN[:cnf] + [0] * (MAX_FRAMES - cnf)


def _make_models(dtype=torch.bfloat16):
    """Create RefCausalWanModel, CausalWanModel (CPU), CausalWanModel (Neuron)."""
    model_args = dict(
        patch_size=PATCH_SIZE, text_len=TEXT_LEN,
        in_dim=IN_DIM, dim=DIM, ffn_dim=FFN_DIM, freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM, out_dim=OUT_DIM, num_heads=NUM_HEADS, num_layers=1,
    )

    cpu_ref = RefCausalWanModel(**model_args).to(dtype).eval()
    model_cpu = CausalWanModel(model_type='t2v', **model_args).to(dtype).eval()

    # Load ref weights into model_cpu (strict=False: ref has no jit wrappers)
    sd = cpu_ref.state_dict()
    model_cpu.load_state_dict(sd, strict=False)

    model_neuron = copy.deepcopy(model_cpu).to("neuron")

    return cpu_ref, model_cpu, model_neuron


# ---------------------------------------------------------------------------
# Test cases: reuse attn_block's _build_test_cases (simulates 42-block pipeline)
# Each case: (current_start, updating_cache, nvf, ge, le, desc)
# We add the timestep pattern for each case.
# ---------------------------------------------------------------------------

_ATTN_TEST_CASES = _build_attn_test_cases()

_TEST_CASES = []
for current_start, updating_cache, nvf, ge, le, desc in _ATTN_TEST_CASES:
    t_list = _build_timestep(current_start, nvf, updating_cache)
    _TEST_CASES.append((current_start, updating_cache, nvf, ge, le, t_list, desc))


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_idx",
    range(len(_TEST_CASES)),
    ids=[c[-1] for c in _TEST_CASES],
)
def test_causal_model_e2e(case_idx):
    """RefCausalWanModel (CPU) vs CausalWanModel (CPU) vs CausalWanModel (Neuron)."""
    cpu_ref, model_cpu, model_neuron = _make_models()
    current_start, updating_cache, nvf, ge, le, t_list, desc = _TEST_CASES[case_idx]

    dtype = torch.bfloat16
    t = torch.tensor([t_list], dtype=torch.float32)

    torch.manual_seed(case_idx)

    x = torch.randn(1, IN_DIM, nvf, H_IN, W_IN, dtype=dtype)
    context = torch.randn(1, TEXT_LEN, TEXT_DIM, dtype=dtype)
    cache_k = torch.randn(1, KV_CACHE_ALLOC, NUM_HEADS, HEAD_DIM, dtype=dtype)
    cache_v = torch.randn(1, KV_CACHE_ALLOC, NUM_HEADS, HEAD_DIM, dtype=dtype)

    def _setup_kv(device):
        kv = {0: _make_kv_cache(NUM_HEADS, HEAD_DIM, dtype, device)}
        kv[0]["k"].copy_(cache_k if device == "cpu" else cache_k.to(device))
        kv[0]["v"].copy_(cache_v if device == "cpu" else cache_v.to(device))
        kv[0]["global_end_index"] = ge
        kv[0]["local_end_index"] = le
        return kv

    def _run(model, device):
        kv = _setup_kv(device)
        crossattn = {0: {"is_init": False, "k": None, "v": None}}
        bufs = _make_shared_buffers(NUM_HEADS, HEAD_DIM, dtype, device)
        x_d = x if device == "cpu" else x.to(device)
        ctx_d = context if device == "cpu" else context.to(device)
        return model(
            x=x_d, t=t, context=ctx_d,
            updating_cache=updating_cache,
            kv_cache=kv, crossattn_cache=crossattn,
            current_start=current_start, cache_start=current_start,
            num_valid_frames=nvf, shared_buffers=bufs)

    with torch.no_grad():
        expected = _run(cpu_ref, "cpu")
        result_cpu = _run(model_cpu, "cpu")
        result_neuron = _run(model_neuron, "neuron")

    assert expected.shape == (1, OUT_DIM, nvf, H_IN, W_IN)

    # Ref CPU vs Model CPU
    torch.testing.assert_close(result_cpu, expected, rtol=2e-2, atol=2e-2)
    # Model CPU vs Model Neuron
    torch.testing.assert_close(result_cpu, result_neuron.cpu(), rtol=2e-2, atol=2e-2)
