"""Test WanDiffusionWrapper end-to-end with real checkpoint weights.

Three-way comparison: RefWanDiffusionWrapper (pure CPU, fp64 convert)
vs WanDiffusionWrapper (CPU) vs WanDiffusionWrapper (Neuron).

Two cases matching real pipeline calls from rolling_forcing_inference_opt.py:
1. Steady-state denoise (updating_cache=False, nvf=15)
2. Steady-state cache-update (updating_cache=True, nvf=3)

Input construction mirrors CausalInferencePipeline exactly:
- WanDiffusionWrapper(is_causal=True) with default model_kwargs
- KV cache: list of dicts, [B, 51480, 12, 128], Python int indices
- Crossattn cache: list of dicts, pre-allocated [B, 512, 12, 128]
- Shared buffers: tuple of (buffer_k, buffer_v), [B, 51480, 12, 128]
- current_start = current_start_frame * frame_seq_length (int)
- No cache_start passed (pipeline doesn't use it)
"""
import copy

import pytest
import torch
import torch.nn as nn

from models.causal_model_wrapper import WanDiffusionWrapper
from models.layers import ATTN_SEQLEN_MULTIPLE

from tests.wan_modules.test_wan_causal_model import RefCausalWanModel
from tests.wan_modules.test_wan_convert_flow_pred import ref_convert_flow_pred_to_x0


# ---------------------------------------------------------------------------
# RefWanDiffusionWrapper — pure CPU baseline (mirrors GPU wan_wrapper_opt.py)
# ---------------------------------------------------------------------------

class RefWanDiffusionWrapper(nn.Module):
    """Pure CPU baseline matching GPU wan_wrapper_opt.py.

    Uses RefCausalWanModel (nn.Conv3d, nn.GELU, nn.SiLU) and fp64 convert.
    """

    def __init__(self, model_name="Wan2.1-T2V-1.3B"):
        super().__init__()
        import json
        from safetensors.torch import load_file

        pretrained_path = f"wan_models/{model_name}/"
        with open(pretrained_path + "config.json") as f:
            cfg = json.load(f)
        self.model = RefCausalWanModel(
            patch_size=PATCH_SIZE, text_len=cfg["text_len"], in_dim=cfg["in_dim"],
            dim=cfg["dim"], ffn_dim=cfg["ffn_dim"], freq_dim=cfg["freq_dim"],
            text_dim=TEXT_DIM, out_dim=cfg["out_dim"], num_heads=cfg["num_heads"],
            num_layers=NUM_LAYERS, eps=cfg["eps"],
        ).to(torch.bfloat16).eval()
        sd = load_file(pretrained_path + "diffusion_pytorch_model.safetensors")
        self.model.load_state_dict(sd, strict=False)

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                kv_cache=None, crossattn_cache=None,
                current_start=None,
                updating_cache=False, num_valid_frames=None,
                shared_buffers=None, sigma=None):
        prompt_embeds = conditional_dict["prompt_embeds"]

        assert kv_cache is not None
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep, context=prompt_embeds,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache,
            current_start=current_start, cache_start=current_start,
            updating_cache=updating_cache, num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = ref_convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            sigma.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0


# ---------------------------------------------------------------------------
# Constants — match real pipeline (rolling_forcing_inference_opt.py)
# ---------------------------------------------------------------------------
IN_DIM = 16
OUT_DIM = 16
TEXT_DIM = 4096
TEXT_LEN = 512
PATCH_SIZE = (1, 2, 2)
DIM = 1536
FFN_DIM = 8960
FREQ_DIM = 256
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS  # 128
NUM_LAYERS = 1
EPS = 1e-6

H_IN, W_IN = 60, 104
NFPB = 3
NDS = 5
MAX_FRAMES = NDS * NFPB  # 15
FRAME_SEQ_LENGTH = (H_IN // PATCH_SIZE[1]) * (W_IN // PATCH_SIZE[2])  # 1560

# Match _initialize_kv_cache exactly
KV_CACHE_ALLOC = FRAME_SEQ_LENGTH * 24   # 37440
BUF_SIZE = FRAME_SEQ_LENGTH * 21  # 32760: max_attention_size
BUF_SIZE = (BUF_SIZE + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE  # 32768

# Timestep patterns from _build_timestep_patterns (steady-state = pattern 0)
DENOISING_STEPS = [999, 893, 786, 680, 573]
STEADY_TIMESTEP = []
for _step in reversed(DENOISING_STEPS):
    STEADY_TIMESTEP.extend([float(_step)] * NFPB)

# Sigma patterns from _build_sigma_patterns (precomputed from FlowMatchScheduler shift=8.0)
# These are the actual sigmas the pipeline passes, computed via _timestep_to_sigma
SIGMA_VALUES = [0.014793, 0.558011, 0.770115, 0.882698, 0.952494]
S1, S2, S3, S4, S5 = SIGMA_VALUES
STEADY_SIGMA = [S1]*NFPB + [S2]*NFPB + [S3]*NFPB + [S4]*NFPB + [S5]*NFPB

# context_noise=0 -> context_sigma via _timestep_to_sigma(0)
CONTEXT_SIGMA = S1  # smallest sigma for timestep ~0

# Steady-state cache indices: after pipeline has filled the cache
# In steady-state (window_index >= nds), current_start_frame = (window_index - nds + 1) * nfpb
# For window_index=12 (first steady): start_block=8, current_start_frame=24
# Cache has been filled with frames 0..26 (end_block=12, cache_end=(24+3)*1560)
STEADY_CURRENT_START_FRAME = 24
STEADY_CURRENT_START = STEADY_CURRENT_START_FRAME * FRAME_SEQ_LENGTH  # 37440
STEADY_GE = (STEADY_CURRENT_START_FRAME + NFPB) * FRAME_SEQ_LENGTH   # 42120
STEADY_LE = FRAME_SEQ_LENGTH * 24  # 37440 (logical max, eviction has happened)


# ---------------------------------------------------------------------------
# Helpers — mirror _initialize_kv_cache / _initialize_crossattn_cache exactly
# ---------------------------------------------------------------------------

def _make_kv_cache(dtype, device):
    """List of dicts, matching GPU pipeline's _initialize_kv_cache."""
    kv = []
    for _ in range(NUM_LAYERS):
        kv.append({
            "k": torch.zeros(1, KV_CACHE_ALLOC, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
            "v": torch.zeros(1, KV_CACHE_ALLOC, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
            "global_end_index": 0,
            "local_end_index": 0,
        })
    return kv


def _make_crossattn_cache(dtype, device):
    """List of dicts, matching GPU pipeline's _initialize_crossattn_cache."""
    cache = []
    for _ in range(NUM_LAYERS):
        cache.append({
            "k": torch.zeros(1, TEXT_LEN, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
            "v": torch.zeros(1, TEXT_LEN, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
            "is_init": False,
        })
    return cache


def _make_shared_buffers(dtype, device):
    """Tuple of (buffer_k, buffer_v), matching GPU pipeline."""
    return (
        torch.zeros(1, BUF_SIZE, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
        torch.zeros(1, BUF_SIZE, NUM_HEADS, HEAD_DIM, dtype=dtype, device=device),
    )


def _make_wrappers():
    """Create RefWanDiffusionWrapper (CPU), WanDiffusionWrapper (CPU), WanDiffusionWrapper (Neuron)."""
    ref_wrapper = RefWanDiffusionWrapper().eval()
    wrapper_cpu = WanDiffusionWrapper(is_causal=True, num_layers=NUM_LAYERS)
    wrapper_neuron = copy.deepcopy(wrapper_cpu).to("neuron")
    return ref_wrapper, wrapper_cpu, wrapper_neuron


# ---------------------------------------------------------------------------
# Test cases — match real pipeline call sites
# ---------------------------------------------------------------------------

_TEST_CASES = [
    # (updating_cache, nvf, current_start, ge, le, timestep_list, sigma_list, desc)
    #
    # Case 1: Denoise call (line 243-253 in rolling_forcing_inference_opt.py)
    # padded_input = noisy_cache[:, current_start_frame : current_start_frame + max_frames]
    # padded_timestep = timestep_patterns[0] (steady-state)
    # padded_sigma = sigma_patterns[0]
    # current_start = current_start_frame * frame_seq_length
    (False, MAX_FRAMES, STEADY_CURRENT_START, STEADY_GE, STEADY_LE,
     STEADY_TIMESTEP, STEADY_SIGMA, "steady-denoise"),
    #
    # Case 2: Cache-update call — dedicated 3-frame tensors (no padding)
    # cache_input = denoised_pred[:, :nfpb]  (3 frames)
    # cache_timestep = [context_noise] * nfpb (3 values)
    # cache_sigma = [context_sigma] * nfpb   (3 values)
    # updating_cache=True, num_valid_frames=nfpb
    # current_start same as denoise call (same window)
    (True, NFPB, STEADY_CURRENT_START, STEADY_GE, STEADY_LE,
     [0.0] * NFPB,
     [CONTEXT_SIGMA] * NFPB,
     "steady-cache-update"),
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _run_wrapper(wrapper, x, context, t, sigma, ge, le, current_start,
                 updating_cache, nvf, device):
    dtype = torch.bfloat16
    kv = _make_kv_cache(dtype, device)
    for i in range(NUM_LAYERS):
        kv[i]["global_end_index"] = ge
        kv[i]["local_end_index"] = le
    crossattn = _make_crossattn_cache(dtype, device)
    bufs = _make_shared_buffers(dtype, device)
    x_d = x if device == "cpu" else x.to(device)
    ctx_d = context if device == "cpu" else context.to(device)
    sigma_d = sigma if device == "cpu" else sigma.to(device)
    return wrapper(
        noisy_image_or_video=x_d,
        conditional_dict={"prompt_embeds": ctx_d},
        timestep=t, kv_cache=kv, crossattn_cache=crossattn,
        current_start=current_start,
        updating_cache=updating_cache, num_valid_frames=nvf,
        shared_buffers=bufs, sigma=sigma_d,
    )


@pytest.mark.parametrize(
    "case_idx",
    range(len(_TEST_CASES)),
    ids=[c[-1] for c in _TEST_CASES],
)
def test_wan_wrapper_e2e(case_idx):
    """RefWanDiffusionWrapper (CPU) vs WanDiffusionWrapper (CPU) vs WanDiffusionWrapper (Neuron)."""
    updating_cache, nvf, current_start, ge, le, t_list, sigma_list, desc = _TEST_CASES[case_idx]

    torch.manual_seed(case_idx)

    # Wrapper input: [B, nvf, C, H, W] — 15 frames for denoise, 3 for cache-update
    x = torch.randn(1, nvf, IN_DIM, H_IN, W_IN, dtype=torch.bfloat16)
    # Context: [B, 512, 4096] — from text encoder
    context = torch.randn(1, TEXT_LEN, TEXT_DIM, dtype=torch.bfloat16)
    # Timestep: [B, nvf]
    t = torch.tensor([t_list], dtype=torch.float32)
    # Sigma: [B, nvf]
    sigma = torch.tensor([sigma_list], dtype=torch.float32)

    run_kwargs = dict(
        x=x, context=context, t=t, sigma=sigma,
        ge=ge, le=le, current_start=current_start,
        updating_cache=updating_cache, nvf=nvf,
    )

    ref_wrapper, wrapper_cpu, wrapper_neuron = _make_wrappers()

    with torch.no_grad():
        ref_flow, ref_x0 = _run_wrapper(ref_wrapper, device="cpu", **run_kwargs)
        cpu_flow, cpu_x0 = _run_wrapper(wrapper_cpu, device="cpu", **run_kwargs)
        neuron_flow, neuron_x0 = _run_wrapper(wrapper_neuron, device="neuron", **run_kwargs)

    assert ref_flow.shape == (1, nvf, OUT_DIM, H_IN, W_IN)

    # Ref CPU vs Model CPU
    torch.testing.assert_close(cpu_flow, ref_flow, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(cpu_x0, ref_x0, rtol=1e-1, atol=1e-1)

    # Model CPU vs Model Neuron
    torch.testing.assert_close(neuron_flow.cpu(), cpu_flow, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(neuron_x0.cpu(), cpu_x0, rtol=1e-1, atol=1e-1)
