import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import convert_flow_pred_to_x0


# Real pipeline sigma values (from FlowMatchScheduler with shift=5.0, 1000 steps)
SIGMA_VALUES = [0.014793, 0.558011, 0.770115, 0.882698, 0.952494, 1.000000]
S_PAD, S1, S2, S3, S4, S5 = SIGMA_VALUES

# 10 sigma patterns: 9 pipeline patterns + 1 context_sigma pattern
# Each has 15 values (5 blocks x 3 frames/block)
NFP = 3  # num_frame_per_block

SIGMA_PATTERNS = [
    # Pattern 0: steady-state — all 5 blocks active
    [S1]*NFP + [S2]*NFP + [S3]*NFP + [S4]*NFP + [S5]*NFP,
    # Patterns 1-4: ramp-up (newest blocks first, rest padded)
    [S5]*NFP + [S_PAD]*12,
    [S4]*NFP + [S5]*NFP + [S_PAD]*9,
    [S3]*NFP + [S4]*NFP + [S5]*NFP + [S_PAD]*6,
    [S2]*NFP + [S3]*NFP + [S4]*NFP + [S5]*NFP + [S_PAD]*3,
    # Patterns 5-8: ramp-down (oldest blocks first, rest padded)
    [S1]*NFP + [S_PAD]*12,
    [S1]*NFP + [S2]*NFP + [S_PAD]*9,
    [S1]*NFP + [S2]*NFP + [S3]*NFP + [S_PAD]*6,
    [S1]*NFP + [S2]*NFP + [S3]*NFP + [S4]*NFP + [S_PAD]*3,
    # Pattern 9: context_sigma — cache-update call, dedicated 3-frame tensor
    [S_PAD]*3,
]

PATTERN_IDS = [
    "steady", "ramp-up-1", "ramp-up-2", "ramp-up-3", "ramp-up-4",
    "ramp-down-1", "ramp-down-2", "ramp-down-3", "ramp-down-4",
    "context-sigma",
]


def ref_convert_flow_pred_to_x0(flow_pred, xt, sigma_t):
    """CPU fp64 reference (matches GPU wan_wrapper_opt.py)."""
    flow_pred = flow_pred.double()
    xt = xt.double()
    sigma_t = sigma_t.double().reshape(-1, 1, 1, 1)
    return (xt - sigma_t * flow_pred).to(torch.bfloat16)


@pytest.mark.parametrize("pattern_idx", range(len(SIGMA_PATTERNS)), ids=PATTERN_IDS)
def test_convert_flow_pred_to_x0(pattern_idx):
    """Three-way: fp64 ref vs fp32 CPU vs Neuron, for each sigma pattern."""
    dtype = torch.bfloat16
    # Real pipeline shape: flattened to [F, C, H, W] where F varies by pattern
    C, H, W = 16, 60, 104
    F = len(SIGMA_PATTERNS[pattern_idx])

    flow_pred = torch.randn(F, C, H, W, dtype=dtype)
    xt = torch.randn(F, C, H, W, dtype=dtype)
    sigma_t = torch.tensor(SIGMA_PATTERNS[pattern_idx], dtype=torch.float32)

    # 1. CPU fp64 reference
    expected = ref_convert_flow_pred_to_x0(flow_pred, xt, sigma_t)

    # 2. CPU fp32 (our function)
    result_cpu = convert_flow_pred_to_x0(flow_pred, xt, sigma_t)
    torch.testing.assert_close(result_cpu, expected, rtol=1e-2, atol=1e-2)

    # 3. Neuron
    convert_neuron = jit(convert_flow_pred_to_x0)
    result_neuron = convert_neuron(
        flow_pred.to("neuron"), xt.to("neuron"), sigma_t.to("neuron"))
    torch.testing.assert_close(result_neuron.cpu(), expected, rtol=1e-2, atol=1e-2)
