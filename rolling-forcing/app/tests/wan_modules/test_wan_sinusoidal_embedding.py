import pytest
import torch

from torch_neuronx.jit import jit

from models.layers import sinusoidal_embedding_1d


def _sinusoidal_embedding_1d_f64(dim, position):
    """GPU reference: float64 precision (from wan/modules/model.py)."""
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


# Rolling-forcing pipeline calls sinusoidal_embedding_1d(freq_dim=256, t.flatten())
# where t is padded_timestep [B=1, 15] (float32). The 15 slots correspond to
# 5 blocks × nfpb=3 frames/block. Each block carries a denoising step from
# denoising_step_list=[999, 893, 786, 680, 573], repeated nfpb times.
#
# The pipeline produces 2*nds-1 = 9 unique timestep patterns across three
# window phases, plus cache-update patterns where the first nfpb slots are
# overwritten with context_noise=0.
FREQ_DIM = 256

# --- Denoising call patterns ---

# Steady-state: full 5-block window, denoising steps in reverse order
STEADY_STATE = [573, 573, 573, 680, 680, 680, 786, 786, 786, 893, 893, 893, 999, 999, 999]

# Ramp-up: window grows from 1 to 4 blocks, active blocks at tail, rest zero-padded
RAMP_UP_1BLK = [999, 999, 999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RAMP_UP_2BLK = [893, 893, 893, 999, 999, 999, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RAMP_UP_3BLK = [786, 786, 786, 893, 893, 893, 999, 999, 999, 0, 0, 0, 0, 0, 0]
RAMP_UP_4BLK = [680, 680, 680, 786, 786, 786, 893, 893, 893, 999, 999, 999, 0, 0, 0]

# Ramp-down: window shrinks from 4 to 1 blocks, active blocks at head, rest zero-padded
RAMP_DOWN_4BLK = [573, 573, 573, 680, 680, 680, 786, 786, 786, 893, 893, 893, 0, 0, 0]
RAMP_DOWN_3BLK = [573, 573, 573, 680, 680, 680, 786, 786, 786, 0, 0, 0, 0, 0, 0]
RAMP_DOWN_2BLK = [573, 573, 573, 680, 680, 680, 0, 0, 0, 0, 0, 0, 0, 0, 0]
RAMP_DOWN_1BLK = [573, 573, 573, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# --- Cache-update call patterns ---
# After each denoising call, the pipeline reruns the generator with dedicated
# 3-frame tensors: context_noise=0 for all nfpb=3 frames. No stale slots.

CACHE_UPDATE_AFTER_STEADY = [0, 0, 0]
CACHE_UPDATE_AFTER_RAMP_UP_2BLK = [0, 0, 0]
CACHE_UPDATE_AFTER_RAMP_DOWN_2BLK = [0, 0, 0]


@pytest.mark.parametrize("name,timesteps", [
    ("steady_state", STEADY_STATE),
    ("ramp_up_1blk", RAMP_UP_1BLK),
    ("ramp_up_2blk", RAMP_UP_2BLK),
    ("ramp_up_3blk", RAMP_UP_3BLK),
    ("ramp_up_4blk", RAMP_UP_4BLK),
    ("ramp_down_4blk", RAMP_DOWN_4BLK),
    ("ramp_down_3blk", RAMP_DOWN_3BLK),
    ("ramp_down_2blk", RAMP_DOWN_2BLK),
    ("ramp_down_1blk", RAMP_DOWN_1BLK),
    ("cache_update_after_steady", CACHE_UPDATE_AFTER_STEADY),
    ("cache_update_after_ramp_up_2blk", CACHE_UPDATE_AFTER_RAMP_UP_2BLK),
    ("cache_update_after_ramp_down_2blk", CACHE_UPDATE_AFTER_RAMP_DOWN_2BLK),
])
def test_sinusoidal_embedding_1d(name, timesteps):
    """sinusoidal_embedding_1d: [F] -> [F, 256] positional embeddings

    Tests all 9 denoising window patterns (1 steady + 4 ramp-up + 4 ramp-down,
    each 15 values) and 3 cache-update patterns (dedicated 3-frame tensors,
    all context_noise=0).
    """
    position = torch.tensor(timesteps, dtype=torch.float32)

    # GPU reference (float64 precision)
    expected_f64 = _sinusoidal_embedding_1d_f64(FREQ_DIM, position).float()

    # CPU (our float32 version)
    result_cpu = sinusoidal_embedding_1d(FREQ_DIM, position)
    torch.testing.assert_close(result_cpu, expected_f64, rtol=1e-4, atol=1e-4)

    # Neuron
    sinusoidal_neuron = jit(sinusoidal_embedding_1d)
    result_neuron = sinusoidal_neuron(FREQ_DIM, position.to("neuron"))
    torch.testing.assert_close(result_neuron.cpu(), expected_f64, rtol=1e-2, atol=1e-2)
