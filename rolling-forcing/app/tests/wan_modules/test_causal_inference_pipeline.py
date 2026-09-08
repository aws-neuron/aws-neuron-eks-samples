"""End-to-end test for CausalInferencePipeline (1 layer, CPU vs Neuron)."""

import torch
from torch_neuronx.jit import jit

from models.causal_inference_pipeline import CausalInferencePipeline, add_noise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 1
DENOISING_STEP_LIST = [1000, 800, 600, 400, 200]
NUM_FRAME_PER_BLOCK = 3
NUM_FRAMES = 15  # 5 blocks * 3 frames
B, C, H, W = 1, 16, 60, 104
TEXT_LEN, TEXT_DIM = 512, 4096
SEED = 42


def test_add_noise():
    """add_noise: CPU vs Neuron via jit."""
    gen = torch.Generator().manual_seed(0)
    original_samples = torch.randn(3, 16, 60, 104, dtype=torch.bfloat16, generator=gen)
    noise = torch.randn(3, 16, 60, 104, dtype=torch.bfloat16, generator=gen)
    sigma = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float32).reshape(3, 1, 1, 1)

    # CPU reference
    cpu_out = add_noise(original_samples, noise, sigma)

    # Neuron via jit
    jit_add_noise = jit(add_noise)
    neuron_out = jit_add_noise(
        original_samples.to("neuron"), noise.to("neuron"), sigma.to("neuron")).cpu()

    torch.testing.assert_close(neuron_out, cpu_out, rtol=1e-2, atol=1e-2)


def test_causal_inference_pipeline_e2e():
    """CausalInferencePipeline CPU vs Neuron."""
    pipe = CausalInferencePipeline(
        denoising_step_list=DENOISING_STEP_LIST,
        num_frame_per_block=NUM_FRAME_PER_BLOCK,
        num_layers=NUM_LAYERS,
    )

    gen = torch.Generator().manual_seed(0)
    noise = torch.randn(B, NUM_FRAMES, C, H, W, dtype=torch.bfloat16, generator=gen)
    prompt_embeds = torch.randn(B, TEXT_LEN, TEXT_DIM, dtype=torch.bfloat16, generator=gen)
    conditional_dict = {"prompt_embeds": prompt_embeds}

    # --- CPU run ---
    torch.manual_seed(SEED)
    cpu_output = pipe.inference_rolling_forcing(noise, conditional_dict).clone()

    # --- Neuron run ---
    pipe.kv_cache_clean = None
    pipe.crossattn_cache = None
    pipe.generator.model = pipe.generator.model.to("neuron")

    torch.manual_seed(SEED)
    neuron_output = pipe.inference_rolling_forcing(
        noise.to("neuron"), {"prompt_embeds": prompt_embeds.to("neuron")}).cpu()

    assert cpu_output.shape == (B, NUM_FRAMES, C, H, W)
    torch.testing.assert_close(neuron_output, cpu_output, rtol=1e-1, atol=1e-1)
