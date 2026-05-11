"""Neuron inference entry point.

Takes a pre-computed text embedding (.pt) and runs the rolling-forcing
pipeline on Neuron, saving raw latents to disk.
"""
import argparse
import os
from collections import OrderedDict

import torch
from omegaconf import OmegaConf

from models.causal_inference_pipeline import CausalInferencePipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True,
                    help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Path to the checkpoint file")
parser.add_argument("--embedding_path", type=str, required=True,
                    help="Path to .pt file containing prompt_embeds [1, 512, 4096]")
parser.add_argument("--output_path", type=str, required=True,
                    help="Path to save output latent .pt file")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of output frames (must be divisible by num_frame_per_block)")
parser.add_argument("--use_ema", action="store_true",
                    help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--rng_state_path", type=str, default=None,
                    help="Path to cpu_rng_states/ directory or a single .pt file")
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)

# Load config
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

assert hasattr(config, 'denoising_step_list')

# Build pipeline
pipe = CausalInferencePipeline(
    denoising_step_list=config.denoising_step_list,
    num_frame_per_block=getattr(config, "num_frame_per_block", 3),
    context_noise=getattr(config, "context_noise", 0.0),
    warp_denoising_step=getattr(config, "warp_denoising_step", True),
    model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
    timestep_shift=getattr(config, "timestep_shift", 5.0),
)

# Load checkpoint
if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if args.use_ema:
        state_dict_to_load = state_dict['generator_ema']
        def remove_fsdp_prefix(state_dict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if "_fsdp_wrapped_module." in key:
                    new_key = key.replace("_fsdp_wrapped_module.", "")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        state_dict_to_load = remove_fsdp_prefix(state_dict_to_load)
    else:
        state_dict_to_load = state_dict['generator']
    pipe.generator.load_state_dict(state_dict_to_load, strict=True)

torch.save(pipe.generator.state_dict(), "./generator_state_dict_neuron.pt")

# Move model to Neuron
pipe.generator.model = pipe.generator.model.to("neuron")

# Load pre-computed text embedding
prompt_embeds = torch.load(args.embedding_path, map_location="cpu").to(torch.bfloat16)
assert prompt_embeds.dim() == 3, f"Expected [B, 512, 4096], got {prompt_embeds.shape}"

# Restore CPU RNG state from GPU pipeline (if provided) so noise generation
# starts from the exact same RNG position, regardless of model init differences.
if args.rng_state_path:
    rng_path = args.rng_state_path
    if os.path.isdir(rng_path):
        # Derive per-sample filename from embedding_path basename
        # e.g. embedding_path="embeds/prompt_005.pt" -> "prompt_005.pt"
        sample_name = os.path.basename(args.embedding_path)
        rng_path = os.path.join(rng_path, sample_name)
    rng_state = torch.load(rng_path, map_location="cpu")
    torch.random.set_rng_state(rng_state)
    print(f"Restored CPU RNG state from {rng_path}")
print("CPU RNG state hash:", hash(torch.random.get_rng_state().numpy().tobytes()))

# Prepare inputs on Neuron
noise = torch.randn(
    1, args.num_output_frames, 16, 60, 104, dtype=torch.bfloat16
).to("neuron")
conditional_dict = {"prompt_embeds": prompt_embeds.to("neuron")}

# Run inference
latents = pipe.inference_rolling_forcing(noise, conditional_dict).cpu()

# Save output
os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
torch.save(latents, args.output_path)
print(f"Saved latents {latents.shape} {latents.dtype} to {args.output_path}")
