# Optimized inference entry point.
# Uses _opt modules for static shapes, simplified KV cache, etc.
import argparse
import torch
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
from torch.utils.data import DataLoader, SequentialSampler

import pipeline.rolling_forcing_inference_opt as rolling_forcing_module
from pipeline.rolling_forcing_inference_opt import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--rand_on_cpu", action="store_true",
                    help="Generate random tensors on CPU (for reproducibility with Neuron) and dump RNG state")
args = parser.parse_args()
if args.rand_on_cpu:
    rolling_forcing_module.RAND_ON_CPU = True
RAND_ON_CPU = rolling_forcing_module.RAND_ON_CPU
print(args)

device = torch.device("cuda")
set_seed(args.seed)
torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# Initialize pipeline
assert hasattr(config, 'denoising_step_list')
# Few-step inference
pipeline = CausalInferencePipeline(config, device=device)

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
    pipeline.generator.load_state_dict(state_dict_to_load)

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# Create dataset
dataset = TextDataset(prompt_path=args.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory
os.makedirs(args.output_folder, exist_ok=True)
if args.rand_on_cpu:
    rng_state_dir = os.path.join(args.output_folder, "cpu_rng_states")
    os.makedirs(rng_state_dir, exist_ok=True)

for i, batch_data in tqdm(enumerate(dataloader)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * args.num_samples
    else:
        prompts = [prompt] * args.num_samples
    initial_latent = None

    # Save CPU RNG state before noise generation for this sample
    if args.rand_on_cpu:
        rng_state_path = os.path.join(rng_state_dir, f"prompt_{i:03d}.pt")
        rng_state = torch.random.get_rng_state()
        torch.save(rng_state, rng_state_path)
        print(f"Saved CPU RNG state for prompt {i} to {rng_state_path}")

    noise_shape = [args.num_samples, args.num_output_frames, 16, 60, 104]
    if RAND_ON_CPU:
        sampled_noise = torch.randn(noise_shape, dtype=torch.bfloat16).to(device)
    else:
        sampled_noise = torch.randn(noise_shape, device=device, dtype=torch.bfloat16)

    # Generate 81 frames
    video, latents = pipeline.inference_rolling_forcing(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
    )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # All processes save their videos
            if args.save_with_index:
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
