"""Standalone DiT inference on Neuron.

Step 1 of the pipeline: Takes text embeddings and runs the CausalWanModel
diffusion denoising loop. Outputs raw latents to disk.

Usage:
    python run_dit_inference.py \
        --config_path configs/rolling_forcing_dmd.yaml \
        --embedding_path embeds/prompt.pt \
        --output_path latents.pt \
        --num_output_frames 21
"""
import argparse
import os
from collections import OrderedDict

import torch
from omegaconf import OmegaConf

from models.causal_inference_pipeline import CausalInferencePipeline


def main():
    parser = argparse.ArgumentParser(description="DiT inference on Neuron")
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
    parser.add_argument("--device", type=str, default="neuron",
                        help="Device to run on (neuron or cpu)")
    args = parser.parse_args()
    
    print(f"[run_dit_inference] Starting DiT inference")
    print(f"  Config: {args.config_path}")
    print(f"  Embedding: {args.embedding_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Device: {args.device}")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # Load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    assert hasattr(config, 'denoising_step_list')

    # Get spatial dimensions from config (default: 60x104)
    if hasattr(config, 'image_or_video_shape'):
        # [B, F, C, H, W]
        latent_height = config.image_or_video_shape[3]
        latent_width = config.image_or_video_shape[4]
    else:
        latent_height = 60
        latent_width = 104
    
    # Calculate frame_seq_length = (H * W) / (patch_h * patch_w) with patch=(2,2)
    frame_seq_length = (latent_height * latent_width) // 4
    print(f"[run_dit_inference] Spatial: {latent_height}x{latent_width}, frame_seq_length={frame_seq_length}")

    # Build pipeline
    print("[run_dit_inference] Building CausalInferencePipeline...")
    pipe = CausalInferencePipeline(
        denoising_step_list=config.denoising_step_list,
        num_frame_per_block=getattr(config, "num_frame_per_block", 3),
        context_noise=getattr(config, "context_noise", 0.0),
        warp_denoising_step=getattr(config, "warp_denoising_step", True),
        frame_seq_length=frame_seq_length,
        model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
        timestep_shift=getattr(config, "timestep_shift", 5.0),
    )

    # Load checkpoint
    if args.checkpoint_path:
        print(f"[run_dit_inference] Loading checkpoint: {args.checkpoint_path}")
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

    # Move model to device
    print(f"[run_dit_inference] Moving model to {args.device}...")
    pipe.generator.model = pipe.generator.model.to(args.device)

    # Load pre-computed text embedding
    prompt_embeds = torch.load(args.embedding_path, map_location="cpu").to(torch.bfloat16)
    assert prompt_embeds.dim() == 3, f"Expected [B, 512, 4096], got {prompt_embeds.shape}"
    print(f"[run_dit_inference] Loaded embeddings: {prompt_embeds.shape}")

    # Restore CPU RNG state if provided
    if args.rng_state_path:
        rng_path = args.rng_state_path
        if os.path.isdir(rng_path):
            sample_name = os.path.basename(args.embedding_path)
            rng_path = os.path.join(rng_path, sample_name)
        rng_state = torch.load(rng_path, map_location="cpu")
        torch.random.set_rng_state(rng_state)
        print(f"[run_dit_inference] Restored CPU RNG state from {rng_path}")

    # Prepare inputs on device
    print(f"[run_dit_inference] Generating noise and running inference...")
    noise = torch.randn(
        1, args.num_output_frames, 16, latent_height, latent_width, dtype=torch.bfloat16
    ).to(args.device)
    conditional_dict = {"prompt_embeds": prompt_embeds.to(args.device)}

    # Run inference
    latents = pipe.inference_rolling_forcing(noise, conditional_dict).cpu()

    # Save output
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(latents, args.output_path)
    print(f"[run_dit_inference] Saved latents {latents.shape} {latents.dtype} to {args.output_path}")
    print("[run_dit_inference] Done!")


if __name__ == "__main__":
    main()
