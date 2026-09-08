"""Combined inference: T5 + DiT + VAE in a single process.

All models run on the same Neuron device. Models are loaded/unloaded
sequentially to fit in memory.

Usage:
    python run_inference_combined.py \
        --prompt "A cat walking on the beach" \
        --config configs/rolling_forcing_dmd_small.yaml \
        --checkpoint checkpoints/rolling_forcing_dmd.pt \
        --output output.mp4 \
        --use_ema
"""
import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange

# Add gpu/RollingForcing to path for wan modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu", "RollingForcing"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--model_path", type=str, default="wan_models/Wan2.1-T2V-1.3B")
parser.add_argument("--vae_path", type=str, default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
parser.add_argument("--num_frames", type=int, default=21)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fps", type=int, default=16)
args = parser.parse_args()

print("="*60)
print("Combined Inference: T5 + DiT + VAE (single process, single device)")
print("="*60)
print(f"Prompt: {args.prompt}")
print(f"Config: {args.config}")
print("="*60)

torch.manual_seed(args.seed)
torch.set_grad_enabled(False)

# Load config
config = OmegaConf.load(args.config)
default_path = "configs/default_config.yaml"
if os.path.exists(default_path):
    config = OmegaConf.merge(OmegaConf.load(default_path), config)

# Get spatial dimensions from config
if hasattr(config, 'image_or_video_shape'):
    latent_h = config.image_or_video_shape[3]
    latent_w = config.image_or_video_shape[4]
else:
    latent_h = getattr(config, "spatial_h", 30)
    latent_w = getattr(config, "spatial_w", 52)

# frame_seq_length = (H * W) / patch_area, patch=(2,2)
frame_seq_length = (latent_h * latent_w) // 4
print(f"Spatial: {latent_h}x{latent_w}, frame_seq_length={frame_seq_length}")

# ============================================================
# Step 1: T5 Encoding
# ============================================================
print(f"\n[Step 1] T5 Encoding")

from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl

t5_start = time.time()

# Load T5
print("  Loading UMT5-XXL encoder...")
text_encoder = umt5_xxl(
    encoder_only=True, return_tokenizer=False,
    dtype=torch.bfloat16, device=torch.device('cpu')
).eval().requires_grad_(False)

weights_path = os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth")
text_encoder.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=False))
text_encoder = text_encoder.to(device="neuron")

tokenizer_path = os.path.join(args.model_path, "google/umt5-xxl/")
tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=512, clean='whitespace')

# Compile T5
print("  Compiling T5...")
text_encoder.forward = torch.compile(text_encoder.forward, backend="neuron", fullgraph=True, dynamic=False)

# Warmup
dummy_ids, dummy_mask = tokenizer(["warmup"], return_mask=True, add_special_tokens=True)
with torch.no_grad():
    _ = text_encoder(dummy_ids.to("neuron"), dummy_mask.to("neuron"))
torch.neuron.synchronize()

# Encode prompt
ids, mask = tokenizer([args.prompt], return_mask=True, add_special_tokens=True)
ids, mask = ids.to("neuron"), mask.to("neuron")
seq_len = mask.gt(0).sum(dim=1).long()

with torch.no_grad():
    prompt_embeds = text_encoder(ids, mask)
torch.neuron.synchronize()

# Zero padding
prompt_embeds = prompt_embeds.cpu()
prompt_embeds[0, seq_len[0].cpu():] = 0.0

print(f"  T5 output: {prompt_embeds.shape} ({time.time()-t5_start:.1f}s)")

# ============================================================
# Step 2: DiT Inference
# ============================================================
print(f"\n[Step 2] DiT Inference")

from models.causal_inference_pipeline import CausalInferencePipeline

dit_start = time.time()

pipe = CausalInferencePipeline(
    denoising_step_list=config.denoising_step_list,
    num_frame_per_block=getattr(config, "num_frame_per_block", 1),
    context_noise=getattr(config, "context_noise", 0.0),
    warp_denoising_step=getattr(config, "warp_denoising_step", True),
    model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
    timestep_shift=getattr(config, "timestep_shift", 5.0),
    frame_seq_length=frame_seq_length,
)

if args.checkpoint:
    print(f"  Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if args.use_ema:
        sd = state_dict['generator_ema']
        sd = OrderedDict((k.replace("_fsdp_wrapped_module.", ""), v) for k, v in sd.items())
    else:
        sd = state_dict['generator']
    pipe.generator.load_state_dict(sd, strict=True)

print("  Moving DiT to neuron...")
pipe.generator.model = pipe.generator.model.to("neuron")

# Prepare inputs
noise = torch.randn(1, args.num_frames, 16, latent_h, latent_w, dtype=torch.bfloat16).to("neuron")
conditional_dict = {"prompt_embeds": prompt_embeds.to(torch.bfloat16).to("neuron")}

print("  Running inference...")
latents = pipe.inference_rolling_forcing(noise, conditional_dict).cpu()
print(f"  DiT output: {latents.shape} ({time.time()-dit_start:.1f}s)")

# ============================================================
# Step 3: VAE Decode
# ============================================================
print(f"\n[Step 3] VAE Decode")

from wan.modules.vae import _video_vae

vae_start = time.time()

# VAE normalization
mean = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
], dtype=torch.bfloat16)
std = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
], dtype=torch.bfloat16)

print("  Loading VAE...")
vae_model = _video_vae(pretrained_path=args.vae_path, z_dim=16).eval().requires_grad_(False)
vae_model = vae_model.to(dtype=torch.bfloat16, device="neuron")

# Decode: [B, T, C, H, W] -> [B, C, T, H, W]
# Do rearrange BEFORE moving to device (avoids non-contiguous tensor on Neuron)
latents_bcthw = rearrange(latents, 'b t c h w -> b c t h w')
latents_bcthw = latents_bcthw.to(torch.bfloat16).to("neuron")
mean = mean.to("neuron")
std = std.to("neuron")
scale = [mean, 1.0 / std]

print("  Decoding...")
with torch.no_grad():
    video = vae_model.decode(latents_bcthw, scale)

# [B, C, T, H, W] -> [B, T, C, H, W]
video = rearrange(video, 'b c t h w -> b t c h w')
video = video.cpu()  # Move to CPU first (clamp fails on Neuron)
video = (video * 0.5 + 0.5).clamp(0, 1)  # clamp covers [-1,1] -> [0,1]
print(f"  VAE output: {video.shape} ({time.time()-vae_start:.1f}s)")

# ============================================================
# Step 4: Save Video
# ============================================================
print(f"\n[Step 4] Saving video...")

from torchvision.io import write_video

video_out = rearrange(video, 'b t c h w -> b t h w c')
video_out = (255.0 * video_out[0]).to(torch.uint8)

os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

write_video(args.output, video_out, fps=args.fps)
print(f"  Saved: {args.output}")

print("\n" + "="*60)
print("Done!")
print("="*60)
