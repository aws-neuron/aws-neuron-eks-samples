"""Decode saved latents to video.

Usage:
    python decode_latents.py --input gpu_latents.pt --output output.mp4
    python decode_latents.py --input gpu_latents.pt --output output.mp4 --device cuda --fps 16
"""
import argparse

import torch
from einops import rearrange
from torchvision.io import write_video

from utils.wan_wrapper_opt import WanVAEWrapper


def main():
    parser = argparse.ArgumentParser(description="Decode latents to video")
    parser.add_argument("--input", type=str, required=True, help="Path to latents .pt file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--fps", type=int, default=16, help="Video FPS")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load latents: [B, num_frames, 16, 60, 104]
    latents = torch.load(args.input, map_location=device)
    print(f"Loaded latents: {latents.shape}, dtype={latents.dtype}")

    # Decode
    vae = WanVAEWrapper().to(device=device, dtype=latents.dtype)
    with torch.no_grad():
        video = vae.decode_to_pixel(latents, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)

    # [B, T, C, H, W] -> [B, T, H, W, C] uint8
    video = rearrange(video, 'b t c h w -> b t h w c')
    video = (255.0 * video).to(torch.uint8).cpu()

    for i in range(video.shape[0]):
        path = args.output if video.shape[0] == 1 else args.output.replace(".mp4", f"_{i}.mp4")
        write_video(path, video[i], fps=args.fps)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
