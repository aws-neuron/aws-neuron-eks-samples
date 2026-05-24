"""VAE decode on Neuron (single device/core).

Step 3 of the pipeline: Decode latents to video on neuron:2.

Usage:
    python run_vae_decode.py \
        --latent_path latents.pt \
        --output_path output.mp4 \
        --device neuron:2
"""
import argparse
import os

import torch
from torchvision.io import write_video
from einops import rearrange

from wan.modules.vae import _video_vae


def main():
    parser = argparse.ArgumentParser(description="VAE decode on Neuron")
    parser.add_argument("--latent_path", type=str, required=True,
                        help="Path to latents .pt file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output video path (.mp4)")
    parser.add_argument("--vae_path", type=str, 
                        default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                        help="Path to VAE checkpoint")
    parser.add_argument("--device", type=str, default="neuron:2",
                        help="Device (neuron:0, neuron:1, etc.)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    args = parser.parse_args()

    print(f"[run_vae_decode] Starting VAE decode on {args.device}")
    print(f"  Latent: {args.latent_path}")
    print(f"  Output: {args.output_path}")

    device = torch.device(args.device)

    # VAE normalization stats
    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ], dtype=torch.float32)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ], dtype=torch.float32)

    # Load VAE (eager mode - no torch.compile due to grid_sample issues)
    print("[run_vae_decode] Loading VAE decoder (eager mode)...")
    vae = _video_vae(
        pretrained_path=args.vae_path,
        z_dim=16,
    ).eval().requires_grad_(False)

    # Move to Neuron device
    print(f"[run_vae_decode] Moving VAE to {args.device}...")
    vae = vae.to(device=device, dtype=torch.bfloat16)

    # Load latents
    latents = torch.load(args.latent_path, map_location="cpu")
    print(f"[run_vae_decode] Loaded latents: {latents.shape}")

    # Decode
    print("[run_vae_decode] Decoding latents to video...")
    
    # from [batch_size, num_frames, num_channels, height, width]
    # to [batch_size, num_channels, num_frames, height, width]
    zs = latents.permute(0, 2, 1, 3, 4).to(device=device, dtype=torch.bfloat16)

    scale = [mean.to(device=device, dtype=torch.bfloat16),
             1.0 / std.to(device=device, dtype=torch.bfloat16)]

    with torch.no_grad():
        output = []
        for u in zs:
            # Decode on Neuron - skip intermediate clamp (VAE output is bounded)
            # The final clamp to [0,1] after normalization handles any outliers
            decoded = vae.decode(u.unsqueeze(0), scale).float().squeeze(0)
            output.append(decoded)
        output = torch.stack(output, dim=0)

    # from [batch_size, num_channels, num_frames, height, width]
    # to [batch_size, num_frames, num_channels, height, width]
    video = output.permute(0, 2, 1, 3, 4)

    # Move to CPU FIRST, then do all post-processing (avoids Neuron JIT compilation bugs)
    video_out = rearrange(video, "b t c h w -> b t h w c").cpu()
    video_out = video_out * 0.5 + 0.5  # normalize on CPU
    video_out = video_out.clamp(0, 1)  # clamp on CPU
    video_out = (255.0 * video_out).to(torch.uint8)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    write_video(args.output_path, video_out[0], fps=args.fps)
    print(f"[run_vae_decode] Saved video to {args.output_path}")
    print("[run_vae_decode] Done!")


if __name__ == "__main__":
    main()
