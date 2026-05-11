"""Standalone VAE decode on Neuron.

Step 2 of the pipeline: Takes latents from DiT and decodes to video frames.

Usage:
    python run_vae_decode.py \
        --latent_path latents.pt \
        --output_path output.mp4 \
        --device neuron
"""
import argparse
import os
import sys

import torch
from einops import rearrange

# Add gpu/RollingForcing to path to access the VAE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu", "RollingForcing"))

from wan.modules.vae import WanVAE


class WanVAEWrapper(torch.nn.Module):
    """Wrapper for WanVAE that handles device placement."""
    
    def __init__(self, vae_pth="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        self.vae_pth = vae_pth
        self.model = None
        self.mean = None
        self.std = None
        self.scale = None
        
    def _init_model(self, device, dtype):
        """Lazy initialization on first use."""
        if self.model is not None:
            return
            
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        
        # Keep mean/std on CPU for now, move during decode
        self.mean = torch.tensor(mean, dtype=dtype)
        self.std = torch.tensor(std, dtype=dtype)
        
        # Load the VAE model
        from wan.modules.vae import _video_vae
        self.model = _video_vae(
            pretrained_path=self.vae_pth,
            z_dim=self.z_dim,
        ).eval().requires_grad_(False)
        
    def to(self, device=None, dtype=None):
        """Override to handle lazy initialization."""
        if self.model is None:
            self._init_model(device, dtype or torch.bfloat16)
        if device is not None:
            self.model = self.model.to(device)
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)
            self.mean = self.mean.to(dtype)
            self.std = self.std.to(dtype)
        return self
    
    def decode_to_pixel(self, latents, use_cache=False):
        """Decode latents to pixel space.
        
        Args:
            latents: [B, T, C, H, W] latent tensor
            use_cache: Whether to use cached decoding (not supported on Neuron)
            
        Returns:
            [B, T, 3, H*8, W*8] video tensor in [-1, 1] range
        """
        device = latents.device
        dtype = latents.dtype
        
        # Ensure model is initialized and on correct device
        self._init_model(device, dtype)
        
        # Build scale on the correct device
        scale = [self.mean.to(device), (1.0 / self.std).to(device)]
        
        # Convert from [B, T, C, H, W] to [B, C, T, H, W] for VAE
        latents = rearrange(latents, 'b t c h w -> b c t h w')
        
        # Decode on Neuron (skip clamp here - Neuron compiler bug)
        with torch.no_grad():
            video = self.model.decode(latents, scale)
        
        # Convert back to [B, T, C, H, W]
        video = rearrange(video, 'b c t h w -> b t c h w')
        return video


def main():
    parser = argparse.ArgumentParser(description="VAE decode on Neuron")
    parser.add_argument("--latent_path", type=str, required=True,
                        help="Path to latents .pt file from DiT")
    parser.add_argument("--output_path", type=str, default="output.mp4",
                        help="Output video path")
    parser.add_argument("--vae_path", type=str, 
                        default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                        help="Path to VAE checkpoint")
    parser.add_argument("--device", type=str, default="neuron",
                        help="Device (neuron, cuda, or cpu)")
    parser.add_argument("--fps", type=int, default=16,
                        help="Video FPS")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save individual frames instead of video")
    parser.add_argument("--no_compile", action="store_true",
                        help="Skip torch.compile (run eager mode)")
    args = parser.parse_args()

    print(f"[run_vae_decode] Starting VAE decode")
    print(f"  Latent: {args.latent_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Device: {args.device}")

    # Load latents: [B, num_frames, 16, 60, 104]
    latents = torch.load(args.latent_path, map_location="cpu")
    print(f"[run_vae_decode] Loaded latents: {latents.shape}, dtype={latents.dtype}")
    
    # Convert to bfloat16 if needed
    if latents.dtype != torch.bfloat16:
        latents = latents.to(torch.bfloat16)

    # Initialize and load VAE
    print(f"[run_vae_decode] Loading VAE from {args.vae_path}...")
    vae = WanVAEWrapper(vae_pth=args.vae_path)
    vae = vae.to(device=args.device, dtype=latents.dtype)
    
    # Move latents to device
    latents = latents.to(args.device)

    # Compile VAE decode for Neuron
    if args.device == "neuron" and not args.no_compile:
        import time
        print("[run_vae_decode] Compiling VAE with torch.compile(backend='neuron')...")
        
        # Compile the inner model's decode function
        # fullgraph=False allows graph breaks at non-tensor ops (e.g., string comparisons in cache)
        vae.model.decode = torch.compile(
            vae.model.decode,
            backend="neuron",
            fullgraph=False,
            dynamic=False
        )
        
        # Warmup / compilation pass
        print("[run_vae_decode] Warmup pass (NEFF compilation)...")
        start = time.time()
        with torch.no_grad():
            _ = vae.decode_to_pixel(latents, use_cache=False)
        print(f"[run_vae_decode] Compilation done in {time.time() - start:.2f}s")

    # Decode
    print("[run_vae_decode] Decoding latents to video...")
    with torch.no_grad():
        video = vae.decode_to_pixel(latents, use_cache=False)
    
    # Normalize to [0, 1] range (without clamp on Neuron - compiler bug)
    video = video * 0.5 + 0.5
    print(f"[run_vae_decode] Decoded video: {video.shape}")

    # Move to CPU for saving, then clamp (avoids Neuron compiler bug)
    video = video.cpu().clamp(0, 1)

    # Save output
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    
    if args.save_frames:
        # Save individual frames
        from torchvision.utils import save_image
        frame_dir = args.output_path.replace(".mp4", "_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for b in range(video.shape[0]):
            for t in range(video.shape[1]):
                frame_path = os.path.join(frame_dir, f"batch{b:02d}_frame{t:04d}.png")
                save_image(video[b, t], frame_path)
        print(f"[run_vae_decode] Saved frames to {frame_dir}")
    else:
        # Save as video - try multiple methods
        video_out = rearrange(video, 'b t c h w -> b t h w c')
        video_out = (255.0 * video_out).to(torch.uint8).numpy()
        
        saved = False
        for i in range(video_out.shape[0]):
            path = args.output_path if video_out.shape[0] == 1 else args.output_path.replace(".mp4", f"_{i}.mp4")
            
            # Try imageio first
            try:
                import imageio
                writer = imageio.get_writer(path, fps=args.fps, codec='libx264')
                for frame in video_out[i]:
                    writer.append_data(frame)
                writer.close()
                print(f"[run_vae_decode] Saved {path} (imageio)")
                saved = True
            except ImportError:
                pass
            
            # Fallback: save individual PNG frames
            if not saved:
                import numpy as np
                from PIL import Image
                frame_dir = path.replace(".mp4", "_frames")
                os.makedirs(frame_dir, exist_ok=True)
                for t, frame in enumerate(video_out[i]):
                    img = Image.fromarray(frame)
                    img.save(os.path.join(frame_dir, f"frame_{t:04d}.png"))
                print(f"[run_vae_decode] Saved PNG frames to {frame_dir}")
                saved = True
    
    print("[run_vae_decode] Done!")


if __name__ == "__main__":
    main()
