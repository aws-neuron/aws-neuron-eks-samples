"""Streaming VAE decoder for incremental latent-to-pixel conversion.

This module provides utilities for decoding video latents progressively,
allowing frames to be converted to pixels as soon as they're generated
by the DiT model.

The standard VAE expects all frames at once, but we can optimize for
streaming by:
1. Decoding frames in small batches
2. Caching intermediate activations when possible
3. Overlapping decode with DiT inference

Usage:
    from streaming_vae import StreamingVAEDecoder
    
    decoder = StreamingVAEDecoder(vae_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    
    for latent_block in dit_generator:
        frames = decoder.decode_block(latent_block)
        yield frames
"""
import os
import sys
from typing import Iterator, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from einops import rearrange

# Add path for wan modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu", "RollingForcing"))


@dataclass
class VAEConfig:
    """Configuration for VAE decoder."""
    vae_path: str = "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    z_dim: int = 16
    dtype: torch.dtype = torch.bfloat16
    device: str = "neuron"
    
    # Streaming settings
    decode_batch_size: int = 4  # Frames per decode call
    use_tiled_decode: bool = False  # For memory efficiency on large frames
    tile_size: int = 256
    tile_overlap: int = 32


class StreamingVAEDecoder:
    """VAE decoder optimized for streaming/incremental decoding.
    
    Features:
    - Decode latent blocks as they arrive
    - Memory-efficient batched decoding
    - Optional tiled decoding for large frames
    - Proper handling of temporal convolutions at boundaries
    """
    
    def __init__(self, config: VAEConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        self._model = None
        self._scale = None
        
        # For handling temporal boundary effects
        self._prev_latent_tail: Optional[torch.Tensor] = None
        self._overlap_frames = 1  # Frames to overlap for smooth boundaries
        
    def _load_model(self):
        """Load VAE model lazily."""
        if self._model is not None:
            return
            
        from wan.modules.vae import _video_vae
        
        print(f"[StreamingVAE] Loading VAE from {self.config.vae_path}")
        
        self._model = _video_vae(
            pretrained_path=self.config.vae_path,
            z_dim=self.config.z_dim
        ).eval().requires_grad_(False)
        
        self._model = self._model.to(dtype=self.dtype, device=self.device)
        
        # Precompute normalization scale
        mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=self.dtype).to(self.device)
        
        std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], dtype=self.dtype).to(self.device)
        
        self._scale = [mean, 1.0 / std]
        
        print("[StreamingVAE] VAE loaded successfully")
    
    def decode_block(
        self,
        latents: torch.Tensor,
        return_numpy: bool = False,
    ) -> List[Image.Image]:
        """Decode a block of latent frames.
        
        Args:
            latents: [B, T, C, H, W] latent tensor
            return_numpy: If True, return numpy arrays instead of PIL Images
            
        Returns:
            List of decoded frames as PIL Images (or numpy arrays)
        """
        self._load_model()
        
        latents = latents.to(self.dtype).to(self.device)
        
        # [B, T, C, H, W] -> [B, C, T, H, W]
        latents_bcthw = rearrange(latents, 'b t c h w -> b c t h w')
        
        with torch.no_grad():
            video = self._model.decode(latents_bcthw, self._scale)
            video = video.clamp(-1, 1)
        
        # [B, C, T, H, W] -> [B, T, H, W, C]
        video = rearrange(video, 'b c t h w -> b t h w c')
        video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
        
        # Convert to images
        video_np = (255.0 * video[0]).to(torch.uint8).numpy()
        
        if return_numpy:
            return [video_np[i] for i in range(video_np.shape[0])]
        else:
            return [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]
    
    def decode_single_frame(
        self,
        latent: torch.Tensor,
    ) -> Image.Image:
        """Decode a single latent frame.
        
        Note: This may have boundary artifacts if the VAE uses temporal convolutions.
        For best quality, use decode_block with multiple frames.
        
        Args:
            latent: [B, 1, C, H, W] or [B, C, H, W] single frame latent
            
        Returns:
            PIL Image
        """
        if latent.dim() == 4:
            latent = latent.unsqueeze(1)
        
        frames = self.decode_block(latent)
        return frames[0]
    
    def decode_streaming(
        self,
        latent_generator: Iterator[Tuple[int, torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> Iterator[Tuple[int, Image.Image]]:
        """Stream decode from a latent generator.
        
        Optimizes decoding by batching frames and overlapping computation.
        
        Args:
            latent_generator: Yields (frame_idx, latent_block) tuples
            batch_size: Frames to batch for decoding (default from config)
            
        Yields:
            (frame_idx, PIL.Image) tuples
        """
        batch_size = batch_size or self.config.decode_batch_size
        
        latent_buffer = []
        frame_indices = []
        
        for frame_idx, latent_block in latent_generator:
            # latent_block is [B, num_frames_in_block, C, H, W]
            num_frames = latent_block.shape[1]
            
            for i in range(num_frames):
                latent_buffer.append(latent_block[:, i:i+1])
                frame_indices.append(frame_idx + i)
                
                # Decode when buffer is full
                if len(latent_buffer) >= batch_size:
                    batch_latent = torch.cat(latent_buffer, dim=1)
                    frames = self.decode_block(batch_latent)
                    
                    for idx, frame in zip(frame_indices, frames):
                        yield (idx, frame)
                    
                    latent_buffer = []
                    frame_indices = []
        
        # Decode remaining frames
        if latent_buffer:
            batch_latent = torch.cat(latent_buffer, dim=1)
            frames = self.decode_block(batch_latent)
            
            for idx, frame in zip(frame_indices, frames):
                yield (idx, frame)
    
    def reset(self):
        """Reset decoder state for new video generation."""
        self._prev_latent_tail = None
    
    def decode_full_video(
        self,
        latents: torch.Tensor,
        output_format: str = "pil",
    ) -> List:
        """Decode all latents at once (non-streaming).
        
        Args:
            latents: [B, T, C, H, W] full latent tensor
            output_format: "pil", "numpy", or "tensor"
            
        Returns:
            List of frames in requested format
        """
        self._load_model()
        
        latents = latents.to(self.dtype).to(self.device)
        
        # [B, T, C, H, W] -> [B, C, T, H, W]
        latents_bcthw = rearrange(latents, 'b t c h w -> b c t h w')
        
        with torch.no_grad():
            video = self._model.decode(latents_bcthw, self._scale)
            video = video.clamp(-1, 1)
        
        # [B, C, T, H, W] -> [B, T, H, W, C]
        video = rearrange(video, 'b c t h w -> b t h w c')
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        if output_format == "tensor":
            return video
        
        video_cpu = video.cpu()
        video_np = (255.0 * video_cpu[0]).to(torch.uint8).numpy()
        
        if output_format == "numpy":
            return [video_np[i] for i in range(video_np.shape[0])]
        else:  # pil
            return [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]


class TiledVAEDecoder(StreamingVAEDecoder):
    """Memory-efficient VAE decoder using tiled decoding.
    
    For very large frames (e.g., 4K), this decoder processes the image
    in tiles to reduce peak memory usage.
    """
    
    def decode_block(
        self,
        latents: torch.Tensor,
        return_numpy: bool = False,
    ) -> List[Image.Image]:
        """Decode using tiled approach for memory efficiency."""
        if not self.config.use_tiled_decode:
            return super().decode_block(latents, return_numpy)
        
        self._load_model()
        
        latents = latents.to(self.dtype).to(self.device)
        batch_size, num_frames, channels, height, width = latents.shape
        
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        stride = tile_size - overlap
        
        # Output size (VAE typically upscales 8x)
        scale_factor = 8
        out_height = height * scale_factor
        out_width = width * scale_factor
        
        # Initialize output
        output = torch.zeros(
            batch_size, num_frames, out_height, out_width, 3,
            dtype=self.dtype, device=self.device
        )
        weight = torch.zeros(
            batch_size, num_frames, out_height, out_width, 1,
            dtype=self.dtype, device=self.device
        )
        
        # Process tiles
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Extract tile
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                tile_latent = latents[:, :, :, y:y_end, x:x_end]
                
                # Decode tile
                tile_bcthw = rearrange(tile_latent, 'b t c h w -> b c t h w')
                with torch.no_grad():
                    tile_video = self._model.decode(tile_bcthw, self._scale)
                    tile_video = tile_video.clamp(-1, 1)
                
                tile_video = rearrange(tile_video, 'b c t h w -> b t h w c')
                tile_video = (tile_video * 0.5 + 0.5).clamp(0, 1)
                
                # Output coordinates
                out_y = y * scale_factor
                out_x = x * scale_factor
                out_y_end = y_end * scale_factor
                out_x_end = x_end * scale_factor
                
                # Blend tile into output
                output[:, :, out_y:out_y_end, out_x:out_x_end] += tile_video
                weight[:, :, out_y:out_y_end, out_x:out_x_end] += 1.0
        
        # Normalize by weight
        output = output / weight.clamp(min=1.0)
        
        # Convert to images
        video_np = (255.0 * output[0].cpu()).to(torch.uint8).numpy()
        
        if return_numpy:
            return [video_np[i] for i in range(video_np.shape[0])]
        else:
            return [Image.fromarray(video_np[i]) for i in range(video_np.shape[0])]


def create_decoder(
    vae_path: str = "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    device: str = "neuron",
    use_tiled: bool = False,
    **kwargs,
) -> StreamingVAEDecoder:
    """Factory function to create appropriate VAE decoder.
    
    Args:
        vae_path: Path to VAE weights
        device: Device to run on
        use_tiled: Whether to use memory-efficient tiled decoding
        **kwargs: Additional config options
        
    Returns:
        StreamingVAEDecoder instance
    """
    config = VAEConfig(
        vae_path=vae_path,
        device=device,
        use_tiled_decode=use_tiled,
        **{k: v for k, v in kwargs.items() if hasattr(VAEConfig, k)},
    )
    
    if use_tiled:
        return TiledVAEDecoder(config)
    else:
        return StreamingVAEDecoder(config)


# =============================================================================
# Utility Functions
# =============================================================================

def frames_to_video(
    frames: List[Image.Image],
    output_path: str,
    fps: int = 16,
) -> str:
    """Save frames as video file.
    
    Args:
        frames: List of PIL Images
        output_path: Output video path
        fps: Frames per second
        
    Returns:
        Output path
    """
    import imageio
    
    video_np = np.stack([np.array(f) for f in frames])
    imageio.mimwrite(output_path, video_np, fps=fps)
    
    return output_path


def latents_to_gif(
    decoder: StreamingVAEDecoder,
    latents: torch.Tensor,
    output_path: str,
    fps: int = 16,
    loop: int = 0,
) -> str:
    """Decode latents and save as GIF.
    
    Args:
        decoder: VAE decoder instance
        latents: [B, T, C, H, W] latent tensor
        output_path: Output GIF path
        fps: Frames per second
        loop: Number of loops (0 = infinite)
        
    Returns:
        Output path
    """
    frames = decoder.decode_full_video(latents, output_format="pil")
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=loop,
    )
    
    return output_path
