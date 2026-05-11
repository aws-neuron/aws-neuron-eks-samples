"""Streaming inference pipeline for progressive video generation.

This module wraps CausalInferencePipeline to yield frames/chunks progressively
instead of waiting for full video generation.

Usage:
    from streaming_pipeline import StreamingInferencePipeline
    
    pipe = StreamingInferencePipeline(config_path="configs/rolling_forcing_dmd_small.yaml")
    
    # Frame-by-frame streaming
    for frame in pipe.generate_streaming(prompt="A cat walking"):
        display(frame)  # PIL Image
    
    # Chunk-based streaming
    for chunk_path in pipe.generate_chunked(prompt="A cat walking", chunk_size=6):
        play_video(chunk_path)
"""
import os
import sys
import time
import tempfile
from typing import Iterator, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange

# Add gpu/RollingForcing to path for wan modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu", "RollingForcing"))


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    config_path: str
    checkpoint_path: Optional[str] = None
    model_path: str = "wan_models/Wan2.1-T2V-1.3B"
    vae_path: str = "wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
    num_frames: int = 21
    use_ema: bool = True
    seed: int = 0
    fps: int = 16
    device: str = "neuron"  # legacy fallback — used when per-component devices are not set
    # Per-component device placement (2 NDs with lnc=2 → 4 logical devices)
    # neuron:0 → ND0 NC0+NC1, neuron:1 → ND0 NC2+NC3
    # neuron:2 → ND1 NC0+NC1, neuron:3 → ND1 NC2+NC3
    dit_device: str = "neuron:0"    # DiT transformer on ND0 (NC0+NC1)
    t5_device: str = "neuron:2"     # T5 text encoder on ND1 (NC0+NC1)
    vae_device: str = "neuron:3"    # VAE decoder on ND1 (NC2+NC3)


class StreamingInferencePipeline:
    """Pipeline that yields video frames/chunks progressively during generation."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.device = config.device  # legacy fallback
        self.dit_device = config.dit_device
        self.t5_device = config.t5_device
        self.vae_device = config.vae_device
        
        torch.manual_seed(config.seed)
        torch.set_grad_enabled(False)
        
        # Load model config
        self.model_config = OmegaConf.load(config.config_path)
        default_path = "configs/default_config.yaml"
        if os.path.exists(default_path):
            self.model_config = OmegaConf.merge(
                OmegaConf.load(default_path), self.model_config
            )
        
        # Get spatial dimensions
        if hasattr(self.model_config, 'image_or_video_shape'):
            self.latent_h = self.model_config.image_or_video_shape[3]
            self.latent_w = self.model_config.image_or_video_shape[4]
        else:
            self.latent_h = getattr(self.model_config, "spatial_h", 30)
            self.latent_w = getattr(self.model_config, "spatial_w", 52)
        
        self.frame_seq_length = (self.latent_h * self.latent_w) // 4
        
        # Models will be loaded lazily
        self._text_encoder = None
        self._tokenizer = None
        self._dit_pipeline = None
        self._vae_model = None
        self._vae_scale = None
        
    def _load_t5(self):
        """Load T5 text encoder lazily."""
        if self._text_encoder is not None:
            return
            
        from wan.modules.tokenizers import HuggingfaceTokenizer
        from wan.modules.t5 import umt5_xxl
        
        print("[StreamingPipeline] Loading T5 encoder...")
        
        self._text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.bfloat16, device=torch.device('cpu')
        ).eval().requires_grad_(False)
        
        weights_path = os.path.join(
            self.config.model_path, "models_t5_umt5-xxl-enc-bf16.pth"
        )
        self._text_encoder.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False)
        )
        self._text_encoder = self._text_encoder.to(device=self.t5_device)
        print(f"[StreamingPipeline] T5 encoder placed on {self.t5_device}")
        
        tokenizer_path = os.path.join(self.config.model_path, "google/umt5-xxl/")
        self._tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace'
        )
        
        # Compile T5 for Neuron
        if "neuron" in self.t5_device:
            self._text_encoder.forward = torch.compile(
                self._text_encoder.forward,
                backend="neuron", fullgraph=True, dynamic=False
            )
            
            # Warmup
            dummy_ids, dummy_mask = self._tokenizer(
                ["warmup"], return_mask=True, add_special_tokens=True
            )
            with torch.no_grad():
                _ = self._text_encoder(
                    dummy_ids.to(self.t5_device), dummy_mask.to(self.t5_device)
                )
            torch.neuron.synchronize()
    
    def _load_dit(self):
        """Load DiT model lazily."""
        if self._dit_pipeline is not None:
            return
            
        from models.causal_inference_pipeline import CausalInferencePipeline
        
        print("[StreamingPipeline] Loading DiT model...")
        
        self._dit_pipeline = CausalInferencePipeline(
            denoising_step_list=self.model_config.denoising_step_list,
            num_frame_per_block=getattr(self.model_config, "num_frame_per_block", 1),
            context_noise=getattr(self.model_config, "context_noise", 0.0),
            warp_denoising_step=getattr(self.model_config, "warp_denoising_step", True),
            model_name=getattr(self.model_config, "model_name", "Wan2.1-T2V-1.3B"),
            timestep_shift=getattr(self.model_config, "timestep_shift", 5.0),
            frame_seq_length=self.frame_seq_length,
        )
        
        if self.config.checkpoint_path:
            print(f"  Loading checkpoint: {self.config.checkpoint_path}")
            state_dict = torch.load(
                self.config.checkpoint_path, map_location="cpu"
            )
            if self.config.use_ema:
                sd = state_dict['generator_ema']
                sd = OrderedDict(
                    (k.replace("_fsdp_wrapped_module.", ""), v) 
                    for k, v in sd.items()
                )
            else:
                sd = state_dict['generator']
            self._dit_pipeline.generator.load_state_dict(sd, strict=True)
        
        self._dit_pipeline.generator.model = self._dit_pipeline.generator.model.to(
            self.dit_device
        )
        print(f"[StreamingPipeline] DiT model placed on {self.dit_device}")
    
    def _load_vae(self):
        """Load VAE model lazily."""
        if self._vae_model is not None:
            return
            
        from wan.modules.vae import _video_vae
        
        print("[StreamingPipeline] Loading VAE...")
        
        self._vae_model = _video_vae(
            pretrained_path=self.config.vae_path, z_dim=16
        ).eval().requires_grad_(False)
        self._vae_model = self._vae_model.to(
            dtype=torch.bfloat16, device=self.vae_device
        )
        print(f"[StreamingPipeline] VAE placed on {self.vae_device}")
        
        # VAE normalization constants
        mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=torch.bfloat16).to(self.vae_device)
        
        std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], dtype=torch.bfloat16).to(self.vae_device)
        
        self._vae_scale = [mean, 1.0 / std]
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings. Runs on t5_device."""
        self._load_t5()
        
        ids, mask = self._tokenizer(
            [prompt], return_mask=True, add_special_tokens=True
        )
        ids, mask = ids.to(self.t5_device), mask.to(self.t5_device)
        seq_len = mask.gt(0).sum(dim=1).long()
        
        with torch.no_grad():
            prompt_embeds = self._text_encoder(ids, mask)
        
        if "neuron" in self.t5_device:
            torch.neuron.synchronize()
        
        prompt_embeds = prompt_embeds.cpu()
        prompt_embeds[0, seq_len[0].cpu():] = 0.0
        
        return prompt_embeds
    
    def decode_latents_to_frames(
        self, latents: torch.Tensor
    ) -> List[Image.Image]:
        """Decode latent tensor to PIL Images.
        
        Args:
            latents: [B, T, C, H, W] latent tensor
            
        Returns:
            List of PIL Images
        """
        self._load_vae()
        
        latents = latents.to(torch.bfloat16).to(self.vae_device)
        
        # [B, T, C, H, W] -> [B, C, T, H, W]
        latents_bcthw = rearrange(latents, 'b t c h w -> b c t h w')
        
        with torch.no_grad():
            video = self._vae_model.decode(latents_bcthw, self._vae_scale)
            video = video.clamp(-1, 1)
        
        # [B, C, T, H, W] -> [B, T, H, W, C]
        video = rearrange(video, 'b c t h w -> b t h w c')
        video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
        
        # Convert to PIL Images
        frames = []
        video_np = (255.0 * video[0]).to(torch.uint8).numpy()
        for i in range(video_np.shape[0]):
            frames.append(Image.fromarray(video_np[i]))
        
        return frames
    
    def generate_streaming(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Iterator[Tuple[int, Image.Image]]:
        """Generate video frames one-by-one, yielding as they become available.
        
        This modifies the rolling forcing loop to yield finalized frames
        progressively instead of waiting for full generation.
        
        Args:
            prompt: Text prompt for video generation
            num_frames: Number of frames to generate (default from config)
            progress_callback: Optional callback(current_frame, total_frames)
            
        Yields:
            Tuple of (frame_index, PIL.Image) as frames are finalized
        """
        num_frames = num_frames or self.config.num_frames
        
        # Encode prompt
        print(f"[Streaming] Encoding prompt: {prompt[:50]}...")
        prompt_embeds = self.encode_prompt(prompt)
        
        # Load DiT
        self._load_dit()
        
        # Prepare noise — on dit_device since DiT consumes it
        noise = torch.randn(
            1, num_frames, 16, self.latent_h, self.latent_w,
            dtype=torch.bfloat16
        ).to(self.dit_device)
        
        conditional_dict = {
            "prompt_embeds": prompt_embeds.to(torch.bfloat16).to(self.dit_device)
        }
        
        # Get pipeline parameters
        nfpb = self._dit_pipeline.num_frame_per_block
        nds = len(self._dit_pipeline.denoising_step_list)
        num_blocks = num_frames // nfpb
        window_num = num_blocks + nds - 1
        
        print(f"[Streaming] Starting generation: {num_frames} frames, {window_num} windows")
        
        # Run full inference (we'll make this truly incremental later)
        print("[Streaming] Running DiT inference...")
        latents = self._dit_pipeline.inference_rolling_forcing(
            noise, conditional_dict
        ).cpu()
        
        print("[Streaming] Decoding frames...")
        
        # Decode and yield frames one by one
        for frame_idx in range(num_frames):
            # Decode single frame
            frame_latent = latents[:, frame_idx:frame_idx+1]
            frames = self.decode_latents_to_frames(frame_latent)
            
            if progress_callback:
                progress_callback(frame_idx + 1, num_frames)
            
            yield (frame_idx, frames[0])
    
    def generate_streaming_true(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Iterator[Tuple[int, Image.Image, torch.Tensor]]:
        """True streaming generation that yields frames during DiT inference.
        
        This is the optimized version that yields frames as soon as they are
        finalized in the rolling forcing process, before full video completion.
        
        Args:
            prompt: Text prompt for video generation
            num_frames: Number of frames to generate
            progress_callback: Optional callback(current_frame, total_frames)
            
        Yields:
            Tuple of (frame_index, PIL.Image, latent_tensor)
        """
        num_frames = num_frames or self.config.num_frames
        
        # Encode prompt
        prompt_embeds = self.encode_prompt(prompt)
        
        # Load models
        self._load_dit()
        self._load_vae()
        
        # Prepare noise and inputs — on dit_device since DiT consumes them
        noise = torch.randn(
            1, num_frames, 16, self.latent_h, self.latent_w,
            dtype=torch.bfloat16
        ).to(self.dit_device)
        
        conditional_dict = {
            "prompt_embeds": prompt_embeds.to(torch.bfloat16).to(self.dit_device)
        }
        
        # Use the streaming variant of inference
        for frame_idx, latent_block in self._inference_rolling_forcing_streaming(
            noise, conditional_dict
        ):
            # Decode the finalized frames
            frames = self.decode_latents_to_frames(latent_block)
            
            for i, frame in enumerate(frames):
                actual_idx = frame_idx + i
                if progress_callback:
                    progress_callback(actual_idx + 1, num_frames)
                yield (actual_idx, frame, latent_block[:, i:i+1])
    
    def _inference_rolling_forcing_streaming(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Modified rolling forcing that yields finalized latent blocks.
        
        This generator wraps the inference loop and yields blocks of latents
        as soon as they are fully denoised (after passing through all
        denoising steps in the rolling window).
        
        Yields:
            Tuple of (start_frame_index, latent_block [B, num_frame_per_block, C, H, W])
        """
        pipe = self._dit_pipeline
        
        batch_size, num_frames, num_channels, height, width = noise.shape
        nfpb = pipe.num_frame_per_block
        nds = len(pipe.denoising_step_list)

        # Round up to next multiple of num_frame_per_block if needed
        requested_frames = num_frames
        if num_frames % nfpb != 0:
            num_frames = ((num_frames // nfpb) + 1) * nfpb
            pad_count = num_frames - requested_frames
            pad_noise = torch.randn(
                batch_size, pad_count, num_channels, height, width,
                dtype=noise.dtype, device=noise.device)
            noise = torch.cat([noise, pad_noise], dim=1)

        num_blocks = num_frames // nfpb
        window_num = num_blocks + nds - 1
        
        # Initialize caches
        if pipe.kv_cache_clean is None:
            pipe._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
            pipe._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            for block_index in range(pipe.num_transformer_blocks):
                pipe.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(pipe.kv_cache_clean)):
                pipe.kv_cache_clean[block_index]["global_end_index"] = 0
                pipe.kv_cache_clean[block_index]["local_end_index"] = 0
        
        # Allocate buffers (simplified from original)
        max_frames = nds * nfpb
        
        output = torch.zeros(
            [batch_size, num_frames + max_frames - nfpb, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        
        noisy_cache = torch.zeros(
            [batch_size, num_frames + max_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        
        if pipe.timestep_patterns.device != noise.device:
            pipe.timestep_patterns = pipe.timestep_patterns.to(noise.device)
            pipe.sigma_patterns = pipe.sigma_patterns.to(noise.device)
        
        padded_input = torch.zeros(
            [batch_size, max_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        padded_timestep = torch.zeros(
            [batch_size, max_frames], device=noise.device, dtype=torch.float32
        )
        padded_sigma = torch.zeros(
            [batch_size, max_frames], device=noise.device, dtype=torch.float32
        )
        
        cache_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )
        cache_timestep = torch.full(
            [batch_size, nfpb], pipe.context_noise,
            device=noise.device, dtype=torch.float32
        )
        cache_sigma = torch.full(
            [batch_size, nfpb], pipe.context_sigma,
            device=noise.device, dtype=torch.float32
        )
        
        # Precompute sigma values
        block_sigma_list = []
        for step in pipe.denoising_step_list:
            sigma_val = pipe._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones(
                    [batch_size * nfpb, 1, 1, 1],
                    dtype=torch.float32, device=noise.device
                )
            )
        
        # Build window indices
        window_start_blocks = []
        window_end_blocks = []
        pattern_indices = []
        
        for window_index in range(window_num):
            start_block = max(0, window_index - nds + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)
            num_blks = end_block - start_block + 1
            if num_blks == nds:
                pattern_indices.append(0)
            elif start_block == 0:
                pattern_indices.append(num_blks)
            else:
                pattern_indices.append(nds - 1 + num_blks)
        
        # Track finalized frames
        last_finalized_block = -1
        
        # Rolling forcing loop
        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]
            
            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame
            
            # Copy noisy cache
            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames]
            )
            
            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame]
                )
            
            padded_timestep[:] = pipe.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = pipe.sigma_patterns[pattern_indices[window_index]]
            
            # Denoise
            _, denoised_pred = pipe.generator(
                noisy_image_or_video=padded_input,
                conditional_dict=conditional_dict,
                timestep=padded_timestep,
                kv_cache=pipe.kv_cache_clean,
                crossattn_cache=pipe.crossattn_cache,
                current_start=current_start_frame * pipe.frame_seq_length,
                num_valid_frames=current_num_frames,
                shared_buffers=(pipe.shared_buffer_k, pipe.shared_buffer_v),
                sigma=padded_sigma,
            )
            
            output[:, current_start_frame:current_start_frame + max_frames].copy_(
                denoised_pred
            )
            
            # Re-noising for non-finalized blocks
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds
            ) else (nds - 1)
            
            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset
                
                if step_index == nds - 1:
                    continue
                
                full_noise = torch.randn(
                    batch_size * current_num_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype
                ).to(noise.device)
                
                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb
                ].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, current_num_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]
                
                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    pipe._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))
            
            # Cache update
            cache_input.copy_(denoised_pred[:, :nfpb])
            pipe.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=cache_timestep,
                kv_cache=pipe.kv_cache_clean,
                crossattn_cache=pipe.crossattn_cache,
                current_start=current_start_frame * pipe.frame_seq_length,
                updating_cache=True,
                num_valid_frames=nfpb,
                shared_buffers=(pipe.shared_buffer_k, pipe.shared_buffer_v),
                sigma=cache_sigma,
            )
            
            # Check for newly finalized blocks
            # A block is finalized when it has passed through all denoising steps
            # This happens when window_index >= block_index + nds - 1
            finalized_block = window_index - nds + 1
            
            if finalized_block > last_finalized_block and finalized_block >= 0:
                # Yield all newly finalized blocks (trim padded frames)
                for blk in range(last_finalized_block + 1, finalized_block + 1):
                    if blk < num_blocks:
                        start_frame = blk * nfpb
                        end_frame = min((blk + 1) * nfpb, requested_frames)
                        if start_frame < requested_frames:
                            yield (start_frame, output[:, start_frame:end_frame].cpu())
                
                last_finalized_block = finalized_block
        
        # Yield any remaining blocks (trim padded frames)
        for blk in range(last_finalized_block + 1, num_blocks):
            start_frame = blk * nfpb
            end_frame = min((blk + 1) * nfpb, requested_frames)
            if start_frame < requested_frames:
                yield (start_frame, output[:, start_frame:end_frame].cpu())
    
    def generate_chunked(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        chunk_size: int = 6,
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Iterator[str]:
        """Generate video in chunks, yielding video segment paths.
        
        Better quality than frame-by-frame due to proper video encoding.
        
        Args:
            prompt: Text prompt for video generation
            num_frames: Number of frames to generate
            chunk_size: Frames per video chunk
            output_dir: Directory for chunk files (default: temp dir)
            progress_callback: Optional callback(current_frame, total_frames)
            
        Yields:
            Path to each video chunk file
        """
        import imageio
        
        num_frames = num_frames or self.config.num_frames
        output_dir = output_dir or tempfile.mkdtemp(prefix="video_chunks_")
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect frames into chunks
        chunk_frames = []
        chunk_idx = 0
        
        for frame_idx, frame in self.generate_streaming(
            prompt, num_frames, progress_callback
        ):
            chunk_frames.append(np.array(frame))
            
            if len(chunk_frames) >= chunk_size:
                # Save chunk as video
                chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.mp4")
                imageio.mimwrite(
                    chunk_path, chunk_frames, fps=self.config.fps
                )
                yield chunk_path
                
                chunk_frames = []
                chunk_idx += 1
        
        # Save remaining frames
        if chunk_frames:
            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.mp4")
            imageio.mimwrite(chunk_path, chunk_frames, fps=self.config.fps)
            yield chunk_path
    
    def generate_full(
        self,
        prompt: str,
        num_frames: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate complete video (non-streaming).
        
        Args:
            prompt: Text prompt
            num_frames: Number of frames
            output_path: Output video path
            
        Returns:
            Path to generated video
        """
        import imageio
        
        num_frames = num_frames or self.config.num_frames
        output_path = output_path or tempfile.mktemp(suffix=".mp4")
        
        frames = []
        for _, frame in self.generate_streaming(prompt, num_frames):
            frames.append(np.array(frame))
        
        imageio.mimwrite(output_path, frames, fps=self.config.fps)
        return output_path
