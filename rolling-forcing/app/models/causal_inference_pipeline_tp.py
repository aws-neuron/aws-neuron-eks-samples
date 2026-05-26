"""TP-aware CausalInferencePipeline for Wan2.1-T2V-14B on Trainium.

Key differences from the single-rank pipeline:
  - KV cache sized for local heads: [B, S, num_heads_per_rank, head_dim]
  - Shared buffers sized for local heads
  - Cross-attention cache sized for local heads
  - All ranks execute in lockstep (same inputs, same control flow)
  - TP communication handled transparently inside the model

Memory planning (all 3 models co-located on each rank):
  - TE (T5-XXL): replicated, run once per prompt
  - VAE: replicated, run once for decode
  - DiT (14B): TP-sharded, performance bottleneck, gets all 4 cores
"""

import os
import time
from typing import List, Optional

import torch
import torch.distributed as dist

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn):
    """No-op wrapper since torch_neuronx.jit is not available."""
    return fn

from models.causal_model_wrapper_tp import WanDiffusionWrapperTP
from models.layers import ATTN_SEQLEN_MULTIPLE
from models.tp_utils import get_tp_rank, get_tp_world_size


def add_noise(original_samples, noise, sigma):
    """Diffusion forward process: mix clean samples with noise.

    Args:
        original_samples: [B*F, C, H, W] clean latents
        noise:            [B*F, C, H, W] random noise
        sigma:            [B*F, 1, 1, 1] precomputed sigma values

    Returns: [B*F, C, H, W] noisy samples
    """
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


class CausalInferencePipelineTP(torch.nn.Module):
    """TP-aware rolling-forcing inference pipeline for Wan2.1-T2V-14B.

    All 4 NeuronCores execute the pipeline in lockstep:
      - Text encoding: replicated (all ranks compute same result)
      - DiT denoising: TP-sharded (all-reduce communication inside model)
      - VAE decode: replicated (all ranks decode same latent)
      - Only rank 0 saves output

    KV cache is sized for local heads (num_heads_per_rank = 10 for TP=4).
    """

    def __init__(
            self,
            denoising_step_list: List[int],
            num_frame_per_block: int = 3,
            context_noise: float = 0.0,
            warp_denoising_step: bool = True,
            frame_seq_length: int = 1560,
            # Model construction
            model_name: str = "Wan2.1-T2V-14B",
            timestep_shift: float = 5.0,
            local_attn_size: int = -1,
            sink_size: int = 0,
            num_layers: Optional[int] = None,
            tp_degree: int = 4,
            # Optional pre-built generator
            generator: Optional[WanDiffusionWrapperTP] = None,
    ):
        super().__init__()

        self.tp_degree = tp_degree
        self.tp_rank = get_tp_rank()

        if generator is None:
            generator = WanDiffusionWrapperTP(
                model_name=model_name,
                timestep_shift=timestep_shift,
                is_causal=True,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                num_layers=num_layers,
                frame_length=frame_seq_length,
                num_frame_per_block=num_frame_per_block,
                tp_degree=tp_degree,
            )
        self.generator = generator

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long)
        if warp_denoising_step:
            timesteps = torch.cat((
                self.scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = frame_seq_length
        self.context_noise = context_noise
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = self.generator.model.local_attn_size

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # TP-aware dimensions for KV cache
        self._num_heads_per_rank = self.generator.num_heads_per_rank
        self._head_dim = self.generator.head_dim
        self._text_len = self.generator.model.text_len

        self.timestep_patterns = self._build_timestep_patterns()
        self.sigma_patterns = self._build_sigma_patterns()
        self.context_sigma = self._timestep_to_sigma(self.context_noise)

        self._add_noise = jit(add_noise)

        self.kv_cache_clean = None
        self.crossattn_cache = None

        print(f"[Rank {self.tp_rank}] CausalInferencePipelineTP initialized: "
              f"{self.num_transformer_blocks} blocks, "
              f"{self._num_heads_per_rank} heads/rank, "
              f"frame_seq_length={frame_seq_length}")

    def release_device_memory(self):
        """Release KV cache and shared buffer tensors to free device HBM."""
        if self.kv_cache_clean is not None:
            for cache in self.kv_cache_clean:
                for key in ("k", "v"):
                    if key in cache:
                        del cache[key]
            self.kv_cache_clean = None
        if self.crossattn_cache is not None:
            for cache in self.crossattn_cache:
                for key in ("k", "v"):
                    if key in cache:
                        del cache[key]
            self.crossattn_cache = None
        if hasattr(self, 'shared_buffer_k') and self.shared_buffer_k is not None:
            del self.shared_buffer_k
            self.shared_buffer_k = None
        if hasattr(self, 'shared_buffer_v') and self.shared_buffer_v is not None:
            del self.shared_buffer_v
            self.shared_buffer_v = None
        import gc
        gc.collect()
        print(f"[Rank {self.tp_rank}] Released device memory")

    @torch.no_grad()
    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """Run rolling-forcing inference with TP.

        All ranks execute identical control flow with identical inputs.
        The only difference is internal weight/KV sharding.

        Args:
            noise: [B, num_frames, C, H, W] initial noise (same on all ranks)
            conditional_dict: {"prompt_embeds": [B, 512, 4096]} (same on all ranks)

        Returns:
            output latents [B, num_output_frames, C, H, W] (identical on all ranks)
        """
        profile = os.environ.get("PROFILE_PIPELINE", "0") == "1"

        batch_size, num_frames, num_channels, height, width = noise.shape

        # Round up to next multiple of num_frame_per_block if needed
        nfpb = self.num_frame_per_block
        requested_frames = num_frames
        if num_frames % nfpb != 0:
            num_frames = ((num_frames // nfpb) + 1) * nfpb
            pad_count = num_frames - requested_frames
            pad_noise = torch.randn(
                batch_size, pad_count, num_channels, height, width,
                dtype=noise.dtype, device=noise.device)
            noise = torch.cat([noise, pad_noise], dim=1)

        num_blocks = num_frames // nfpb
        num_output_frames = requested_frames

        if profile:
            init_start = time.perf_counter()

        # Initialize or reset caches (sized for local heads)
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_clean)):
                self.kv_cache_clean[block_index]["global_end_index"] = 0
                self.kv_cache_clean[block_index]["local_end_index"] = 0

        # Construct rolling forcing windows
        nds = len(self.denoising_step_list)
        rolling_window_length_blocks = nds
        window_start_blocks = []
        window_end_blocks = []
        pattern_indices = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
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

        # Static shape constants
        max_frames = rolling_window_length_blocks * nfpb

        output = torch.zeros(
            [batch_size, num_output_frames + max_frames - nfpb,
             num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        noisy_cache = torch.zeros(
            [batch_size, num_output_frames + max_frames,
             num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        if self.timestep_patterns.device != noise.device:
            self.timestep_patterns = self.timestep_patterns.to(noise.device)
            self.sigma_patterns = self.sigma_patterns.to(noise.device)

        padded_input = torch.zeros(
            [batch_size, max_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        padded_timestep = torch.zeros(
            [batch_size, max_frames],
            device=noise.device, dtype=torch.float32)
        padded_sigma = torch.zeros(
            [batch_size, max_frames],
            device=noise.device, dtype=torch.float32)

        cache_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        cache_timestep = torch.full(
            [batch_size, nfpb], self.context_noise,
            device=noise.device, dtype=torch.float32)
        cache_sigma = torch.full(
            [batch_size, nfpb], self.context_sigma,
            device=noise.device, dtype=torch.float32)

        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones(
                    [batch_size * nfpb, 1, 1, 1],
                    dtype=torch.float32, device=noise.device))

        if profile:
            init_end = time.perf_counter()
            diffusion_start = time.perf_counter()
            window_times = []

        # Denoising loop with rolling forcing (all ranks in lockstep)
        for window_index in range(window_num):
            if profile:
                window_start = time.perf_counter()

            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame

            if self.tp_rank == 0:
                print(f"[DiT-TP] Window {window_index}/{window_num} | "
                      f"frames {current_start_frame}-{current_end_frame-1}", flush=True)

            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])

            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])

            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]

            num_valid_frames = current_num_frames

            # DiT forward (TP-sharded, all-reduce inside model)
            _, denoised_pred = self.generator(
                noisy_image_or_video=padded_input,
                conditional_dict=conditional_dict,
                timestep=padded_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                num_valid_frames=num_valid_frames,
                shared_buffers=(self.shared_buffer_k, self.shared_buffer_v),
                sigma=padded_sigma,
            )

            # Copy denoised prediction to output, clamping to avoid overflow
            copy_end = min(current_start_frame + max_frames, output.shape[1])
            copy_len = copy_end - current_start_frame
            output[:, current_start_frame:copy_end].copy_(
                denoised_pred[:, :copy_len])

            # Re-noising (local computation, identical on all ranks due to same RNG seed)
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset

                if step_index == nds - 1:
                    continue

                # Noise sized for max_frames (matching denoised_pred shape)
                full_noise = torch.randn(
                    batch_size * max_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(noise.device)

                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb
                ].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, max_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]

                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    self._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))

            # Cache-update call
            cache_input.copy_(denoised_pred[:, :nfpb])
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=cache_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                updating_cache=True,
                num_valid_frames=nfpb,
                shared_buffers=(self.shared_buffer_k, self.shared_buffer_v),
                sigma=cache_sigma,
            )

            if profile:
                wt = time.perf_counter() - window_start
                window_times.append(wt)
                if self.tp_rank == 0:
                    print(f"  Window {window_index}: {wt*1000:.2f} ms", flush=True)

        if profile:
            diffusion_end = time.perf_counter()
            init_time = (init_end - init_start) * 1000
            diffusion_time = (diffusion_end - diffusion_start) * 1000
            total_time = init_time + diffusion_time
            if self.tp_rank == 0:
                print(f"\n[TP Profiling] (rank 0):")
                print(f"  Init: {init_time:.2f} ms")
                print(f"  Diffusion: {diffusion_time:.2f} ms "
                      f"({window_num} windows, avg {diffusion_time/window_num:.1f} ms/window)")
                print(f"  Total: {total_time:.2f} ms")

        return output[:, :num_output_frames]

    @torch.no_grad()
    def inference_rolling_forcing_streaming(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):
        """Generator version that yields finalized blocks for streaming.

        Same as inference_rolling_forcing but yields intermediate results
        as blocks complete all denoising steps.

        Yields:
            Tuple of (start_frame_index, latent_block [B, nfpb, C, H, W] on CPU)
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        nfpb = self.num_frame_per_block
        requested_frames = num_frames

        if num_frames % nfpb != 0:
            num_frames = ((num_frames // nfpb) + 1) * nfpb
            pad_count = num_frames - requested_frames
            pad_noise = torch.randn(
                batch_size, pad_count, num_channels, height, width,
                dtype=noise.dtype, device=noise.device)
            noise = torch.cat([noise, pad_noise], dim=1)

        num_blocks = num_frames // nfpb

        if self.kv_cache_clean is None:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_clean)):
                self.kv_cache_clean[block_index]["global_end_index"] = 0
                self.kv_cache_clean[block_index]["local_end_index"] = 0

        nds = len(self.denoising_step_list)
        max_frames = nds * nfpb
        window_num = num_blocks + nds - 1

        output = torch.zeros(
            [batch_size, num_frames + max_frames - nfpb,
             num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        noisy_cache = torch.zeros(
            [batch_size, num_frames + max_frames,
             num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        if self.timestep_patterns.device != noise.device:
            self.timestep_patterns = self.timestep_patterns.to(noise.device)
            self.sigma_patterns = self.sigma_patterns.to(noise.device)

        padded_input = torch.zeros(
            [batch_size, max_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        padded_timestep = torch.zeros(
            [batch_size, max_frames], device=noise.device, dtype=torch.float32)
        padded_sigma = torch.zeros(
            [batch_size, max_frames], device=noise.device, dtype=torch.float32)

        cache_input = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        cache_timestep = torch.full(
            [batch_size, nfpb], self.context_noise,
            device=noise.device, dtype=torch.float32)
        cache_sigma = torch.full(
            [batch_size, nfpb], self.context_sigma,
            device=noise.device, dtype=torch.float32)

        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones(
                    [batch_size * nfpb, 1, 1, 1],
                    dtype=torch.float32, device=noise.device))

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

        last_finalized_block = -1

        for window_index in range(window_num):
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame

            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])

            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])

            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]

            num_valid_frames = current_num_frames

            _, denoised_pred = self.generator(
                noisy_image_or_video=padded_input,
                conditional_dict=conditional_dict,
                timestep=padded_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                num_valid_frames=num_valid_frames,
                shared_buffers=(self.shared_buffer_k, self.shared_buffer_v),
                sigma=padded_sigma,
            )

            # Copy denoised prediction to output, clamping to avoid overflow
            copy_end = min(current_start_frame + max_frames, output.shape[1])
            copy_len = copy_end - current_start_frame
            output[:, current_start_frame:copy_end].copy_(
                denoised_pred[:, :copy_len])

            # Re-noising
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset
                if step_index == nds - 1:
                    continue

                # Noise sized for max_frames (matching denoised_pred shape)
                full_noise = torch.randn(
                    batch_size * max_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(noise.device)

                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb
                ].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, max_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]

                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    self._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))

            # Cache-update call
            cache_input.copy_(denoised_pred[:, :nfpb])
            self.generator(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=cache_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                updating_cache=True,
                num_valid_frames=nfpb,
                shared_buffers=(self.shared_buffer_k, self.shared_buffer_v),
                sigma=cache_sigma,
            )

            # Yield finalized blocks
            finalized_block = window_index - nds + 1
            if finalized_block > last_finalized_block and finalized_block >= 0:
                for blk in range(last_finalized_block + 1, finalized_block + 1):
                    if blk < num_blocks:
                        sf = blk * nfpb
                        ef = min((blk + 1) * nfpb, requested_frames)
                        if sf < requested_frames:
                            yield (sf, output[:, sf:ef].clone().cpu())
                last_finalized_block = finalized_block

        # Yield remaining blocks
        for blk in range(last_finalized_block + 1, num_blocks):
            sf = blk * nfpb
            ef = min((blk + 1) * nfpb, requested_frames)
            if sf < requested_frames:
                yield (sf, output[:, sf:ef].clone().cpu())

    # ─── Helper methods ─────────────────────────────────────────────────

    def _build_timestep_patterns(self):
        """Build timestep patterns — same logic as single-rank pipeline."""
        nds = len(self.denoising_step_list)
        nfpb = self.num_frame_per_block
        max_frames = nds * nfpb

        steady = []
        for ts in reversed(self.denoising_step_list):
            steady.extend([ts.item()] * nfpb)

        patterns = [steady]
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[-cnf:] + [0.0] * (max_frames - cnf))
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[:cnf] + [0.0] * (max_frames - cnf))

        return torch.tensor(patterns, dtype=torch.float32)

    def _timestep_to_sigma(self, timestep_val):
        """Map timestep to sigma."""
        idx = torch.argmin((self.scheduler.timesteps - timestep_val).abs())
        return self.scheduler.sigmas[idx].item()

    def _build_sigma_patterns(self):
        """Precompute sigma patterns."""
        sigma_patterns = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern):
                sigma_patterns[i, j] = self._timestep_to_sigma(t.item())
        return sigma_patterns

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize per-layer KV cache sized for LOCAL heads (TP-aware).

        Shape: [B, cache_size, num_heads_per_rank, head_dim]
        For 14B with TP=4: [1, 37440, 10, 128]
        """
        kv_cache_alloc_size = self.frame_seq_length * 24  # 37440
        block_length = self.num_frame_per_block * self.frame_seq_length
        eviction_copy_size = kv_cache_alloc_size - block_length
        max_buffer_size = max(self.frame_seq_length * 21, eviction_copy_size)
        max_buffer_size = (
            (max_buffer_size + ATTN_SEQLEN_MULTIPLE - 1)
            // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE)

        kv_cache_clean = []
        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_alloc_size,
                     self._num_heads_per_rank, self._head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, kv_cache_alloc_size,
                     self._num_heads_per_rank, self._head_dim],
                    dtype=dtype, device=device),
                "global_end_index": 0,
                "local_end_index": 0,
            })

        self.kv_cache_clean = kv_cache_clean

        # Shared buffers for attention assembly (sized for local heads)
        self.shared_buffer_k = torch.zeros(
            [batch_size, max_buffer_size,
             self._num_heads_per_rank, self._head_dim],
            dtype=dtype, device=device)
        self.shared_buffer_v = torch.zeros(
            [batch_size, max_buffer_size,
             self._num_heads_per_rank, self._head_dim],
            dtype=dtype, device=device)

        if self.tp_rank == 0:
            kv_mem_gb = (
                self.num_transformer_blocks * 2 *
                batch_size * kv_cache_alloc_size *
                self._num_heads_per_rank * self._head_dim * 2  # bf16
            ) / (1024**3)
            buf_mem_gb = (
                2 * batch_size * max_buffer_size *
                self._num_heads_per_rank * self._head_dim * 2
            ) / (1024**3)
            print(f"[TP KV Cache] Per-rank allocation:")
            print(f"  KV cache: {self.num_transformer_blocks} layers × "
                  f"[{batch_size}, {kv_cache_alloc_size}, "
                  f"{self._num_heads_per_rank}, {self._head_dim}] = {kv_mem_gb:.2f} GB")
            print(f"  Shared buffers: [{batch_size}, {max_buffer_size}, "
                  f"{self._num_heads_per_rank}, {self._head_dim}] = {buf_mem_gb:.3f} GB")

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize per-layer cross-attention cache (sized for local heads)."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros(
                    [batch_size, self._text_len,
                     self._num_heads_per_rank, self._head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, self._text_len,
                     self._num_heads_per_rank, self._head_dim],
                    dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache
