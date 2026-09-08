import os
import time
from typing import List, Optional

import torch

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn):
    """No-op wrapper since torch_neuronx.jit is not available."""
    return fn

from models.causal_model_wrapper import WanDiffusionWrapper
from models.layers import ATTN_SEQLEN_MULTIPLE


def add_noise(original_samples, noise, sigma):
    """Diffusion forward process: mix clean samples with noise.

    Replaces FlowMatchScheduler.add_noise() with precomputed sigma
    (no argmin lookup).

    Args:
        original_samples: [B*F, C, H, W] clean latents
        noise:            [B*F, C, H, W] random noise
        sigma:            [B*F, 1, 1, 1] precomputed sigma values

    Returns: [B*F, C, H, W] noisy samples, same dtype as noise
    """
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            denoising_step_list: List[int],
            num_frame_per_block: int = 3,
            context_noise: float = 0.0,
            warp_denoising_step: bool = True,
            frame_seq_length: int = 1560,
            # Model construction (used only if generator is None)
            model_name: str = "Wan2.1-T2V-1.3B",
            timestep_shift: float = 5.0,
            local_attn_size: int = -1,
            sink_size: int = 0,
            num_layers: Optional[int] = None,
            # Optional pre-built generator
            generator: Optional[WanDiffusionWrapper] = None,
    ):
        super().__init__()

        if generator is None:
            generator = WanDiffusionWrapper(
                model_name=model_name,
                timestep_shift=timestep_shift,
                is_causal=True,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                num_layers=num_layers,
                frame_length=frame_seq_length,
                num_frame_per_block=num_frame_per_block,
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

        # Derive cache dimensions from model config
        self._num_heads = self.generator.model.num_heads
        self._head_dim = self.generator.model.dim // self.generator.model.num_heads
        self._text_len = self.generator.model.text_len

        self.timestep_patterns = self._build_timestep_patterns()
        self.sigma_patterns = self._build_sigma_patterns()
        self.context_sigma = self._timestep_to_sigma(self.context_noise)

        self._add_noise = jit(add_noise)

        self.kv_cache_clean = None
        self.crossattn_cache = None

    def release_device_memory(self):
        """Release KV cache and shared buffer tensors to free device HBM.

        Call between generation requests to prevent HBM OOM from accumulated
        scratchpad memory. The caches will be re-allocated on the next request.
        """
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
        print("[CausalInferencePipeline] Released device memory (KV cache + shared buffers)")

    @torch.no_grad()
    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        profile = os.environ.get("PROFILE_PIPELINE", "0") == "1"
        async_mode = profile and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) > 0

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

        # Initialize or reset caches
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
        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps
        nds = num_denoising_steps
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
                pattern_indices.append(0)                   # steady-state
            elif start_block == 0:
                pattern_indices.append(num_blks)             # ramp-up
            else:
                pattern_indices.append(nds - 1 + num_blks)   # ramp-down

        # Static shape constants
        max_frames = rolling_window_length_blocks * self.num_frame_per_block
        nfpb = self.num_frame_per_block

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

        # Pre-allocate 3-frame buffers for cache-update call (constant timestep/sigma)
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
                sigma_val * torch.ones([batch_size * nfpb, 1, 1, 1], dtype=torch.float32, device=noise.device))

        if profile:
            init_end = time.perf_counter()
            diffusion_start = time.perf_counter()
            window_times = []

        # Denoising loop with rolling forcing
        for window_index in range(window_num):
            if profile:
                window_start = time.perf_counter()

            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame

            print(f"[DiT] Window {window_index}/{window_num} | frames {current_start_frame}-{current_end_frame-1}", flush=True)

            if not async_mode and profile:
                _t = time.perf_counter()
            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])
            if not async_mode and profile:
                _t_copy_noisy = (time.perf_counter() - _t) * 1000

            _t_copy_noise = 0.0
            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                if not async_mode and profile:
                    _t = time.perf_counter()
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])
                if not async_mode and profile:
                    _t_copy_noise = (time.perf_counter() - _t) * 1000

            if not async_mode and profile:
                _t = time.perf_counter()
            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            if not async_mode and profile:
                _t_timestep = (time.perf_counter() - _t) * 1000

            if not async_mode and profile:
                _t = time.perf_counter()
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]
            if not async_mode and profile:
                _t_sigma = (time.perf_counter() - _t) * 1000

            num_valid_frames = current_num_frames

            # Denoise call
            if not async_mode and profile:
                print(f"  [denoise]")
                _t = time.perf_counter()
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
            if not async_mode and profile:
                _t_denoise = (time.perf_counter() - _t) * 1000

            if not async_mode and profile:
                _t = time.perf_counter()
            copy_end = min(current_start_frame + max_frames, output.shape[1])
            copy_len = copy_end - current_start_frame
            output[:, current_start_frame:copy_end].copy_(
                denoised_pred[:, :copy_len])
            if not async_mode and profile:
                _t_output_copy = (time.perf_counter() - _t) * 1000

            # Re-noising
            if not async_mode and profile:
                _t = time.perf_counter()
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset

                if step_index == nds - 1:
                    continue

                full_noise = torch.randn(
                    batch_size * num_valid_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(noise.device)

                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb
                ].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, num_valid_frames)
                )[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                block_sigma = block_sigma_list[step_index + 1]

                noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                    self._add_noise(block_pred, block_noise, block_sigma) \
                    .unflatten(0, (batch_size, nfpb))
            if not async_mode and profile:
                _t_renoise = (time.perf_counter() - _t) * 1000

            # Cache-update call (3-frame input, no padding)
            if not async_mode and profile:
                _t = time.perf_counter()
            cache_input.copy_(denoised_pred[:, :nfpb])
            if not async_mode and profile:
                _t_cache_copy = (time.perf_counter() - _t) * 1000

            if not async_mode and profile:
                print(f"  [cache-update]")
                _t = time.perf_counter()
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
            if not async_mode and profile:
                _t_cache_call = (time.perf_counter() - _t) * 1000

            if profile:
                if async_mode:
                    # Transfer a scalar to CPU to sync with the Neuron runtime
                    output[0, 0, 0, 0, 0].cpu()
                    wt = time.perf_counter() - window_start
                    window_times.append(wt)
                    print(f"Window {window_index}: {wt*1000:.2f} ms", flush=True)
                else:
                    wt = time.perf_counter() - window_start
                    window_times.append(wt)
                    print(f"  denoise={_t_denoise:.1f}ms  output_copy={_t_output_copy:.1f}ms  renoise={_t_renoise:.1f}ms  cache_copy={_t_cache_copy:.1f}ms  cache_call={_t_cache_call:.1f}ms")
                    print(f"Window {window_index}: {wt*1000:.2f} ms  (setup: copy_noisy={_t_copy_noisy:.1f}ms  copy_noise={_t_copy_noise:.1f}ms  timestep={_t_timestep:.1f}ms  sigma={_t_sigma:.1f}ms)", flush=True)

        if profile:
            diffusion_end = time.perf_counter()
            init_time = (init_end - init_start) * 1000
            diffusion_time = (diffusion_end - diffusion_start) * 1000
            total_time = init_time + diffusion_time

            print("Profiling results:")
            print(f"  - Initialization time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, wt in enumerate(window_times):
                wt_ms = wt * 1000
                print(f"    - Window {i} time: {wt_ms:.2f} ms ({100 * wt_ms / diffusion_time:.2f}% of diffusion)")
            print(f"  - Total time: {total_time:.2f} ms")

        return output[:, :num_output_frames]

    @torch.no_grad()
    def inference_rolling_forcing_streaming(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):
        """Generator version of inference_rolling_forcing that yields finalized blocks.

        Same computation as inference_rolling_forcing, but yields intermediate
        results as blocks complete all denoising steps, enabling true streaming.

        Yields:
            Tuple of (start_frame_index, latent_block [B, nfpb, C, H, W] on CPU)
        """
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

        # Initialize or reset caches
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
        nfpb = self.num_frame_per_block
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

            # Denoise call
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

            output[:, current_start_frame:current_start_frame + max_frames].copy_(
                denoised_pred)

            # Re-noising
            num_blks = end_block - start_block + 1
            step_base = (num_blks - 1) if (
                start_block == 0 and num_blks < nds) else (nds - 1)

            for block_idx in range(start_block, end_block + 1):
                local_offset = block_idx - start_block
                step_index = step_base - local_offset

                if step_index == nds - 1:
                    continue

                full_noise = torch.randn(
                    batch_size * num_valid_frames, *denoised_pred.shape[2:],
                    dtype=denoised_pred.dtype).to(noise.device)

                block_pred = denoised_pred[
                    :, local_offset * nfpb:(local_offset + 1) * nfpb
                ].flatten(0, 1)
                block_noise = full_noise.unflatten(
                    0, (batch_size, num_valid_frames)
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

            # Yield finalized blocks — a block is finalized once it has passed
            # through all nds denoising steps in the rolling window
            finalized_block = window_index - nds + 1
            if finalized_block > last_finalized_block and finalized_block >= 0:
                for blk in range(last_finalized_block + 1, finalized_block + 1):
                    if blk < num_blocks:
                        sf = blk * nfpb
                        ef = min((blk + 1) * nfpb, requested_frames)
                        if sf < requested_frames:
                            yield (sf, output[:, sf:ef].clone().cpu())
                last_finalized_block = finalized_block

        # Yield any remaining blocks from the ramp-down phase
        for blk in range(last_finalized_block + 1, num_blocks):
            sf = blk * nfpb
            ef = min((blk + 1) * nfpb, requested_frames)
            if sf < requested_frames:
                yield (sf, output[:, sf:ef].clone().cpu())

    def _build_timestep_patterns(self):
        """Build unique timestep patterns for all window types (CPU, float32).

        Returns a [2*nds-1, max_frames] tensor where:
          - Pattern 0: steady-state (full window)
          - Patterns 1..nds-1: ramp-up (window growing from start)
          - Patterns nds..2*nds-2: ramp-down (window shrinking from end)
        """
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
        """Map a single timestep value to its corresponding sigma."""
        idx = torch.argmin((self.scheduler.timesteps - timestep_val).abs())
        return self.scheduler.sigmas[idx].item()

    def _build_sigma_patterns(self):
        """Precompute sigma values for each timestep pattern.

        Mirrors _build_timestep_patterns layout: [2*nds-1, max_frames].
        """
        sigma_patterns = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern):
                sigma_patterns[i, j] = self._timestep_to_sigma(t.item())
        return sigma_patterns

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize per-layer KV cache with Python int indices."""
        kv_cache_clean = []
        kv_cache_alloc_size = self.frame_seq_length * 24   # 37440
        # Buffer must be large enough for both:
        #  1. max attention window (frame_seq_length * 21)
        #  2. eviction copy (kv_cache_alloc_size - block_length)
        block_length = self.num_frame_per_block * self.frame_seq_length
        eviction_copy_size = kv_cache_alloc_size - block_length
        max_buffer_size = max(self.frame_seq_length * 21, eviction_copy_size)
        max_buffer_size = (max_buffer_size + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_alloc_size, self._num_heads, self._head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, kv_cache_alloc_size, self._num_heads, self._head_dim],
                    dtype=dtype, device=device),
                "global_end_index": 0,
                "local_end_index": 0,
            })

        self.kv_cache_clean = kv_cache_clean
        self.shared_buffer_k = torch.zeros(
            [batch_size, max_buffer_size, self._num_heads, self._head_dim],
            dtype=dtype, device=device)
        self.shared_buffer_v = torch.zeros(
            [batch_size, max_buffer_size, self._num_heads, self._head_dim],
            dtype=dtype, device=device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """Initialize per-layer cross-attention cache."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros(
                    [batch_size, self._text_len, self._num_heads, self._head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, self._text_len, self._num_heads, self._head_dim],
                    dtype=dtype, device=device),
                "is_init": False,
            })
        self.crossattn_cache = crossattn_cache
