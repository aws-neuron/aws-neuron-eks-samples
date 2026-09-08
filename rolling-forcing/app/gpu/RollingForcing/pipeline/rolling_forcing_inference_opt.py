# Optimized version of rolling_forcing_inference.py:
# - Static input shapes (padded to max_frames with q_lens/k_lens masking)
# - Both generator calls use identical shape [B, max_frames, C, H, W]
# - Redundant re-noising eliminated (add_noise only on needed block slice)
# - Python int KV cache indices
# - Debug prints removed
import os
import sys
from typing import List, Optional
import torch

from utils.wan_wrapper_opt import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

# When True, generate random tensors on CPU then move to device (matches Neuron behavior).
# When False, generate random tensors directly on device (original GPU behavior).
RAND_ON_CPU = False


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
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_clean = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.timestep_patterns = self._build_timestep_patterns()
        self.sigma_patterns = self._build_sigma_patterns()
        self.context_sigma = self._timestep_to_sigma(self.args.context_noise)

    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape

        assert not self.independent_first_frame
        assert initial_latent is None
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache (Python int indices)
            for block_index in range(len(self.kv_cache_clean)):
                self.kv_cache_clean[block_index]["global_end_index"] = 0
                self.kv_cache_clean[block_index]["local_end_index"] = 0

        # Step 2: Cache context feature
        assert initial_latent is None

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # implementing rolling forcing
        # construct the rolling forcing windows
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
                pattern_indices.append(0)                  # steady-state
            elif start_block == 0:
                pattern_indices.append(num_blks)            # ramp-up
            else:
                pattern_indices.append(nds - 1 + num_blks)  # ramp-down

        # Static shape constants
        max_frames = rolling_window_length_blocks * self.num_frame_per_block
        nfpb = self.num_frame_per_block

        output = torch.zeros(
            [batch_size, num_output_frames + max_frames - nfpb, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # init noisy cache (padded with max_frames extra for OOB-safe static reads)
        noisy_cache = torch.zeros(
            [batch_size, num_output_frames + max_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Move pre-computed patterns to device (lazy, once)
        if self.timestep_patterns.device != noise.device:
            self.timestep_patterns = self.timestep_patterns.to(noise.device)
            self.sigma_patterns = self.sigma_patterns.to(noise.device)

        # Pre-allocate padded tensors (reused every iteration for static shape)
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
            [batch_size, nfpb], self.args.context_noise,
            device=noise.device, dtype=torch.float32)
        cache_sigma = torch.full(
            [batch_size, nfpb], self.context_sigma,
            device=noise.device, dtype=torch.float32)

        # Pre-allocate block sigma tensors for each denoising step (reused in re-noising loop)
        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                (sigma_val * torch.ones([batch_size * nfpb, 1, 1, 1], dtype=torch.float32)).to(noise.device))

        record_ops = os.environ.get("LOG_OP", "0") == "1"
        if record_ops:
            log_folder = f"./op_logs"
            summary_folder = f"./op_logs/summary"
            detail_folder = f"./op_logs/detail"
            os.makedirs(log_folder, exist_ok=True)
            os.makedirs(summary_folder, exist_ok=True)
            os.makedirs(detail_folder, exist_ok=True)
            from op_logger import OpLogger
            def range_with_log_ctx(*args):
                for i in range(*args):
                    op_logger = OpLogger()
                    with op_logger:
                        yield i
                    summary_log = f"{summary_folder}/window_{i}.txt"
                    with open(summary_log, "w") as f:
                        original_stdout = sys.stdout
                        sys.stdout = f
                        op_logger.print_summary()
                        sys.stdout = original_stdout
                    detail_log = f"{detail_folder}/window_{i}.txt"
                    op_logger.dump_log(detail_log)
        else:
            range_with_log_ctx = range

        # Denoising loop with rolling forcing
        for window_index in range_with_log_ctx(window_num):

            if profile:
                block_start.record()

            print('window_index:', window_index)
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index]  # include
            print(f"start_block: {start_block}, end_block: {end_block}")

            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb  # not include
            current_num_frames = current_end_frame - current_start_frame

            # Static copy: always max_frames from noisy_cache (OOB-safe due to padding)
            padded_input.copy_(noisy_cache[:, current_start_frame : current_start_frame + max_frames])

            # Overwrite nfpb frames with fresh noise (static length = nfpb)
            # Ramp-up/steady-state: a new block enters at the end, needs fresh noise
            # Ramp-down: no new block, all data from noisy_cache (no overwrite)
            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset : noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb : current_end_frame])

            # Static timestep/sigma copy: always max_frames from pre-allocated patterns
            padded_timestep[:] = self.timestep_patterns[pattern_indices[window_index]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[window_index]]

            num_valid_frames = current_num_frames

            # calling DiT with static shape
            _, denoised_pred = self.generator(
                    noisy_image_or_video=padded_input,
                    conditional_dict=conditional_dict,
                    timestep=padded_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    num_valid_frames=num_valid_frames,
                    shared_buffers=(self.shared_buffer_k, self.shared_buffer_v),
                    sigma=padded_sigma
                )

            # Static-length copy to output (garbage in padding zone, overwritten by later windows)
            output[:, current_start_frame:current_start_frame + max_frames].copy_(denoised_pred)

            # Re-noising: partially denoise each block to its next noise level.
            # Each block in the window is at a different denoising stage. The
            # oldest block (local_offset=0) is most denoised; the newest is
            # least. step_index gives each block's current denoising step:
            #   - Ramp-up (start_block==0, window not yet full): step_base = num_blks - 1
            #   - Steady-state / ramp-down: step_base = nds - 1
            #   step_index = step_base - local_offset
            # Blocks at step_index == nds-1 are fully noisy and need no re-noising.
            with torch.no_grad():
                num_blks = end_block - start_block + 1
                step_base = (num_blks - 1) if (start_block == 0 and num_blks < nds) else (nds - 1)

                for block_idx in range(start_block, end_block + 1):
                    local_offset = block_idx - start_block
                    step_index = step_base - local_offset

                    # Skip: this block is at the noisiest level, no re-noising needed
                    if step_index == nds - 1:
                        continue

                    # Draw noise with the same shape as the baseline (num_valid_frames,
                    # not max_frames) to preserve CUDA RNG draw count for bitwise match.
                    # noise_template is an empty tensor used only for shape/device/dtype.
                    noise_shape = (batch_size * num_valid_frames, *denoised_pred.shape[2:])
                    if RAND_ON_CPU:
                        full_noise = torch.randn(
                            noise_shape, dtype=denoised_pred.dtype).to(denoised_pred.device)
                    else:
                        full_noise = torch.randn(
                            noise_shape, device=denoised_pred.device, dtype=denoised_pred.dtype)

                    # Slice out only this block's prediction and noise
                    block_pred = denoised_pred[:, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                    block_noise = full_noise.unflatten(0, (batch_size, num_valid_frames))[
                        :, local_offset * nfpb:(local_offset + 1) * nfpb].flatten(0, 1)
                    # Use pre-allocated sigma tensor for the next denoising step
                    block_sigma = block_sigma_list[step_index + 1]

                    noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                        add_noise(block_pred, block_noise, block_sigma) \
                        .unflatten(0, (batch_size, nfpb))

            # Rerun with context noise to update the clean cache (3-frame input, no padding)
            with torch.no_grad():
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
                    sigma=cache_sigma
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)


        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()


        # Step 4: Trim padding and decode the output
        output = output[:, :num_output_frames]
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video



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

        # Steady-state: denoising steps in reverse, each repeated nfpb times
        steady = []
        for ts in reversed(self.denoising_step_list):
            steady.extend([ts.item()] * nfpb)

        # 2*nds - 1 patterns: [0] steady, [1..nds-1] ramp-up, [nds..2*nds-2] ramp-down
        patterns = [steady]
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[-cnf:] + [0.0] * (max_frames - cnf))
        for i in range(1, nds):
            cnf = i * nfpb
            patterns.append(steady[:cnf] + [0.0] * (max_frames - cnf))

        return torch.tensor(patterns, dtype=torch.float32)

    def _timestep_to_sigma(self, timestep_val):
        """Map a single timestep float value to its corresponding sigma.

        The scheduler stores paired tables: timesteps[i] and sigmas[i].
        Given a timestep value, find the closest entry and return its sigma.
        This is the same argmin lookup that _convert_flow_pred_to_x0 used to
        do per-step; here we do it once at init time.
        """
        idx = torch.argmin((self.scheduler.timesteps - timestep_val).abs())
        return self.scheduler.sigmas[idx].item()

    def _build_sigma_patterns(self):
        """Precompute sigma values for each timestep pattern.

        Mirrors _build_timestep_patterns layout: [2*nds-1, max_frames].
        Eliminates the per-step argmin lookup in _convert_flow_pred_to_x0.
        """
        sigma_patterns = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern):
                sigma_patterns[i, j] = self._timestep_to_sigma(t.item())
        return sigma_patterns

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        Uses Python ints for indices (no GPU scalar tensors).
        KV cache tensors are padded beyond logical size to allow static-length
        copies that read past valid data (garbage masked by k_lens).
        A single shared buffer_k/buffer_v pair is used across all layers.
        """
        kv_cache_clean = []
        kv_cache_alloc_size = 1560 * 24   # 37440
        max_buffer_size = 1560 * 21       # 32760: max_attention_size
        ATTN_SEQLEN_MULTIPLE = 8192
        max_buffer_size = (max_buffer_size + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, kv_cache_alloc_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_alloc_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": 0,
                "local_end_index": 0,
            })

        self.kv_cache_clean = kv_cache_clean

        # Shared buffers across all layers: used as scratch during eviction,
        # then as assembled KV input to attention
        self.shared_buffer_k = torch.zeros([batch_size, max_buffer_size, 12, 128], dtype=dtype, device=device)
        self.shared_buffer_v = torch.zeros([batch_size, max_buffer_size, 12, 128], dtype=dtype, device=device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
