# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Neuron Science Team, Amazon Annapurna Labs
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils import _compile
from utils import parallel_state as ps
from utils.logging_utils import get_logger
from utils.noise_producer import NoiseProducer

from models.dit_attention import CausalWanSelfAttention
from models.dit_layers import ATTN_SEQLEN_MULTIPLE
from models.dit_model import WanDiffusionWrapper

logger = get_logger(__name__)


def init_parallel_groups(sp_degree, tp_degree):
    world_size = dist.get_world_size()
    assert sp_degree * tp_degree == world_size, (
        f"sp_degree * tp_degree ({sp_degree * tp_degree}) must equal "
        f"world_size ({world_size})")

    rank = dist.get_rank()
    sp_rank = rank // tp_degree
    tp_rank = rank % tp_degree

    ps.register_group("world", dist.group.WORLD)

    tp_group = None
    for sp_i in range(sp_degree):
        ranks = list(range(sp_i * tp_degree, (sp_i + 1) * tp_degree))
        grp = dist.new_group(ranks)
        if sp_i == sp_rank:
            tp_group = grp
    ps.register_group("attn-tp", tp_group)

    sp_group = None
    for tp_i in range(tp_degree):
        ranks = list(range(tp_i, world_size, tp_degree))
        grp = dist.new_group(ranks)
        if tp_i == tp_rank:
            sp_group = grp
    ps.register_group("attn-sp", sp_group)


def destroy_parallel_groups():
    ps.destroy_group("attn-tp")
    ps.destroy_group("attn-sp")
    ps.destroy_group("world")


def add_noise(original_samples, noise, sigma):
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


def _shard_full_state_dict(full_sd, num_blocks, dim, num_heads):
    sharded = OrderedDict()
    pat = re.compile(r"^(.*?blocks\.(\d+)\.self_attn\.)(.*)$")

    per_block_sub = {i: {} for i in range(num_blocks)}
    block_prefix = {}
    passthrough = OrderedDict()
    for key, val in full_sd.items():
        m = pat.match(key)
        if m is None:
            passthrough[key] = val
            continue
        per_block_sub[int(m.group(2))][m.group(3)] = val
        block_prefix[int(m.group(2))] = m.group(1)

    for block_idx, sub_sd in per_block_sub.items():
        if not sub_sd:
            continue
        shard_sub = CausalWanSelfAttention.shard_state_dict(
            sub_sd, dim, num_heads)
        prefix = block_prefix[block_idx]
        for k, v in shard_sub.items():
            sharded[prefix + k] = v

    sharded.update(passthrough)
    return sharded


def build_dit_pipeline(config_path, checkpoint_path, tp_degree, use_ema):
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    assert hasattr(config, "denoising_step_list")

    pipe = CausalInferencePipeline(
        denoising_step_list=config.denoising_step_list,
        num_frame_per_block=getattr(config, "num_frame_per_block", 3),
        context_noise=getattr(config, "context_noise", 0.0),
        warp_denoising_step=getattr(config, "warp_denoising_step", True),
        model_name=getattr(config, "model_name", "Wan2.1-T2V-1.3B"),
        timestep_shift=getattr(config, "timestep_shift", 5.0),
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if use_ema:
            state_dict_to_load = state_dict["generator_ema"]
            new_sd = OrderedDict()
            for key, value in state_dict_to_load.items():
                new_sd[key.replace("_fsdp_wrapped_module.", "")] = value
            state_dict_to_load = new_sd
        else:
            state_dict_to_load = state_dict["generator"]

        if tp_degree > 1:
            num_blocks = len(pipe.generator.model.blocks)
            dim = pipe.generator.model.dim
            num_heads = pipe.generator.model.num_heads
            state_dict_to_load = _shard_full_state_dict(
                state_dict_to_load, num_blocks, dim, num_heads)

        model_keys = set(pipe.generator.state_dict().keys())
        remapped = {}
        for k, v in state_dict_to_load.items():
            if k in model_keys:
                remapped[k] = v
                continue
            parts = k.split(".")
            for i in range(len(parts)):
                cand = ".".join(parts[:i+1] + ["_orig_mod"] + parts[i+1:])
                if cand in model_keys:
                    remapped[cand] = v
                    break
            else:
                remapped[k] = v
        pipe.generator.load_state_dict(remapped, strict=True)

    pipe.generator.model = pipe.generator.model.to("neuron")
    return pipe


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            denoising_step_list: List[int],
            num_frame_per_block: int = 3,
            context_noise: float = 0.0,
            warp_denoising_step: bool = True,
            frame_seq_length: int = 1560,
            model_name: str = "Wan2.1-T2V-1.3B",
            timestep_shift: float = 5.0,
            local_attn_size: int = -1,
            sink_size: int = 0,
            num_layers: Optional[int] = None,
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
            )
        self.generator = generator
        self.tp_degree = ps.get_world_size("attn-tp") if ps.is_registered("attn-tp") else 1
        self.sp_degree = ps.get_world_size("attn-sp") if ps.is_registered("attn-sp") else 1
        self.world_size = self.sp_degree * self.tp_degree
        self.rank = ps.get_rank("world") if ps.is_registered("world") else 0

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

        self._num_heads = self.generator.model.num_heads
        self._head_dim = self.generator.model.dim // self.generator.model.num_heads
        self._text_len = self.generator.model.text_len
        self._self_attn_heads = self._num_heads // self.tp_degree

        self.timestep_patterns = self._build_timestep_patterns()
        self.sigma_patterns = self._build_sigma_patterns()
        self.context_sigma = self._timestep_to_sigma(self.context_noise)

        self._add_noise = _compile(add_noise)

        self.kv_cache_clean = None
        self.crossattn_cache = None

    @torch.no_grad()
    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        gen = self._run(noise, conditional_dict, streaming=False)
        final = None
        for x in gen:
            final = x
        return final

    @torch.no_grad()
    def inference_rolling_forcing_stream(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):
        yield from self._run(noise, conditional_dict, streaming=True)

    def _run(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        streaming: bool,
    ):
        profile = os.environ.get("PROFILE_PIPELINE", "0") == "1"

        batch_size, num_frames, num_channels, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        assert num_frames * height * width % self.world_size == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames

        if profile:
            init_start = time.perf_counter()

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
                pattern_indices.append(0)
            elif start_block == 0:
                pattern_indices.append(num_blks)
            else:
                pattern_indices.append(nds - 1 + num_blks)

        max_frames = rolling_window_length_blocks * self.num_frame_per_block
        nfpb = self.num_frame_per_block
        full_frames = nfpb + max_frames

        def build_renoise_plan(phase):
            sb = window_start_blocks[phase]
            eb = window_end_blocks[phase]
            num_blks_p = eb - sb + 1
            step_base = (num_blks_p - 1) if (sb == 0 and num_blks_p < nds) else (nds - 1)
            full_shape = (batch_size * num_blks_p * nfpb,
                          num_channels, height, width)
            plan = []
            for local_offset in range(num_blks_p):
                if (step_base - local_offset) == nds - 1:
                    continue
                if batch_size == 1:
                    sl = slice(local_offset * nfpb, (local_offset + 1) * nfpb)
                else:
                    sl = torch.tensor(
                        [b * num_blks_p * nfpb + local_offset * nfpb + i
                         for b in range(batch_size)
                         for i in range(nfpb)],
                        dtype=torch.long)
                plan.append((full_shape, sl))
            return plan

        if self.world_size > 1:
            cu_L = nfpb * self.frame_seq_length
            dn_L = max_frames * self.frame_seq_length
            assert cu_L % self.world_size == 0, (
                f"merged cu seq_len {cu_L} (nfpb={nfpb} x frame_seq_length="
                f"{self.frame_seq_length}) not divisible by world_size "
                f"{self.world_size}")
            assert dn_L % self.world_size == 0, (
                f"merged dn seq_len {dn_L} (max_frames={max_frames} x "
                f"frame_seq_length={self.frame_seq_length}) not divisible "
                f"by world_size {self.world_size}")

        output = None if streaming else torch.zeros(
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

        padded_input_full = torch.zeros(
            [batch_size, full_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)
        padded_timestep_full = torch.zeros(
            [batch_size, full_frames],
            device=noise.device, dtype=torch.float32)
        padded_sigma_full = torch.zeros(
            [batch_size, full_frames],
            device=noise.device, dtype=torch.float32)
        padded_timestep_full[:, :nfpb] = self.context_noise
        padded_sigma_full[:, :nfpb] = self.context_sigma

        prev_denoised_pred_first_block = torch.zeros(
            [batch_size, nfpb, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        block_sigma_list = []
        for step in self.denoising_step_list:
            sigma_val = self._timestep_to_sigma(step.item())
            block_sigma_list.append(
                sigma_val * torch.ones([batch_size * nfpb, 1, 1, 1],
                                       dtype=torch.float32, device=noise.device))

        if profile:
            init_end = time.perf_counter()
            diffusion_start = time.perf_counter()
            window_times = []

        dn_buffers = (self.shared_buffer_k, self.shared_buffer_v)
        cu_buffers = (self.cu_shared_buffer_k, self.cu_shared_buffer_v)

        noise_producer = NoiseProducer(dtype=noise.dtype)

        for phase in range(window_num):
            if profile:
                window_start = time.perf_counter()

            plan = build_renoise_plan(phase)
            renoise_future = noise_producer.request(plan) if plan else None

            start_block = window_start_blocks[phase]
            end_block = window_end_blocks[phase]
            current_start_frame = start_block * nfpb
            current_end_frame = (end_block + 1) * nfpb
            current_num_frames = current_end_frame - current_start_frame
            num_valid_frames_dn = current_num_frames

            padded_input.copy_(
                noisy_cache[:, current_start_frame:current_start_frame + max_frames])
            if current_num_frames == max_frames or current_start_frame == 0:
                noise_offset = current_num_frames - nfpb
                padded_input[:, noise_offset:noise_offset + nfpb].copy_(
                    noise[:, current_end_frame - nfpb:current_end_frame])
            padded_timestep[:] = self.timestep_patterns[pattern_indices[phase]]
            padded_sigma[:] = self.sigma_patterns[pattern_indices[phase]]

            if phase >= 1:
                cu_start_block = window_start_blocks[phase - 1]
                cache_update_start = cu_start_block * nfpb * self.frame_seq_length

                padded_input_full[:, :nfpb].copy_(prev_denoised_pred_first_block)
                padded_input_full[:, nfpb:].copy_(padded_input)
                padded_timestep_full[:, nfpb:].copy_(padded_timestep)
                padded_sigma_full[:, nfpb:].copy_(padded_sigma)

                _, pred_x0_full = self.generator(
                    noisy_image_or_video=padded_input_full,
                    conditional_dict=conditional_dict,
                    timestep=padded_timestep_full,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    num_valid_frames=num_valid_frames_dn,
                    shared_buffers=dn_buffers,
                    sigma=padded_sigma_full,
                    mode="merged",
                    cache_update_start=cache_update_start,
                    cu_shared_buffers=cu_buffers,
                    nfpb_cu=nfpb,
                )
                denoised_pred = pred_x0_full[:, nfpb:]
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=padded_input,
                    conditional_dict=conditional_dict,
                    timestep=padded_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    num_valid_frames=num_valid_frames_dn,
                    shared_buffers=dn_buffers,
                    sigma=padded_sigma,
                    mode="denoise",
                )

            first_block = denoised_pred[:, :nfpb]
            if phase < window_num - 1:
                prev_denoised_pred_first_block.copy_(first_block)

            if not streaming:
                output[:, current_start_frame:current_start_frame + max_frames].copy_(
                    denoised_pred)

            if renoise_future is not None:
                packed_noise = renoise_future.result().to(noise.device)
                num_blks = end_block - start_block + 1
                step_base = (num_blks - 1) if (
                    start_block == 0 and num_blks < nds) else (nds - 1)
                active_idx = 0
                for block_idx in range(start_block, end_block + 1):
                    local_offset = block_idx - start_block
                    step_index = step_base - local_offset
                    if step_index == nds - 1:
                        continue
                    block_pred = denoised_pred[
                        :, local_offset * nfpb:(local_offset + 1) * nfpb
                    ].flatten(0, 1)
                    block_noise = packed_noise[
                        active_idx * batch_size * nfpb:
                        (active_idx + 1) * batch_size * nfpb]
                    active_idx += 1
                    block_sigma = block_sigma_list[step_index + 1]
                    noisy_cache[:, block_idx * nfpb:(block_idx + 1) * nfpb] = \
                        self._add_noise(block_pred, block_noise, block_sigma) \
                        .unflatten(0, (batch_size, nfpb))

            if profile:
                torch.neuron.synchronize()
                wt = time.perf_counter() - window_start
                window_times.append(wt)
                logger.info(f"Phase {phase}: {wt*1000:.2f} ms")

            if streaming and phase >= nds - 1:
                yield first_block

        noise_producer.shutdown()

        if profile:
            diffusion_end = time.perf_counter()
            init_time = (init_end - init_start) * 1000
            diffusion_time = (diffusion_end - diffusion_start) * 1000
            total_time = init_time + diffusion_time
            logger.info("Profiling results:")
            logger.info(f"  - Initialization time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            logger.info(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, wt in enumerate(window_times):
                wt_ms = wt * 1000
                logger.info(f"    - Phase {i} time: {wt_ms:.2f} ms ({100 * wt_ms / diffusion_time:.2f}% of diffusion)")
            logger.info(f"  - Total time: {total_time:.2f} ms")

        if not streaming:
            yield output[:, :num_output_frames]

    def _build_timestep_patterns(self):
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
        idx = torch.argmin((self.scheduler.timesteps - timestep_val).abs())
        return self.scheduler.sigmas[idx].item()

    def _build_sigma_patterns(self):
        sigma_patterns = torch.zeros_like(self.timestep_patterns)
        for i, pattern in enumerate(self.timestep_patterns):
            for j, t in enumerate(pattern):
                sigma_patterns[i, j] = self._timestep_to_sigma(t.item())
        return sigma_patterns

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache_clean = []
        kv_cache_alloc_size = self.frame_seq_length * 24
        max_buffer_size = self.frame_seq_length * 21
        max_buffer_size = (max_buffer_size + ATTN_SEQLEN_MULTIPLE - 1) // ATTN_SEQLEN_MULTIPLE * ATTN_SEQLEN_MULTIPLE

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_alloc_size, self._self_attn_heads, self._head_dim],
                    dtype=dtype, device=device),
                "v": torch.zeros(
                    [batch_size, kv_cache_alloc_size, self._self_attn_heads, self._head_dim],
                    dtype=dtype, device=device),
                "global_end_index": 0,
                "local_end_index": 0,
            })

        self.kv_cache_clean = kv_cache_clean
        self.shared_buffer_k = torch.zeros(
            [batch_size, max_buffer_size, self._self_attn_heads, self._head_dim],
            dtype=dtype, device=device)
        self.shared_buffer_v = torch.zeros(
            [batch_size, max_buffer_size, self._self_attn_heads, self._head_dim],
            dtype=dtype, device=device)
        self.cu_shared_buffer_k = torch.zeros(
            [batch_size, max_buffer_size, self._self_attn_heads, self._head_dim],
            dtype=dtype, device=device)
        self.cu_shared_buffer_v = torch.zeros(
            [batch_size, max_buffer_size, self._self_attn_heads, self._head_dim],
            dtype=dtype, device=device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
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
