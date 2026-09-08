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

import types
from typing import List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging as diffusers_logging

from utils import _compile
from utils import parallel_state as ps
from utils.scheduler import SchedulerInterface, FlowMatchScheduler

from models.dit_layers import (
    CausalHead,
    WanPatchEmbed,
    convert_flow_pred_to_x0,
    rope_params,
    sinusoidal_embedding_1d,
    unpatchify,
)
from models.dit_attention import (
    CausalWanAttentionBlock,
    expand_e_shard,
)


def _get_attention_block_cls():
    return CausalWanAttentionBlock


def _init_rope_freqs(dim, num_heads):
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads
    cos_0, sin_0 = rope_params(1024, d - 4 * (d // 6))
    cos_1, sin_1 = rope_params(1024, 2 * (d // 6))
    cos_2, sin_2 = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_0, cos_1, cos_2], dim=1), torch.cat([sin_0, sin_1, sin_2], dim=1)


class CausalWanModel(ModelMixin, ConfigMixin):

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['CausalWanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()

        assert model_type == 't2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.world_size = ps.get_world_size("world") if ps.is_registered("world") else 1

        self.patch_embedding = WanPatchEmbed(in_dim, dim, patch_size)
        self.text_embedding = _compile(nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim)))
        self.time_embedding = _compile(nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)))
        self.time_projection = _compile(nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6)))

        AttentionBlock = _get_attention_block_cls()
        self.blocks = nn.ModuleList([
            AttentionBlock(
                't2v_cross_attn', dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm,
                eps, layer_idx)
            for layer_idx in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        self._sinusoidal_embedding_1d = _compile(sinusoidal_embedding_1d)
        self._unpatchify = _compile(unpatchify)

        if self.world_size > 1:
            self._expand_e_shard_neuron = _compile(expand_e_shard)

        self.freqs_cos, self.freqs_sin = _init_rope_freqs(dim, num_heads)

    def _forward_inference(
        self,
        x,
        t,
        context,
        updating_cache=False,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        num_valid_frames: int = None,
        shared_buffers=None,
        mode: str = "denoise",
        cache_update_start: int = None,
        cu_shared_buffers=None,
        nfpb_cu: int = None,
    ):
        assert self.model_type == 't2v'
        assert x.shape[0] == 1
        assert not torch.is_grad_enabled()

        device = self.patch_embedding.weight.device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)

        def _get_grid_sizes(x):
            F, H, W = x.shape[2:]
            pT, pH, pW = self.patch_embedding.patch_size
            return (F // pT, H // pH, W // pW)
        grid_sizes = _get_grid_sizes(x)
        x = self.patch_embedding(x)

        if self.world_size > 1:
            L = x.shape[1]
            assert L % self.world_size == 0, (
                f"sequence length {L} not divisible by world_size {self.world_size}")
            shard_len = L // self.world_size
            rank = ps.get_rank("world")
            x = x[:, rank * shard_len:(rank + 1) * shard_len].contiguous()

        e = self.time_embedding(
            self._sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        assert context.size(1) == self.text_len
        context = self.text_embedding(context)

        rope_grid_cache = {}

        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos,
            freqs_sin=self.freqs_sin,
            context=context,
            context_lens=context_lens,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
            mode=mode,
            cache_update_start=cache_update_start,
            cu_shared_buffers=cu_shared_buffers,
            nfpb_cu=nfpb_cu,
            rope_grid_cache=rope_grid_cache,
        )

        if self.world_size > 1:
            num_frames = e0.shape[1]
            frame_seqlen = grid_sizes[1] * grid_sizes[2]
            L_full = num_frames * frame_seqlen
            shard_len_e = L_full // self.world_size
            rank = ps.get_rank("world")
            sp_start = rank * shard_len_e
            sp_end = sp_start + shard_len_e
            start_frame = sp_start // frame_seqlen
            end_frame = (sp_end - 1) // frame_seqlen + 1
            start_off = sp_start - start_frame * frame_seqlen
            kwargs["e"] = self._expand_e_shard_neuron(
                e0, start_frame, end_frame, start_off, shard_len_e, frame_seqlen)

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start,
                }
            )
            x = block(x, **kwargs)

        if self.world_size > 1:
            B, shard_len, C = x.shape
            full = torch.empty(B * self.world_size * shard_len, C,
                               dtype=x.dtype, device=x.device)
            ps.all_gather_into_tensor(full, x.reshape(-1, C), "world")
            x = full.reshape(B, self.world_size * shard_len, C)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = x.flatten(1, 2)
        result = self._unpatchify(x, self.out_dim, self.patch_size, grid_sizes).unsqueeze(0)
        return result

    def forward(self, *args, **kwargs):
        assert kwargs.get('kv_cache', None) is not None
        return self._forward_inference(*args, **kwargs)


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            num_layers=None,
    ):
        super().__init__()

        assert is_causal
        tp_degree = ps.get_world_size("attn-tp") if ps.is_registered("attn-tp") else 1
        kwargs = dict(
            local_attn_size=local_attn_size, sink_size=sink_size,
            torch_dtype=torch.bfloat16,
        )
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        if tp_degree > 1:
            kwargs["ignore_mismatched_sizes"] = True
            _prev_verbosity = diffusers_logging.get_verbosity()
            diffusers_logging.set_verbosity_error()
        self.model = CausalWanModel.from_pretrained(
            f"wan_models/{model_name}/", **kwargs)
        if tp_degree > 1:
            diffusers_logging.set_verbosity(_prev_verbosity)

        self.model.eval()
        self._convert_flow_pred_to_x0 = _compile(convert_flow_pred_to_x0)

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000)

        self.post_init()

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        updating_cache: Optional[bool] = False,
        num_valid_frames: Optional[int] = None,
        shared_buffers=None,
        sigma: Optional[torch.Tensor] = None,
        mode: str = "denoise",
        cache_update_start: Optional[int] = None,
        cu_shared_buffers=None,
        nfpb_cu: Optional[int] = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        assert kv_cache is not None

        x = noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous()

        flow_pred = self.model(
            x,
            t=timestep, context=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
            mode=mode,
            cache_update_start=cache_update_start,
            cu_shared_buffers=cu_shared_buffers,
            nfpb_cu=nfpb_cu,
        )

        flow_pred = flow_pred.permute(0, 2, 1, 3, 4).contiguous()

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            sigma_t=sigma.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
