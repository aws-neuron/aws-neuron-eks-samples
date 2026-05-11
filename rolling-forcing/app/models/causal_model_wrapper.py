import os
import time
import types
from typing import List, Optional

import torch

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn):
    """No-op wrapper since torch_neuronx.jit is not available."""
    return fn

from models.causal_model import CausalWanModel
from models.layers import convert_flow_pred_to_x0
from utils.scheduler import SchedulerInterface, FlowMatchScheduler


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            num_layers=None,
            frame_length=1560,
            num_frame_per_block=3,
    ):
        super().__init__()

        assert is_causal
        kwargs = dict(
            local_attn_size=local_attn_size, sink_size=sink_size,
            torch_dtype=torch.bfloat16,
            frame_length=frame_length,
        )
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        self.model = CausalWanModel.from_pretrained(
            f"wan_models/{model_name}/", **kwargs)
        self.model._update_frame_length(frame_length, num_frame_per_block)
        self.model.eval()
        self._convert_flow_pred_to_x0 = jit(convert_flow_pred_to_x0)

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
        sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _profile = os.environ.get("PROFILE_PIPELINE", "0") == "1" and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) == 0
        prompt_embeds = conditional_dict["prompt_embeds"]

        assert kv_cache is not None

        if _profile:
            _t = time.perf_counter()
        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        if _profile:
            _t_permute_in = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()

        flow_pred = self.model(
            x,
            t=timestep, context=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers
        )
        if _profile:
            _t_model = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()

        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
        if _profile:
            _t_permute_out = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()

        # Positional args required: DeviceKernel CPU pass-through only checks
        # positional inputs for device, keyword-only args bypass the check.
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            sigma.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        if _profile:
            _t_convert = (time.perf_counter() - _t) * 1000
            print(f"    [wrapper] permute_in={_t_permute_in:.1f}ms  model={_t_model:.1f}ms  permute_out={_t_permute_out:.1f}ms  convert_x0={_t_convert:.1f}ms  total={_t_permute_in+_t_model+_t_permute_out+_t_convert:.1f}ms")

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
