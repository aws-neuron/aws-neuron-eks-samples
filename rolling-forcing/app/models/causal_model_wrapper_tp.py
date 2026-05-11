"""TP-aware WanDiffusionWrapper for Wan2.1-T2V-14B on Trainium.

Loads the full 14B model, applies tensor parallelism sharding, and provides
the same interface as the single-rank WanDiffusionWrapper.

Architecture:
  - All 4 NeuronCores hold the full TE + VAE (replicated, small)
  - DiT is sharded across 4 cores via TP (each core has 10/40 heads)
  - KV cache sized for local heads (10 heads × 128 head_dim per rank)
"""

import os
import time
import types
from typing import List, Optional

import torch

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn):
    """No-op wrapper since torch_neuronx.jit is not available."""
    return fn

from models.causal_model_tp import CausalWanModelTP
from models.layers import convert_flow_pred_to_x0
from models.tp_utils import (
    init_tp_group,
    get_tp_rank,
    get_tp_world_size,
    shard_model_tp,
)
from utils.scheduler import SchedulerInterface, FlowMatchScheduler


class WanDiffusionWrapperTP(torch.nn.Module):
    """TP-aware diffusion model wrapper.

    Loads full 14B weights, then shards across tp_degree ranks.
    Each rank operates on num_heads//tp_degree heads independently
    with all-reduce communication after O-proj and FFN-down.
    """

    def __init__(
            self,
            model_name="Wan2.1-T2V-14B",
            timestep_shift=5.0,
            is_causal=True,
            local_attn_size=-1,
            sink_size=0,
            num_layers=None,
            frame_length=1560,
            num_frame_per_block=3,
            tp_degree=4,
    ):
        super().__init__()
        assert is_causal

        self.tp_degree = tp_degree
        self.tp_rank = get_tp_rank()

        # Load model with full weights (all ranks load independently)
        print(f"[Rank {self.tp_rank}] Loading {model_name} weights...")
        kwargs = dict(
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            torch_dtype=torch.bfloat16,
            frame_length=frame_length,
        )
        if num_layers is not None:
            kwargs["num_layers"] = num_layers

        self.model = CausalWanModelTP.from_pretrained(
            f"wan_models/{model_name}/", **kwargs)
        self.model._update_frame_length(frame_length, num_frame_per_block)

        # Apply TP sharding — splits weights across ranks
        print(f"[Rank {self.tp_rank}] Applying TP sharding (tp_degree={tp_degree})...")
        shard_model_tp(self.model, self.tp_rank, tp_degree)
        self.model.eval()

        # After sharding: num_heads_per_rank = num_heads // tp_degree
        self.num_heads_per_rank = self.model.num_heads_per_rank
        self.head_dim = self.model.dim // self.model.num_heads  # 128

        self._convert_flow_pred_to_x0 = jit(convert_flow_pred_to_x0)

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000)

        self.post_init()
        print(f"[Rank {self.tp_rank}] WanDiffusionWrapperTP ready: "
              f"{self.num_heads_per_rank} heads/rank, head_dim={self.head_dim}")

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        updating_cache: Optional[bool] = False,
        num_valid_frames: Optional[int] = None,
        shared_buffers=None,
        sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass — same interface as non-TP wrapper.

        All ranks execute in lockstep. TP communication is handled internally.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]
        assert kv_cache is not None

        x = noisy_image_or_video.permute(0, 2, 1, 3, 4)

        flow_pred = self.model(
            x,
            t=timestep,
            context=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers
        )

        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

        # Convert flow prediction to x0 (local computation, no comm needed)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            sigma.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """Bind scheduler interface methods."""
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
        """Post-initialization: bind scheduler methods."""
        self.get_scheduler()

    def load_distilled_weights(self, checkpoint_path: str, use_ema: bool = True):
        """Load distilled (RollingForcing DMD) weights into the sharded model.

        The checkpoint contains full (unsharded) weights. This method:
        1. Loads the full state dict
        2. For each TP-sharded layer, extracts the local rank's shard
        3. Loads into the already-sharded model

        Args:
            checkpoint_path: Path to the DMD checkpoint (.pt file)
            use_ema: Whether to use EMA weights from checkpoint
        """
        from collections import OrderedDict

        print(f"[Rank {self.tp_rank}] Loading distilled weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        if use_ema:
            state_dict_to_load = state_dict.get('generator_ema', state_dict)
        else:
            state_dict_to_load = state_dict.get('generator', state_dict)

        # Remove FSDP prefix and "model." prefix if present.
        # The checkpoint was saved from generator (WanDiffusionWrapper) which has self.model,
        # so keys look like "model.blocks.0...". But _load_sharded_state_dict iterates
        # self.model.state_dict() which has keys like "blocks.0..." (no "model." prefix).
        cleaned = OrderedDict()
        for key, value in state_dict_to_load.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")
            # Strip "model." prefix to match self.model.state_dict() keys
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            cleaned[new_key] = value

        # Diagnostic: print key comparison
        model_keys = list(self.model.state_dict().keys())
        ckpt_keys = list(cleaned.keys())
        print(f"[Rank {self.tp_rank}] Checkpoint keys (first 5): {ckpt_keys[:5]}")
        print(f"[Rank {self.tp_rank}] Model keys (first 5): {model_keys[:5]}")
        matched = sum(1 for k in model_keys if k in cleaned)
        print(f"[Rank {self.tp_rank}] Key match: {matched}/{len(model_keys)} model keys found in checkpoint")

        self._load_sharded_state_dict(cleaned)
        print(f"[Rank {self.tp_rank}] Distilled weights loaded successfully")

    def _load_sharded_state_dict(self, full_state_dict: dict):
        """Load a full (unsharded) state dict into the already-sharded model.

        For column-parallel layers (Q/K/V, fc1): slice rows from full weight
        For row-parallel layers (O, fc2): slice columns from full weight
        For replicated layers: load directly
        """
        tp_rank = self.tp_rank
        tp_degree = self.tp_degree
        model_state = self.model.state_dict()

        new_state_dict = {}
        for key, param in model_state.items():
            if key not in full_state_dict:
                print(f"  [Rank {tp_rank}] WARNING: {key} not in checkpoint, skipping")
                new_state_dict[key] = param
                continue

            full_param = full_state_dict[key]

            # Determine if this is a sharded parameter by comparing shapes
            if param.shape == full_param.shape:
                # Replicated parameter — load directly
                new_state_dict[key] = full_param
            elif param.shape[0] < full_param.shape[0] and (
                    len(param.shape) == len(full_param.shape)):
                # Column-parallel: output dim sharded (Q/K/V weight, fc1 weight, biases)
                chunk_size = param.shape[0]
                start = tp_rank * chunk_size
                end = start + chunk_size
                new_state_dict[key] = full_param[start:end].contiguous()
            elif len(param.shape) >= 2 and param.shape[1] < full_param.shape[1]:
                # Row-parallel: input dim sharded (O weight, fc2 weight)
                chunk_size = param.shape[1]
                start = tp_rank * chunk_size
                end = start + chunk_size
                new_state_dict[key] = full_param[:, start:end].contiguous()
            else:
                # Fallback: shapes don't match expectations
                print(f"  [Rank {tp_rank}] WARNING: Shape mismatch for {key}: "
                      f"model={param.shape}, checkpoint={full_param.shape}")
                new_state_dict[key] = full_param

        # Load the sharded state dict
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  [Rank {tp_rank}] Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  [Rank {tp_rank}] Unexpected keys: {unexpected[:5]}...")
