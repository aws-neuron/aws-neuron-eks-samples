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

import argparse
import os

import torch
import torch.distributed as dist

from models.dit_pipeline import (
    build_dit_pipeline,
    init_parallel_groups,
    destroy_parallel_groups,
)
from utils.logging_utils import configure_logging, get_logger
from utils.rng import restore_cpu_rng

configure_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_output_frames", type=int, default=21)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rng_state_path", type=str, default=None,
                        help="Path to cpu_rng_states/ directory or a single .pt file")
    parser.add_argument("--tp_degree", type=int, default=1,
                        help="Tensor-parallel degree for self-attention. "
                             "sp_degree is derived as world_size // tp_degree.")
    args = parser.parse_args()

    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info("%s", args)

    tp_degree = args.tp_degree
    assert world_size % tp_degree == 0, (
        f"world_size {world_size} not divisible by tp_degree {tp_degree}")
    sp_degree = world_size // tp_degree

    init_parallel_groups(sp_degree, tp_degree)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    pipe = build_dit_pipeline(
        args.config_path, args.checkpoint_path, tp_degree, args.use_ema,
    )

    prompt_embeds = torch.load(args.embedding_path, map_location="cpu").to(torch.bfloat16)
    assert prompt_embeds.dim() == 3, (
        f"Expected [B, 512, 4096], got {prompt_embeds.shape}")

    restore_cpu_rng(args.rng_state_path,
                    sample_name=os.path.basename(args.embedding_path),
                    verbose=(rank == 0))
    logger.info("CPU RNG state hash: %s",
                hash(torch.random.get_rng_state().numpy().tobytes()))

    noise = torch.randn(
        1, args.num_output_frames, 16, 60, 104, dtype=torch.bfloat16
    ).to("neuron")
    conditional_dict = {"prompt_embeds": prompt_embeds.to("neuron")}

    latents = pipe.inference_rolling_forcing(noise, conditional_dict).cpu()

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        torch.save(latents, args.output_path)
        logger.info("Saved latents %s %s to %s",
                    latents.shape, latents.dtype, args.output_path)

    destroy_parallel_groups()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
