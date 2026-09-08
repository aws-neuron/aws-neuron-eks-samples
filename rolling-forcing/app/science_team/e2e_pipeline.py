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
import time

import torch
import torch.distributed as dist

from models.dit_pipeline import (
    build_dit_pipeline,
    destroy_parallel_groups,
    init_parallel_groups,
)
from models.t5 import (
    build_text_encoder,
    destroy_t5_parallel_group,
    encode_one_prompt,
    init_t5_parallel_group,
)
from models.vae import (
    build_vae,
    destroy_vae_parallel_group,
    init_vae_parallel_group,
)
from utils import w_shard
from utils.logging_utils import configure_logging, get_logger
from utils.rng import restore_cpu_rng
from utils.video import gather_and_save

configure_logging()
logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end T2V streaming pipeline")
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--num_output_frames", type=int, default=126)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--tp_degree", type=int, default=4,
                   help="DiT tensor-parallel degree; sp = world / tp.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rng_state_path", type=str, default=None,
                   help="Directory of per-prompt CPU RNG states, or a single .pt file.")
    p.add_argument("--output_folder", type=str, default="videos")
    p.add_argument("--chunk-size", type=int, default=3,
                   help="VAE streaming chunk size (frames per decode call).")
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def stream_decode_prompt(pipe, vae, prompt_embeds, noise, rank, world,
                         profile=False):
    video_chunks = []
    gen = pipe.inference_rolling_forcing_stream(
        noise, {"prompt_embeds": prompt_embeds})

    if profile:
        torch.neuron.synchronize()
        t = time.perf_counter()

    for chunk_idx, chunk in enumerate(gen):
        if profile:
            torch.neuron.synchronize()
            dit_ms = (time.perf_counter() - t) * 1000
            t = time.perf_counter()

        chunk_latent = w_shard(chunk, rank, world)
        chunk_device = vae.decode_to_pixel_device(
            chunk_latent, use_cache=True, chunk_idx=chunk_idx)

        if profile:
            torch.neuron.synchronize()
            vae_ms = (time.perf_counter() - t) * 1000

        chunk_video = vae.postprocess_pixels(chunk_device)
        video_chunks.append(chunk_video)

        if profile:
            frames = chunk_video.shape[1]
            block_ms = dit_ms + vae_ms
            fps = frames * 1000.0 / block_ms if block_ms > 0 else 0.0
            logger.info("  block %2d: DiT %7.1f ms  VAE %6.1f ms  %2d frames  %5.2f fps",
                        chunk_idx, dit_ms, vae_ms, frames, fps)
            t = time.perf_counter()

    return torch.cat(video_chunks, dim=1)


def main():
    args = parse_args()

    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()

    assert world % args.tp_degree == 0
    sp_degree = world // args.tp_degree

    init_t5_parallel_group()
    init_parallel_groups(sp_degree, args.tp_degree)
    init_vae_parallel_group()

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    with open(args.prompt_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompt_file)

    logger.info("Building T5 text encoder...")
    text_encoder = build_text_encoder(device="neuron")
    logger.info("Building DiT pipeline...")
    pipe = build_dit_pipeline(
        args.config_path, args.checkpoint_path, args.tp_degree, args.use_ema,
    )
    logger.info("Building VAE decoder...")
    vae = build_vae(dtype=torch.bfloat16)

    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    dist.barrier()

    profile = os.environ.get("PROFILE_E2E_PIPELINE", "0") == "1"

    for prompt_idx, prompt in enumerate(prompts):
        sample_name = f"prompt_{prompt_idx:03d}.pt"
        restore_cpu_rng(args.rng_state_path, sample_name=sample_name,
                        verbose=(rank == 0))
        noise = torch.randn(
            1, args.num_output_frames, 16, 60, 104, dtype=torch.bfloat16,
        ).to("neuron")
        vae.model.clear_cache()

        if profile:
            logger.info("[prompt %3d/%d] %s...",
                        prompt_idx, len(prompts), prompt[:60])

        if profile:
            torch.neuron.synchronize()
            t = time.perf_counter()
        prompt_embeds = encode_one_prompt(text_encoder, prompt)
        if profile:
            torch.neuron.synchronize()
            t5_ms = (time.perf_counter() - t) * 1000
            logger.info("  T5:          %7.1f ms", t5_ms)

        video_local = stream_decode_prompt(
            pipe, vae, prompt_embeds, noise, rank, world, profile=profile)

        out_path = os.path.join(args.output_folder, f"prompt_{prompt_idx:03d}.mp4")
        gather_and_save(video_local, out_path, args.fps, rank, world)

    destroy_t5_parallel_group()
    destroy_parallel_groups()
    destroy_vae_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
