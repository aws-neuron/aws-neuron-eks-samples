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

from models.vae import build_vae, destroy_vae_parallel_group, init_vae_parallel_group
from utils import w_shard
from utils.logging_utils import configure_logging, get_logger
from utils.video import gather_and_save

configure_logging()
logger = get_logger(__name__)


def decode_latents(vae, latents_cpu, device, stream, chunk_size, profile=False):
    num_frames = latents_cpu.shape[1]
    if not stream:
        return vae.postprocess_pixels(
            vae.decode_to_pixel_device(latents_cpu.to(device), use_cache=False))

    vae.model.clear_cache()
    video_chunks = []

    if profile:
        torch.neuron.synchronize()
        t = time.perf_counter()

    for idx, start in enumerate(range(0, num_frames, chunk_size)):
        end = min(start + chunk_size, num_frames)
        chunk = latents_cpu[:, start:end].to(device)

        chunk_device = vae.decode_to_pixel_device(
            chunk, use_cache=True, chunk_idx=idx)

        if profile:
            torch.neuron.synchronize()
            vae_ms = (time.perf_counter() - t) * 1000
            logger.info("  chunk %2d: %.1fms", idx, vae_ms)

        chunk_video = vae.postprocess_pixels(chunk_device)
        video_chunks.append(chunk_video)

        if profile:
            t = time.perf_counter()

    vae.model.clear_cache()
    return torch.cat(video_chunks, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    dist.init_process_group(backend="neuron")
    init_vae_parallel_group()
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(0)

    latents = torch.load(args.input, map_location="cpu")
    logger.info("Loaded latents: %s, dtype=%s", latents.shape, latents.dtype)

    vae = build_vae(dtype=latents.dtype)
    latents_local_cpu = w_shard(latents, rank, world)
    device = torch.device("neuron")

    profile = os.environ.get("PROFILE_VAE", "0") == "1"
    if profile:
        torch.neuron.synchronize()
    t_start = time.perf_counter()
    video_local = decode_latents(
        vae, latents_local_cpu, device, args.stream, args.chunk_size,
        profile=profile)
    if profile:
        torch.neuron.synchronize()
    logger.info("Decode: %.1fms", (time.perf_counter() - t_start) * 1000)

    if os.environ.get("DUMP_VIDEO_TENSOR", "0") == "1":
        torch.save(video_local, f"video_local_rank{rank}.pt")

    gather_and_save(video_local, args.output, args.fps, rank, world)
    destroy_vae_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
