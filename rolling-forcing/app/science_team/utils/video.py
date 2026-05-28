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
import tempfile

import av
import torch
import torch.distributed as dist
from einops import rearrange


def video_tensor_to_uint8(video: torch.Tensor) -> torch.Tensor:
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = rearrange(video, "b t c h w -> b t h w c")
    return (255.0 * video).to(torch.uint8)


def _write_mp4(frames_uint8: torch.Tensor, path: str, fps: int) -> None:
    container = av.open(path, mode="w")
    try:
        _, H, W, _ = frames_uint8.shape
        stream = container.add_stream("h264", rate=fps)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        for frame in frames_uint8.numpy():
            av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def save_video(video: torch.Tensor, output_path: str, fps: int) -> None:
    video_uint8 = video_tensor_to_uint8(video)
    B = video_uint8.shape[0]
    for i in range(B):
        path = output_path if B == 1 \
               else output_path.replace(".mp4", f"_{i}.mp4")
        _write_mp4(video_uint8[i], path, fps)
        print(f"Saved {path}")


def gather_and_save(video_local: torch.Tensor, output_path: str, fps: int,
                    rank: int, world: int) -> None:
    scratch_dir = os.path.join(tempfile.gettempdir(), "vae_shards")
    if rank == 0:
        os.makedirs(scratch_dir, exist_ok=True)
    dist.barrier()

    torch.save(video_local, os.path.join(scratch_dir, f"shard_rank{rank}.pt"))
    dist.barrier()

    if rank == 0:
        shards = [
            torch.load(os.path.join(scratch_dir, f"shard_rank{r}.pt"),
                       map_location="cpu")
            for r in range(world)
        ]
        save_video(torch.cat(shards, dim=-1), output_path, fps)

    dist.barrier()
    if rank == 0:
        for r in range(world):
            os.remove(os.path.join(scratch_dir, f"shard_rank{r}.pt"))
        os.rmdir(scratch_dir)
