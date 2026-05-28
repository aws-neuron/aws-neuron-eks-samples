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

from models.t5 import (
    build_text_encoder,
    destroy_t5_parallel_group,
    encode_one_prompt,
    init_t5_parallel_group,
)
from utils.logging_utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output", type=str, default="prompt_embeds.pt")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    assert args.prompt or args.prompt_file, "Provide --prompt or --prompt_file"

    torch.set_grad_enabled(False)
    os.environ.setdefault("NEURON_FALLBACK_ENABLED", "0")
    dist.init_process_group(backend="neuron")
    init_t5_parallel_group()
    rank = dist.get_rank()

    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    logger.info("Loading UMT5-XXL encoder (TP-sharded)...")
    text_encoder = build_text_encoder(device="neuron")

    profile = os.environ.get("PROFILE_T5", "0") == "1"

    logger.info("Encoding %d prompt(s)...", len(prompts))
    results = []
    for i, prompt in enumerate(prompts):
        if profile:
            torch.neuron.synchronize()
            t = time.perf_counter()
        context = encode_one_prompt(text_encoder, prompt)
        if profile:
            torch.neuron.synchronize()
            logger.info("  [%d] %.1fms: %s...",
                        i, (time.perf_counter() - t) * 1000, prompt[:80])
        else:
            logger.info("  [%d] done: %s...", i, prompt[:80])
        if rank == 0:
            results.append(context.cpu())

    if rank == 0:
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            for i, emb in enumerate(results):
                torch.save(emb, os.path.join(args.output_dir, f"prompt_{i:03d}.pt"))
            logger.info("Saved %d files to %s/", len(results), args.output_dir)
        else:
            all_embeds = torch.cat(results, dim=0)
            torch.save(all_embeds, args.output)
            logger.info("Saved %s, shape=%s, dtype=%s",
                        args.output, all_embeds.shape, all_embeds.dtype)

    destroy_t5_parallel_group()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
