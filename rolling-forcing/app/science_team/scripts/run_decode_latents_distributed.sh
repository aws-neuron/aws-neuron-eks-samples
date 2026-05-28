#!/bin/bash
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

set -e

WORLD_SIZE=8
INPUT_PATH="output_latent.pt"
OUTPUT_PATH="output.mp4"
CHUNK_SIZE=3
FPS=16

export NEURON_FALLBACK_ENABLED=0
export NEURON_LOGICAL_NC_CONFIG=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0

PROFILE_VAE="${PROFILE_VAE:-1}" \
    torchrun --nproc_per_node "$WORLD_SIZE" decode_latents.py \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --fps "$FPS" \
    --stream \
    --chunk-size "$CHUNK_SIZE"
