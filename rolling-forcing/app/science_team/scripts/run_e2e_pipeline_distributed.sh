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
TP_DEGREE=4
CHUNK_SIZE=3
NUM_OUTPUT_FRAMES=126
FPS=16

CONFIG_PATH="configs/rolling_forcing_dmd.yaml"
CHECKPOINT_PATH="checkpoints/rolling_forcing_dmd.pt"
PROMPT_FILE="prompts/example_prompts.txt"
OUTPUT_FOLDER="videos_pipeline"

export NEURON_FALLBACK_ENABLED=0
export NEURON_LOGICAL_NC_CONFIG=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0

PROFILE_E2E_PIPELINE="${PROFILE_E2E_PIPELINE:-1}" \
    torchrun --nproc_per_node "$WORLD_SIZE" e2e_pipeline.py \
    --prompt_file "$PROMPT_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --tp_degree "$TP_DEGREE" \
    --num_output_frames "$NUM_OUTPUT_FRAMES" \
    --chunk-size "$CHUNK_SIZE" \
    --fps "$FPS" \
    --use_ema \
    --rng_state_path cpu_rng_states
