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
PROMPT_FILE="prompts/example_prompts.txt"
OUTPUT_DIR="text_embeds"

export NEURON_FALLBACK_ENABLED=0
export NEURON_LOGICAL_NC_CONFIG=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0

PROFILE_T5="${PROFILE_T5:-1}" \
    torchrun --nproc_per_node "$WORLD_SIZE" encode_prompt.py \
    --prompt_file "$PROMPT_FILE" \
    --output_dir "$OUTPUT_DIR"
