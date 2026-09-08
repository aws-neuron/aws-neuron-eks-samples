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

import torch


def restore_cpu_rng(rng_state_path, sample_name, verbose=False):
    if not rng_state_path:
        return
    path = rng_state_path
    if os.path.isdir(path):
        path = os.path.join(path, sample_name)
    rng_state = torch.load(path, map_location="cpu")
    torch.random.set_rng_state(rng_state)
    if verbose:
        print(f"Restored CPU RNG state from {path}", flush=True)
