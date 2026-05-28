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

import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 128


def _compile(mod_or_fn):
    return torch.compile(mod_or_fn, backend="neuron", dynamic=False, fullgraph=True)


def w_shard(tensor, rank, world):
    W = tensor.shape[-1]
    assert W % world == 0, f"W={W} not divisible by world={world}"
    s = W // world
    return tensor[..., rank * s:(rank + 1) * s].contiguous()
