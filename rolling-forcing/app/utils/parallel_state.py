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

import torch.distributed as dist


_GROUPS = {}


def register_group(name, group):
    assert name not in _GROUPS, f"group {name!r} already registered"
    _GROUPS[name] = group


def destroy_group(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    del _GROUPS[name]


def is_registered(name):
    return name in _GROUPS


def _get(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    return _GROUPS[name]


def get_group(name):
    return _get(name)


def get_world_size(name):
    return dist.get_world_size(_get(name))


def get_rank(name):
    return dist.get_rank(_get(name))


def all_gather_into_tensor(output, input, group_name):
    dist.all_gather_into_tensor(output, input, group=_get(group_name))


def reduce_scatter_tensor(output, input, group_name):
    dist.reduce_scatter_tensor(output, input, group=_get(group_name))


def all_reduce(tensor, group_name, op=dist.ReduceOp.SUM):
    dist.all_reduce(tensor, op=op, group=_get(group_name))
