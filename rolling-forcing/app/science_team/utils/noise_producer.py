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

import queue
import threading
from concurrent.futures import Future

import torch


class NoiseProducer:

    _STOP = object()

    def __init__(self, dtype, max_inflight=1):
        self._dtype = dtype
        self._requests = queue.Queue(maxsize=max_inflight)

        self._gen = torch.Generator()
        self._gen.set_state(torch.random.get_rng_state())

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="noise-producer")
        self._thread.start()

    def request(self, plan):
        future = Future()
        self._requests.put((list(plan), future))
        return future

    def _run(self):
        while True:
            item = self._requests.get()
            if item is self._STOP:
                return
            plan, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                slices = []
                for full_shape, sl in plan:
                    draw = torch.randn(full_shape, dtype=self._dtype, generator=self._gen)
                    slices.append(draw[sl].clone())
                packed = torch.cat(slices, dim=0)
            except BaseException as e:
                future.set_exception(e)
            else:
                future.set_result(packed)

    def shutdown(self):
        self._requests.put(self._STOP)
        self._thread.join()
