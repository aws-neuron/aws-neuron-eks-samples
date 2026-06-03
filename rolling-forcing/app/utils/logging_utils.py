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

import logging
import os


PROJECT_LOGGER = "video_streaming"

_CONFIGURED = False


class _RankDowngradeFilter(logging.Filter):

    def __init__(self, handler_level):
        super().__init__()
        self._handler_level = handler_level

    def filter(self, record):
        rank = os.environ.get("RANK", "0")
        if rank == "0":
            return True
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
        record.msg = f"[rank{rank}] {record.msg}"
        return record.levelno >= self._handler_level


def configure_logging(level=logging.INFO):
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(message)s",
        datefmt="%H:%M:%S",
    ))
    handler.setLevel(level)
    handler.addFilter(_RankDowngradeFilter(level))
    parent = logging.getLogger(PROJECT_LOGGER)
    parent.setLevel(logging.DEBUG)
    parent.addHandler(handler)
    parent.propagate = False
    _CONFIGURED = True


def get_logger(name):
    return logging.getLogger(f"{PROJECT_LOGGER}.{name}")
