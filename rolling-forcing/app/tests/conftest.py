"""Pytest configuration for unit tests.

This configuration disables CPU fallback so tests fail when operations
cannot be executed on the Neuron device.
"""

import os
import pytest
import torch


@pytest.fixture(scope="function", autouse=True)
def init_per_function():
    torch.manual_seed(42)


@pytest.fixture(scope="session", autouse=True)
def init_per_session():
    # Disable CPU fallback entirely - both implemented and unimplemented ops
    # will error if they can't run on device
    os.environ["NEURON_FALLBACK_ENABLED"] = "0"
    # FIXME: the rope test case failed when enabling NEURON_RT_ENABLE_DGE_NOTIFICATIONS
    # os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
