"""Reusable neuron-profile wrapper for profiling NEFF kernels.

Usage:
    from tests.neuron_profiler import profile_kernel

    # After warmup call (so _neff_path is set):
    result = profile_kernel(kernel_obj)
    print(f"Execution time: {result['total_active_time_us']:.2f} us")

The kernel object is the DeviceKernel returned by @jit or jit().
"""

import json
import os
import subprocess
from pathlib import Path


def profile_neff(
    neff_path: str,
    num_exec: int = 5,
    profile_nth_exec: int = 5,
) -> dict:
    """Profile a NEFF file using neuron-profile CLI.

    Runs the NEFF with synchronous execution, captures a profile on the
    nth execution, then parses the summary JSON.

    Args:
        neff_path: Path to the compiled .neff file.
        num_exec: Total number of executions.
        profile_nth_exec: Which execution to profile.

    Returns:
        Parsed summary-json dict from neuron-profile view.
    """
    neff_path = Path(neff_path).resolve()
    if not neff_path.exists():
        raise FileNotFoundError(f"NEFF not found: {neff_path}")

    neff_dir = neff_path.parent
    ntff_base = "profile.ntff"
    ntff_result = neff_dir / f"profile_exec_{profile_nth_exec}.ntff"

    # Capture profile with synchronous execution
    env = os.environ.copy()
    env["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "0"

    capture_cmd = [
        "neuron-profile", "capture",
        "-n", str(neff_path),
        "-s", ntff_base,
        f"--num-exec={num_exec}",
        f"--profile-nth-exec={profile_nth_exec}",
    ]
    subprocess.run(capture_cmd, cwd=neff_dir, env=env, check=True, capture_output=True)

    if not ntff_result.exists():
        raise FileNotFoundError(
            f"Expected profile output not found: {ntff_result}"
        )

    # View summary
    view_cmd = [
        "neuron-profile", "view",
        "-n", str(neff_path),
        "-s", str(ntff_result),
        "--output-format", "summary-json",
    ]
    result = subprocess.run(
        view_cmd, cwd=neff_dir, check=True, capture_output=True, text=True,
    )

    summary = json.loads(result.stdout)

    # Clean up ntff files
    for f in neff_dir.glob("profile*.ntff"):
        f.unlink(missing_ok=True)

    # Extract per-core results; return the first core's summary
    # (single-core kernels have exactly one entry)
    cores = list(summary.values())
    if len(cores) == 1:
        return cores[0]
    return summary


def profile_kernel(kernel_obj, num_exec: int = 5, profile_nth_exec: int = 5) -> dict:
    """Profile a DeviceKernel by reading its _neff_path.

    Args:
        kernel_obj: A DeviceKernel instance (from @jit or jit()) that has
            already been called at least once (so _neff_path is set).
        num_exec: Total number of executions.
        profile_nth_exec: Which execution to profile.

    Returns:
        Parsed summary-json dict from neuron-profile view.
    """
    neff_path = getattr(kernel_obj, "_neff_path", None)
    if neff_path is None:
        raise RuntimeError(
            "Kernel has no _neff_path. Call the kernel at least once before profiling."
        )
    return profile_neff(neff_path, num_exec, profile_nth_exec)
