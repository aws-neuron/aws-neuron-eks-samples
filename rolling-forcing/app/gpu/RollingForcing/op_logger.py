"""
Device-agnostic dual-level op logger for comparing ops across backends.

Two log levels:
- torch level (TorchFunctionMode): High-level ops like F.cross_entropy, F.rms_norm
- aten level (TorchDispatchMode): Decomposed ATen ops like aten.mm.default

Output is deduplicated: identical op calls (same op, location, input/output
shapes/dtypes) are collapsed into a single entry with a ``count`` field.

Copied from NanoChat repo
"""

import sys
import json
from collections import OrderedDict

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

# Non-device ops to skip: Python property accesses, metadata queries, etc.
_SKIP_TORCH_OPS = {
    "getset_descriptor.__get__",
    "getset_descriptor.__set__",
    "TensorBase.size",
    "_set_grad_enabled",
    "_VariableFunctionsClass.is_complex",
    "Tensor.backward",
    "TensorBase.__bool__",
}

_SKIP_ATEN_OPS = {
    "prim.device.default",
    "aten.lift_fresh.default",
    "aten._local_scalar_dense.default",
    "aten._unsafe_view.default",
    "profiler._record_function_enter_new.default",
    "profiler._record_function_exit._RecordFunction",
}


def _get_user_caller():
    """Walk stack to find nearest frame outside torch/ and this module."""
    try:
        frame = sys._getframe(2)
    except ValueError:
        # Stack too shallow (e.g. C++ autograd engine callback)
        return "???", 0
    while frame:
        fname = frame.f_code.co_filename
        if "/torch/" not in fname and fname != __file__:
            # Return just the basename for readability
            basename = fname.rsplit("/", 1)[-1] if "/" in fname else fname
            return basename, frame.f_lineno
        frame = frame.f_back
    return "???", 0


def _extract_tensor_info(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}


def _extract_info(v):
    if isinstance(v, torch.Tensor):
        return _extract_tensor_info(v)
    elif isinstance(v, (list, tuple)):
        infos = [_extract_info(x) for x in v]
        return infos
    return repr(v)


def _extract_args(args):
    return [_extract_info(a) for a in args]


def _signature_of(v):
    """Extract a hashable (shape, dtype) signature, ignoring device."""
    if isinstance(v, dict) and "shape" in v and "dtype" in v:
        return (tuple(v["shape"]), v["dtype"])
    elif isinstance(v, list):
        return tuple(_signature_of(x) for x in v)
    return v


def _make_key(entry):
    """Build a hashable dedup key from an entry dict."""
    inputs_sig = tuple(_signature_of(x) for x in entry["inputs"])
    kwargs_sig = tuple(
        sorted((k, _signature_of(v)) for k, v in entry.get("input_kwargs", {}).items())
    )
    output_sig = _signature_of(entry["output"])
    return (
        entry["level"],
        entry["op"],
        entry["loc"],
        inputs_sig,
        kwargs_sig,
        output_sig,
    )


class _FuncLogger(TorchFunctionMode):
    def __init__(self, record_fn):
        super().__init__()
        self._record = record_fn

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__qualname__", getattr(func, "__name__", str(func)))
        if name in _SKIP_TORCH_OPS:
            return func(*args, **kwargs)
        fname, lineno = _get_user_caller()
        entry = {
            "level": "torch",
            "op": name,
            "loc": f"{fname}:{lineno}",
            "inputs": _extract_args(args),
            "input_kwargs": {k: _extract_info(v) for k, v in kwargs.items()},
        }
        result = func(*args, **kwargs)
        entry["output"] = _extract_info(result)
        self._record(entry)
        return result


class _DispatchLogger(TorchDispatchMode):
    def __init__(self, record_fn):
        super().__init__()
        self._record = record_fn

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        op_name = str(func.overloadpacket) + "." + func._overloadname
        if op_name in _SKIP_ATEN_OPS:
            return result
        fname, lineno = _get_user_caller()
        entry = {
            "level": "aten",
            "op": op_name,
            "loc": f"{fname}:{lineno}",
            "inputs": _extract_args(args),
            "input_kwargs": {k: _extract_info(v) for k, v in kwargs.items()},
            "output": _extract_info(result),
        }
        self._record(entry)
        return result


class OpLogger:
    """Dual-level op logger with deduplication. Use as context manager."""

    def __init__(self):
        # OrderedDict preserves insertion order; values are (entry, count)
        self._unique = OrderedDict()
        self._func_logger = _FuncLogger(self._record)
        self._dispatch_logger = _DispatchLogger(self._record)

    def _record(self, entry):
        key = _make_key(entry)
        if key in self._unique:
            _, count = self._unique[key]
            self._unique[key] = (self._unique[key][0], count + 1)
        else:
            self._unique[key] = (entry, 1)

    def __enter__(self):
        self._func_logger.__enter__()
        self._dispatch_logger.__enter__()
        return self

    def __exit__(self, *args):
        self._dispatch_logger.__exit__(*args)
        self._func_logger.__exit__(*args)

    def dump_log(self, filepath):
        with open(filepath, "w") as f:
            for entry, count in self._unique.values():
                out = {**entry, "count": count}
                f.write(json.dumps(out, default=str) + "\n")

    def print_summary(self):
        total_calls = sum(c for _, c in self._unique.values())
        num_unique = len(self._unique)
        print(
            f"\n=== Op Log Summary ({num_unique} unique signatures, {total_calls} total calls) ==="
        )
        for level in ("torch", "aten"):
            ops = {}
            for entry, count in self._unique.values():
                if entry["level"] == level:
                    name = entry["op"]
                    ops[name] = ops.get(name, 0) + count
            if ops:
                level_unique = sum(
                    1 for (e, _) in self._unique.values() if e["level"] == level
                )
                level_total = sum(ops.values())
                print(f"\n[{level}] ({level_unique} unique, {level_total} total calls)")
                for op, count in sorted(ops.items(), key=lambda x: -x[1]):
                    print(f"  {op}: {count}x")
