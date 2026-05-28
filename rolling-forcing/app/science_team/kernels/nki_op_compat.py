"""Compatibility shim for alpha nki_op API.

nki_op supports mutating kernel arguments (mutates_args={"cache"}).
Our SDK only has wrap_nki which requires immutable inputs.
This stub uses wrap_nki and ignores mutation semantics — the caller
must handle mutation externally (e.g., tensor.copy_() after kernel call).
"""
from torch_neuronx.nki_hop import wrap_nki


def nki_op(name, mutates_args=None):
    """Decorator stub replacing alpha nki_op with wrap_nki."""
    def decorator(fn):
        return wrap_nki(fn)
    return decorator
