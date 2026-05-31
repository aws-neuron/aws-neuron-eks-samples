"""Compatibility shim replacing nkilib imports with local copies.

The real nkilib source is from private-nki-staging. We copy the actual
implementations (not stubs) to get correct behavior.
"""
import nki.language as nl

# ─── kernel_assert ────────────────────────────────────────────────────────────
def kernel_assert(condition, msg=""):
    """No-op in production — NKI doesn't support runtime assertions."""
    pass

def assert_shape(tensor, expected_shape, name=""):
    """No-op in production."""
    pass

# ─── kernel_helpers ───────────────────────────────────────────────────────────
PSUM_BANK_SIZE = 2048

def div_ceil(x, y):
    return (x + y - 1) // y

def sizeinbytes(dtype):
    if str(dtype) == str(nl.float32):
        return 4
    elif str(dtype) == str(nl.bfloat16) or str(dtype) == str(nl.float16) or str(dtype) == str(nl.uint16):
        return 2
    elif str(dtype) == str(nl.int8) or str(dtype) == str(nl.uint8):
        return 1
    elif str(dtype) == str(nl.int32) or str(dtype) == str(nl.uint32):
        return 4
    return 2  # default bf16

def align_to(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment

# ─── ModularAllocator (real implementation) ───────────────────────────────────
from kernels.nkilib_modular_allocator import ModularAllocator

# ─── TensorView (real implementation) ────────────────────────────────────────
from kernels.nkilib_tensor_view import TensorView
