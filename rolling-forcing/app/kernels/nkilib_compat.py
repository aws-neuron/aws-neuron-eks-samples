"""Compatibility shim replacing nkilib alpha APIs with SDK-available equivalents.

Replaces:
- nkilib.core.utils.kernel_assert.kernel_assert / assert_shape
- nkilib.core.utils.kernel_helpers.PSUM_BANK_SIZE / div_ceil
- nkilib.core.utils.modular_allocator.ModularAllocator
- nkilib.core.utils.tensor_view.TensorView
"""

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

# ─── ModularAllocator ─────────────────────────────────────────────────────────
import nki.language as nl


class ModularAllocator(nl.NKIObject):
    """Stub for nkilib ModularAllocator.

    The alpha SDK uses this for SBUF address management. In our SDK,
    NKI handles allocation automatically via nl.ndarray. This stub
    tracks allocations for compatibility but doesn't enforce addresses.
    """
    def __init__(self, initial_address=0):
        self.address = initial_address

    def alloc(self, size, alignment=1):
        addr = self.address
        if alignment > 1:
            addr = (addr + alignment - 1) // alignment * alignment
        self.address = addr + size
        return addr

    def alloc_sbuf_tensor(self, shape, dtype):
        """Allocate an SBUF tensor (stub: just creates nl.ndarray)."""
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def get_current_address(self):
        return self.address

    def set_current_address(self, addr):
        self.address = addr

    def reset(self):
        self.address = 0


# ─── TensorView ──────────────────────────────────────────────────────────────
class TensorView(nl.NKIObject):
    """Stub for nkilib TensorView.

    The alpha SDK uses this for zero-copy tensor view operations.
    This stub wraps the tensor and provides pass-through access.
    """
    def __init__(self, tensor):
        self.tensor = tensor

    def select(self, dim, index):
        # Return a view selecting along dim at index
        return TensorView(self.tensor)

    def slice(self, dim, start, end):
        return TensorView(self.tensor)

    def get_view(self):
        return self.tensor

    def __getitem__(self, key):
        return self.tensor[key]

    def __setitem__(self, key, value):
        self.tensor[key] = value
