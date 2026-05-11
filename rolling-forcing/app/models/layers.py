import math
import os
import time

import torch
import torch.nn as nn

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn=None, **kwargs):
    """No-op wrapper since torch_neuronx.jit is not available."""
    if fn is None:
        # Called with arguments like @jit(is_nki_kb=True)
        return lambda f: f
    return fn

# NKI kernel loading
# Controlled by USE_NKI_KERNELS env var (default: true)
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "true").lower() == "true"

print("=" * 60)
print("[layers.py] NKI Kernel Loading (USE_NKI_KERNELS=%s)" % USE_NKI_KERNELS)
print("=" * 60)

# --- Kernel 1/4: cross_attention (kernels/cross_attention.py) ---
NKI_AVAILABLE = False
wan_cross_attn = None
if USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.cross_attention import wan_cross_attn
        wan_cross_attn = wrap_nki(wan_cross_attn)
        NKI_AVAILABLE = True
        print("  [1/4] kernels/cross_attention.py  ✓ LOADED")
    except Exception as e:
        print(f"  [1/4] kernels/cross_attention.py  ✗ FAILED: {e}")
else:
    print("  [1/4] kernels/cross_attention.py  — SKIPPED (disabled)")

# --- Kernel 2/4: rope (kernels/rope.py) ---
ROPE_NKI_AVAILABLE = False
causal_rope_rotation_nki = None
if USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki as _wrap_nki_rope
        from kernels.rope import causal_rope_rotation as _causal_rope_rotation
        causal_rope_rotation_nki = _wrap_nki_rope(_causal_rope_rotation)
        ROPE_NKI_AVAILABLE = True
        print("  [2/4] kernels/rope.py              ✓ LOADED")
    except Exception as e:
        print(f"  [2/4] kernels/rope.py              ✗ FAILED: {e}")
else:
    print("  [2/4] kernels/rope.py              — SKIPPED (disabled)")

# --- Kernel 3/4: self_attention (kernels/self_attention.py) ---
SELF_ATTN_NKI_AVAILABLE = False
wan_flash_self_attn_nki = None
if USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki as _wrap_nki_self_attn
        from kernels.self_attention import wan_flash_self_attn as _wan_flash_self_attn
        wan_flash_self_attn_nki = _wrap_nki_self_attn(_wan_flash_self_attn)
        SELF_ATTN_NKI_AVAILABLE = True
        print("  [3/4] kernels/self_attention.py     ✓ LOADED")
    except Exception as e:
        print(f"  [3/4] kernels/self_attention.py     ✗ FAILED: {e}")
else:
    print("  [3/4] kernels/self_attention.py     — SKIPPED (disabled)")

# --- Kernel 4/4: kv_cache_copy (kernels/kv_cache_copy.py) ---
# NOT loaded as NKI — uses tensor.copy_() (optimal DMA on Neuron).
# NKI kv_cache_copy cannot work because input parameters are immutable.
build_rope_grids = None
print("  [4/4] kernels/kv_cache_copy.py     — NOT NKI (uses tensor.copy_() DMA)")

print("=" * 60)
print("[layers.py] Summary: cross_attn=%s  rope=%s  self_attn=%s  kv_cache=tensor.copy_()" % (
    "✓" if NKI_AVAILABLE else "✗",
    "✓" if ROPE_NKI_AVAILABLE else "✗",
    "✓" if SELF_ATTN_NKI_AVAILABLE else "✗"))
print("=" * 60)

# NKI self-attention kernel requires seqlen_k to be a multiple of this value
ATTN_SEQLEN_MULTIPLE = 8192


@jit
class GELU(nn.Module):

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        result = 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return result.to(dtype)


@jit
class SiLU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


@jit
class WanFFN(nn.Module):

    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.gelu = GELU()
        self.fc2 = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


@jit
class WanLayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))

    def norm_(self, x):
        x = x.float()
        mean = torch.sum(x, dim=-1, keepdim=True) / self.dim
        diff = x - mean
        variance = torch.sum(diff * diff, dim=-1, keepdim=True) / self.dim
        return (diff * torch.rsqrt(variance + self.eps)).to(x.dtype)

    def forward(self, x):
        output = self.norm_(x)
        if hasattr(self, 'weight'):
            output = output * self.weight + self.bias
        return output.type_as(x)


@jit
class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


@jit
class WanPatchEmbed(nn.Module):
    """Patch embedding via matmul (equivalent to nn.Conv3d with kernel_size=stride).

    Neuron device does not support Conv3d; this module implements the same
    operation by reshaping input into non-overlapping patch vectors and
    multiplying with the flattened Conv3d weight.

    Weight/bias have the same shape and names as nn.Conv3d, so existing
    checkpoints load directly.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.patch_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        B, C, F, H, W = x.shape
        pT, pH, pW = self.patch_size

        x = x.reshape(B, C, F // pT, pT, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.reshape(B, (F // pT) * (H // pH) * (W // pW), C * pT * pH * pW)

        out = torch.matmul(x, self.weight.flatten(1).t()) + self.bias

        out = out.transpose(1, 2).reshape(
            B, self.out_channels, F // pT, H // pH, W // pW)
        return out


def causal_head_modulate(x, e, modulation):
    """Adaptive modulation: add bias, split into shift/scale, apply.

    Corresponds to CausalHead.forward():
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = ... * (1 + e[1]) + e[0]

    Uses slicing instead of .chunk() (tuple return not traceable).

    Args:
        x: [B, F, S, C] normalized and unflattened input
        e: [B, F, 1, C] time embedding
        modulation: [1, 2, C] learned bias parameter

    Returns: [B, F, S, C] modulated output
    """
    e = modulation.unsqueeze(1) + e # [B, F, 2, C]
    e_shift = e[:, :, 0:1]          # [B, F, 1, C]
    e_scale = e[:, :, 1:2]          # [B, F, 1, C]
    return x * (1 + e_scale) + e_shift


class CausalHead(nn.Module):
    """Final head: LayerNorm + adaptive modulation + linear projection.

    Corresponds to CausalHead in causal_model_opt.py.
    """

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = jit(nn.Linear(dim, out_channels))

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

        self._modulate = jit(causal_head_modulate)

    def forward(self, x, e):
        """
        Args:
            x: [B, L, C] where L = F * frame_seqlen
            e: [B, F, 1, C] time embedding

        Returns: [B, F, frame_seqlen, out_C]
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        x = self.norm(x).unflatten(1, (num_frames, frame_seqlen))
        x = self._modulate(x, e, self.modulation)
        return self.head(x)


def unpatchify(x, out_dim, patch_size, grid_sizes):
    """Reconstruct video tensor from patch embeddings.

    Corresponds to CausalWanModel.unpatchify():
        u = x.squeeze(0).view(f, h, w, pT, pH, pW, c)
        u = u.permute(6, 0, 3, 1, 4, 2, 5)  # [c, f, pT, h, pH, w, pW]
        u = u.reshape(c, f*pT, h*pH, w*pW)

    Args:
        x: [B, F*H*W, out_C] where out_C = prod(patch_size) * out_dim
        out_dim: output channels (16)
        patch_size: (pT, pH, pW) tuple
        grid_sizes: (f, h, w) tuple

    Returns: [c, f*pT, h*pH, w*pW] (assumes B=1, returns squeezed)
    """
    f, h, w = grid_sizes
    pT, pH, pW = patch_size
    u = x.squeeze(0).view(f, h, w, pT, pH, pW, out_dim)
    u = u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
    u = u.reshape(out_dim, f * pT, h * pH, w * pW)
    return u


def convert_flow_pred_to_x0(flow_pred, xt, sigma_t):
    """Convert flow prediction to x0: x0 = xt - sigma_t * flow_pred.

    Uses fp32 (Neuron does not support fp64). GPU uses fp64 but fp32 is
    sufficient for this linear operation.

    Args:
        flow_pred: [B*F, C, H, W] model velocity prediction (bf16)
        xt:        [B*F, C, H, W] noisy input (bf16)
        sigma_t:   [B*F] precomputed sigma for each frame (fp32)

    Returns: [B*F, C, H, W] predicted x0 (bf16)
    """
    dtype = flow_pred.dtype
    flow_pred = flow_pred.float()
    xt = xt.float()
    sigma_t = sigma_t.float().reshape(-1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(dtype)


def modulated_norm_scale(norm_x, scale, ones, num_frames, frame_seqlen):
    """Scale part of DiT-style adaptive modulation: norm_x * (1 + scale).

    Args:
        norm_x: [B, L, C] where L = num_frames * frame_seqlen
        scale:  [B, F, 1, C] (e[1] after modulation chunk)
        num_frames: number of frames F (int, baked in at trace time)
        frame_seqlen: tokens per frame S (int, baked in at trace time)

    Returns: [B, F, S, C]
    """
    y = norm_x.unflatten(1, (num_frames, frame_seqlen))
    return y * (ones + scale)


def modulated_norm_shift(y, shift):
    """Shift part of DiT-style adaptive modulation: y + shift, then flatten.

    Args:
        y:     [B, F, S, C] scaled norm output
        shift: [B, F, 1, C] (e[0] after modulation chunk)

    Returns: [B, L, C]
    """
    return (y + shift).flatten(1, 2)


def modulated_residual(x, y, scale, num_frames, frame_seqlen):
    """Scaled residual: x + unflatten(y) * scale.

    Corresponds to CausalWanAttentionBlock.forward():
        x + (y.unflatten(1, (F, S)) * e[2]).flatten(1, 2)

    Args:
        x:     [B, L, C] residual
        y:     [B, L, C] branch output (self-attn or FFN)
        scale: [B, F, 1, C] (e[2] or e[5] after modulation chunk)
        num_frames: number of frames F (int, baked in at trace time)
        frame_seqlen: tokens per frame S (int, baked in at trace time)

    Returns: [B, L, C]
    """
    return x + (y.unflatten(1, (num_frames, frame_seqlen)) * scale).flatten(1, 2)


def modulation_chunk(modulation, e):
    """Add learned modulation bias and split into 6 per-frame vectors.

    Corresponds to CausalWanAttentionBlock.forward():
        (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

    Args:
        modulation: [1, 6, C] learned bias (self.modulation)
        e:          [B, F, 6, C] time embeddings (e0 from _forward_inference)

    Returns: 6 tensors each [B, F, 1, C]
    """
    e = modulation.unsqueeze(1) + e
    return e[:, :, 0:1], e[:, :, 1:2], e[:, :, 2:3], e[:, :, 3:4], e[:, :, 4:5], e[:, :, 5:6]


def rope_params(max_seq_len, dim, theta=10000):
    """Precompute rotary position embedding frequencies.

    Corresponds to CausalWanModel.__init__():
        self.freqs_cos/sin = torch.cat([rope_params(...), ...], dim=1)

    Args:
        max_seq_len: maximum sequence length (1024)
        dim: frequency dimension (split across frame/height/width)
        theta: base frequency (10000)

    Returns: (cos, sin) each [max_seq_len, dim // 2] float32
    """
    assert dim % 2 == 0
    # Computed on CPU at init time, so float64 is fine (matches GPU precision)
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.cos(freqs).float(), torch.sin(freqs).float()


def sinusoidal_embedding_1d(dim, position):
    """Compute 1-D sinusoidal positional embeddings for timesteps.

    Corresponds to model.sinusoidal_embedding_1d() on GPU.
    Uses float32 instead of float64 (Neuron does not support float64).

    Args:
        dim: embedding dimension (must be even), e.g. freq_dim=256
        position: [N] timestep values

    Returns: [N, dim] sinusoidal embeddings (float32)
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.float()

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def causal_rope_apply(x, grid_sizes, freqs_cos, freqs_sin, start_frame=torch.tensor(0)):
    """Apply 3D rotary position embeddings (frame, height, width).

    Corresponds to CausalWanSelfAttention.forward():
        roped_query = causal_rope_apply(q, grid_sizes, freqs_cos, freqs_sin, ...).type_as(v)

    NOTE: start_frame is a scalar tensor (not a Python int) so that its value
    stays out of the compiled IR.  If it were an int, different start_frame
    values would produce different IRs and thus separate NEFFs.  Using a
    tensor + torch.arange + torch.index_select keeps the IR identical across
    start_frame values, reducing the number of NEFFs (one per unique
    grid_sizes instead of one per unique (grid_sizes, start_frame) pair).
    The trade-off is longer compilation time and larger NEFF size due to the
    extra index_select indirection; we plan to address this in a future
    optimisation pass.

    Args:
        x:         [B, L, N, D] query or key tensor (B=1)
        grid_sizes: (F, H, W) tuple baked in at trace time
        freqs_cos: [max_seq_len, D//2] precomputed cos frequencies
        freqs_sin: [max_seq_len, D//2] precomputed sin frequencies
        start_frame: scalar tensor (shape []), dynamic — not baked into IR

    Returns: [B, L, N, D]
    """
    n, c = x.size(2), x.size(3) // 2
    s0 = c - 2 * (c // 3)
    s1 = c // 3

    f, h, w = grid_sizes
    seq_len = f * h * w
    frame_idx = start_frame + torch.arange(f, device=start_frame.device)

    # build position grids [seq_len, 1, c]
    # use index_select for frame dim so start_frame (tensor) stays out of IR
    cos = torch.cat([
        torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)

    sin = torch.cat([
        torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)

    # rotary embedding on interleaved pairs, in float32 (Neuron does not support float64)
    # use x[:, :seq_len] (slice) instead of x[0, :seq_len] (select — unsupported)
    x_0 = x[:, :seq_len].to(torch.float32)                # [1, seq_len, n, D]
    x_pairs = x_0.reshape(1, seq_len, n, c, 2)            # [1, seq_len, n, c, 2]
    # use 0:1/1:2 slice + reshape instead of [..., 0] select
    x_re = x_pairs[:, :, :, :, 0:1].reshape(1, seq_len, n, c)
    x_im = x_pairs[:, :, :, :, 1:2].reshape(1, seq_len, n, c)

    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos

    # interleave with unsqueeze+cat instead of stack (unsupported)
    x_0 = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], dim=-1)
    x_0 = x_0.reshape(1, seq_len, n, c * 2)               # [1, seq_len, n, D]

    return x_0.type_as(x)


class WanT2VCrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.layer_idx = layer_idx

        # layers
        self.q = jit(nn.Linear(dim, dim))
        self.k = jit(nn.Linear(dim, dim))
        self.v = jit(nn.Linear(dim, dim))
        self.o = jit(nn.Linear(dim, dim))

        assert qk_norm is True
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        # Identity matrix for transpose ops inside wan_cross_attn kernel
        self.register_buffer('identity', torch.eye(self.head_dim), persistent=False)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        _profiling = os.environ.get("PROFILE_PIPELINE", "0") == "1" and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) == 0 and self.layer_idx == 0
        input_dtype = x.dtype
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        if _profiling:
            _tc1 = time.perf_counter()
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if _profiling:
            _t_q = (time.perf_counter() - _tc1) * 1000

        assert crossattn_cache is not None

        if _profiling:
            _tc2 = time.perf_counter()
        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]
        if _profiling:
            _t_kv = (time.perf_counter() - _tc2) * 1000

        # context_lens is unused — all K tokens are valid (same as GPU baseline)
        # Reshape for wan_cross_attn kernel:
        # kernel expects q (bs, d, seq_q), k (bs, d, seq_k), v (bs, seq_k, d)
        # where bs = num_heads (B=1, heads become kernel batch dimension)
        if q.device.type == "neuron" and NKI_AVAILABLE:
            if _profiling:
                _tc3 = time.perf_counter()
            q_nki = q[0].permute(1, 2, 0).contiguous()   # [num_heads, head_dim, L1]
            k_nki = k[0].permute(1, 2, 0).contiguous()   # [num_heads, head_dim, L2]
            v_nki = v[0].permute(1, 0, 2).contiguous()   # [num_heads, L2, head_dim]
            # Pad seqlen_q to multiple of 128 (NKI tile size)
            seqlen_q = q_nki.shape[2]
            P = 128
            pad = (P - seqlen_q % P) % P
            if pad > 0:
                q_nki = torch.nn.functional.pad(q_nki, (0, pad))
            if _profiling:
                _t_reshape = (time.perf_counter() - _tc3) * 1000
                _tc4 = time.perf_counter()
            x_nki = wan_cross_attn(q_nki, k_nki, v_nki, self.identity, softmax_scale=self.softmax_scale)
            if _profiling:
                _t_kernel = (time.perf_counter() - _tc4) * 1000
            # kernel output: [seqlen_q_padded, num_heads, head_dim] → slice to [L1, num_heads, head_dim]
            x = x_nki[:seqlen_q].unsqueeze(0).flatten(2)
        else:
            if _profiling:
                _tc3 = time.perf_counter()
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            import torch.nn.functional as F
            attn_out = F.scaled_dot_product_attention(q, k, v)
            if _profiling:
                _t_reshape = (time.perf_counter() - _tc3) * 1000
                _tc4 = time.perf_counter()
            # attn_out: [B, num_heads, L1, head_dim] → [B, L1, C]
            x = attn_out.permute(0, 2, 1, 3).flatten(2)
            if _profiling:
                _t_kernel = (time.perf_counter() - _tc4) * 1000
        if _profiling:
            _tc6 = time.perf_counter()
        x = self.o(x)
        if _profiling:
            _t_output = (time.perf_counter() - _tc6) * 1000
            print(f"          [cross_attn] q={_t_q:.1f}ms  kv={_t_kv:.1f}ms  reshape={_t_reshape:.1f}ms  kernel={_t_kernel:.1f}ms  output={_t_output:.1f}ms")
        return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=0,
                 frame_length=1560):
        assert dim % num_heads == 0
        assert qk_norm, "qk_norm must be True"
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.eps = eps
        self.frame_length = frame_length
        self.max_attention_size = 21 * self.frame_length       # 32760
        self.block_length = 3 * self.frame_length               # 4680
        self.kv_cache_logical_size = 24 * self.frame_length     # 37440
        self.layer_idx = layer_idx

        # layers — jit() wraps nn.Linear for Neuron tracing
        self.q = jit(nn.Linear(dim, dim))
        self.k = jit(nn.Linear(dim, dim))
        self.v = jit(nn.Linear(dim, dim))
        self.o = jit(nn.Linear(dim, dim))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        # NKI RoPE rotation kernel — validated: kernel + PyTorch cos_sin construction
        # match causal_rope_apply exactly (max abs diff = 0.000000).
        self._rope_kernel = causal_rope_rotation_nki
        self._rope_nki_available = ROPE_NKI_AVAILABLE
        # build_rope_grids not yet ported — grid-building done in PyTorch
        self._build_rope_grids_kernel = build_rope_grids
        # Self-attn NKI kernel
        self._nki_available = SELF_ATTN_NKI_AVAILABLE
        
        # Helper buffers for build_rope_grids NKI kernel
        sign_pattern = torch.ones(self.head_dim, dtype=torch.float32)
        sign_pattern[0::2] = -1.0
        self.register_buffer(
            'sign_pattern',
            sign_pattern.unsqueeze(0).expand(128, -1).contiguous(),
            persistent=False)

        # Buffers for NKI kernel
        self.register_buffer('identity', torch.eye(self.head_dim), persistent=False)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        # Self-attention NKI kernel
        self._self_attn_kernel = wan_flash_self_attn_nki

    def cache_copy_inplace(self, k_dst, k_src, v_dst=None, v_src=None):
        """Cache copy via tensor.copy_() — already uses optimal DMA on Neuron.
        NKI kv_cache_copy cannot work in standard neuronxcc.nki (immutable params)."""
        if k_src.shape != k_dst.shape or k_src.numel() == 0:
            raise AssertionError(
                f"[cache_copy_inplace] shape mismatch: k_dst={k_dst.shape}, k_src={k_src.shape}, "
                f"k_dst.numel()={k_dst.numel()}, k_src.numel()={k_src.numel()}, layer_idx={self.layer_idx}"
            )
        if v_dst is not None and (v_src.shape != v_dst.shape or v_src.numel() == 0):
            raise AssertionError(
                f"[cache_copy_inplace] shape mismatch: v_dst={v_dst.shape}, v_src={v_src.shape}, "
                f"v_dst.numel()={v_dst.numel()}, v_src.numel()={v_src.numel()}, layer_idx={self.layer_idx}"
            )
        k_dst.copy_(k_src)
        if v_dst is not None:
            v_dst.copy_(v_src)

    def _nki_rope_apply(self, x, grid_sizes, freqs_cos, freqs_sin, start_frame):
        """Apply RoPE using the NKI kernel (Neuron) or fall back to traced PyTorch.

        Strategy:
        - Grid-building (cos/sin expansion from freqs tables) done in PyTorch
        - Rotation (x*cos + swap(x)*sin) done in NKI kernel when available
        - Falls back to full PyTorch implementation otherwise

        Args:
            x: [1, seq_len, N, D] query or key tensor
            grid_sizes: (F, H, W) tuple
            freqs_cos: [max_seq_len, D//2]
            freqs_sin: [max_seq_len, D//2]
            start_frame: scalar tensor

        Returns: [1, seq_len, N, D] bfloat16 (same dtype as input)
        """
        # Use PyTorch fallback if not on Neuron OR if NKI rope is not available
        if x.device.type != "neuron" or not self._rope_nki_available:
            return causal_rope_apply(
                x, grid_sizes, freqs_cos, freqs_sin, start_frame=start_frame
            ).type_as(x)

        b, s, n, d = x.shape
        f, h, w = grid_sizes
        seq_len = f * h * w
        c = d // 2  # half of head_dim
        s0 = c - 2 * (c // 3)
        s1 = c // 3

        # ── Build cos/sin grids in PyTorch (same as causal_rope_apply) ──
        frame_idx = start_frame + torch.arange(f, device=x.device)

        cos_half = torch.cat([
            torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)  # [seq_len, D//2]

        sin_half = torch.cat([
            torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)  # [seq_len, D//2]

        # ── Expand to interleaved full-D and apply sign pattern ──────────
        # cos_expanded: [seq_len, D] where cos[..., 2j] = cos[..., 2j+1] = cos_half[..., j]
        cos_expanded = cos_half.repeat_interleave(2, dim=-1)  # [seq_len, D]

        # sin_signed: [seq_len, D] with sign pattern [-sin, +sin, -sin, +sin, ...]
        # sin[2j] = -sin_half[j], sin[2j+1] = +sin_half[j]
        sin_expanded = sin_half.repeat_interleave(2, dim=-1)  # [seq_len, D]
        sign = torch.ones(d, device=x.device, dtype=sin_expanded.dtype)
        sign[0::2] = -1.0
        sin_signed = sin_expanded * sign.unsqueeze(0)

        # ── Pack into [seq_len, 2*D] float32 for NKI kernel ─────────────
        cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()  # [seq_len, 2D]

        # ── Pad seq_len to multiple of 128 (NKI tile size) ──────────────
        P = 128
        pad = (P - seq_len % P) % P
        if pad > 0:
            cos_sin = torch.nn.functional.pad(cos_sin, (0, 0, 0, pad))
            x_nki = torch.nn.functional.pad(x[0, :seq_len], (0, 0, 0, 0, 0, pad))
        else:
            x_nki = x[0, :seq_len]

        # ── Call NKI rotation kernel ─────────────────────────────────────
        out = self._rope_kernel(x_nki, cos_sin, num_heads=n, head_dim=d)

        # Slice back to original seq_len, reshape to [1, seq_len, N, D]
        return out[:seq_len].unsqueeze(0).type_as(x)

    def forward(
        self,
        x,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        updating_cache=False,
        num_valid_frames=None,
        shared_buffers=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(tuple): Python tuple (F, H, W)
            freqs_cos(Tensor): Rope cos, shape [1024, C / num_heads / 2]
            freqs_sin(Tensor): Rope sin, shape [1024, C / num_heads / 2]
            kv_cache(dict): {"k", "v", "global_end_index", "local_end_index"}
            current_start(int): absolute start position in global sequence
            cache_start(int, optional): defaults to current_start
            updating_cache(bool): whether this is a cache update call
            num_valid_frames(int, optional): number of non-padding frames
            shared_buffers(tuple): (buffer_k, buffer_v) for scratch space
        """
        assert kv_cache is not None
        _profiling = os.environ.get("PROFILE_PIPELINE", "0") == "1" and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) == 0 and self.layer_idx == 0
        input_dtype = x.dtype
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, f"Batch size must be 1, got {b}"
        if cache_start is None:
            cache_start = current_start

        # ── Phase 1: QKV projection + RoPE ──────────────────────────────
        if _profiling:
            _t1 = time.perf_counter()
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen
        current_start_frame_t = torch.tensor(current_start_frame, device=x.device)
        roped_query = self._nki_rope_apply(
            q, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)
        roped_key = self._nki_rope_apply(
            k, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)
        if _profiling:
            _t_qkv_rope = (time.perf_counter() - _t1) * 1000

        num_frames_per_block = self.block_length // self.frame_length
        grid_sizes_one_block = (num_frames_per_block, h, w)

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        # ── Phase 2: Cache management (write + eviction) ────────────────
        if _profiling:
            _t2 = time.perf_counter()

        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        kv_cache_size = self.kv_cache_logical_size  # 37440 (logical, not tensor alloc)
        sink_tokens = self.block_length  # keep the first block (anchor) in cache

        # buffer_k/buffer_v: shared buffers [B, max_buffer_size, N, D]
        # Used as scratch during eviction, then as assembled KV for attention.
        buffer_k, buffer_v = shared_buffers

        # Eviction: left-shift old entries when cache overflows
        num_evicted = 0
        if (num_new_tokens > 0) and (
                num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens  # static: 28080
            src_start = sink_tokens + num_evicted
            self.cache_copy_inplace(
                buffer_k[0, :evict_rolled], kv_cache["k"][0, src_start:src_start + evict_rolled],
                buffer_v[0, :evict_rolled], kv_cache["v"][0, src_start:src_start + evict_rolled])
            self.cache_copy_inplace(
                kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled], buffer_k[0, :evict_rolled],
                kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled], buffer_v[0, :evict_rolled])

        # Unified index computation
        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        # Write new block to cache
        if local_start_index == 0:
            # anchor: store un-roped K (RoPE applied later at read time)
            self.cache_copy_inplace(
                kv_cache["k"][0, :self.block_length], k[0, :self.block_length],
                kv_cache["v"][0, :self.block_length], v[0, :self.block_length])
        else:
            self.cache_copy_inplace(
                kv_cache["k"][0, local_start_index:local_end_index], roped_key[0, :self.block_length],
                kv_cache["v"][0, local_start_index:local_end_index], v[0, :self.block_length])

        if num_new_tokens > 0:  # don't update indices when re-caching clean frame
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        if _profiling:
            _t_cache = (time.perf_counter() - _t2) * 1000
        # ── Phase 3: Assemble KV into buffers ────────────────────────────
        if _profiling:
            _t3 = time.perf_counter()
        if updating_cache:
            # Cache-update call: attend over full cache
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)

            # Always copy max_attention_size tokens (static length)
            self.cache_copy_inplace(
                buffer_k[0, :cache_len],
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len],
                buffer_v[0, :cache_len],
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            # Overwrite anchor with RoPEd version if anchor is visible
            if cache_start_pos == 0:
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin, start_frame=torch.tensor(0, device=v.device))
                self.cache_copy_inplace(
                    buffer_k[0, :self.block_length], anchor_roped[0])

            k_len_int = cache_len

        else:
            # Normal denoising (or first block): anchor + working cache + current
            offset = 0
            if local_start_index > 0:
                # Anchor block (roped to virtual past position)
                wc_max = self.max_attention_size - valid_tokens - self.block_length
                wc_end = local_start_index
                wc_start = max(self.block_length, wc_end - wc_max)
                wc_len = wc_end - wc_start

                wc_frame_length = wc_len // self.frame_length
                rope_start_frame = current_start_frame - wc_frame_length - num_frames_per_block
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin, start_frame=torch.tensor(rope_start_frame, device=v.device))
                self.cache_copy_inplace(
                    buffer_k[0, :self.block_length], anchor_roped[0],
                    buffer_v[0, :self.block_length], kv_cache["v"][0, :self.block_length])
                offset = self.block_length

                # Working cache
                if wc_len > 0:
                    self.cache_copy_inplace(
                        buffer_k[0, offset:offset + wc_len], kv_cache["k"][0, wc_start:wc_start + wc_len],
                        buffer_v[0, offset:offset + wc_len], kv_cache["v"][0, wc_start:wc_start + wc_len])
                offset += wc_len  # advance by actual wc_len (dynamic offset)

            # Current tokens
            self.cache_copy_inplace(
                buffer_k[0, offset:offset + valid_tokens], roped_key[0, :valid_tokens],
                buffer_v[0, offset:offset + valid_tokens], v[0, :valid_tokens])
            k_len_int = offset + valid_tokens

        if _profiling:
            _t_assemble = (time.perf_counter() - _t3) * 1000
        # ── Phase 4: Single attention call ──────────────────────────────
        if _profiling:
            _t4 = time.perf_counter()
        # Reshape to NKI kernel layout: q (N, D, seq_q), k (N, D, seq_k), v (N, seq_k, D)
        # Full static-shape Q (garbage Q beyond valid_tokens is harmless)
        q_kern = roped_query[0].permute(1, 2, 0).contiguous()   # [N, D, seq_q]

        # K/V from buffer — already padded to multiple of 8192 at allocation
        k_kern = buffer_k[0].permute(1, 2, 0).contiguous()        # [N, D, seq_k]
        v_kern = buffer_v[0].permute(1, 0, 2).contiguous()        # [N, seq_k, D]

        if q_kern.device.type == "neuron" and self._nki_available:
            seqlen_k = k_kern.shape[2]
            seqlen_q_orig = q_kern.shape[2]
            assert seqlen_k % ATTN_SEQLEN_MULTIPLE == 0, f"k seqlen {seqlen_k} not multiple of {ATTN_SEQLEN_MULTIPLE}"

            # Pad seq_q to multiple of 128 (NKI tile size)
            P = 128
            pad_q = (P - seqlen_q_orig % P) % P
            if pad_q > 0:
                q_kern = torch.nn.functional.pad(q_kern, (0, pad_q))

            # Build mask: (128, seqlen_k) bf16, 0 for valid positions, -inf for masked
            mask = torch.zeros((P, seqlen_k), dtype=torch.bfloat16, device=q_kern.device)
            if k_len_int < seqlen_k:
                mask[:, k_len_int:] = float('-inf')

            num_sections = seqlen_k // ATTN_SEQLEN_MULTIPLE

            x = self._self_attn_kernel(
                q_kern, k_kern, v_kern, self.identity, mask,
                softmax_scale=self.softmax_scale,
                num_sections=num_sections,
            )
            # Output: [seq_q_padded, N, D] bfloat16 → slice to [seq_q, N, D] → [1, seq_q, C]
            x = x[:seqlen_q_orig].unsqueeze(0).flatten(2)
        else:
            # PyTorch fallback (CPU or Neuron without NKI)
            import torch.nn.functional as F
            q_attn = roped_query.permute(0, 2, 1, 3)
            k_attn = buffer_k[:, :k_len_int].permute(0, 2, 1, 3)
            v_attn = buffer_v[:, :k_len_int].permute(0, 2, 1, 3)
            attn_out = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            x = attn_out.permute(0, 2, 1, 3).flatten(2)

        if _profiling:
            _t_attention = (time.perf_counter() - _t4) * 1000
        # ── Phase 5: Output projection ──────────────────────────────────
        if _profiling:
            _t5 = time.perf_counter()
        x = self.o(x)
        if _profiling:
            _t_output = (time.perf_counter() - _t5) * 1000
            print(f"          [self_attn] qkv_rope={_t_qkv_rope:.1f}ms  cache={_t_cache:.1f}ms  assemble={_t_assemble:.1f}ms  attention={_t_attention:.1f}ms  output={_t_output:.1f}ms")
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_idx=0,
                 frame_length=1560):
        super().__init__()
        assert cross_attn_type == 't2v_cross_attn'
        assert cross_attn_norm
        self.layer_idx = layer_idx
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # norms
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)

        # attention
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx, frame_length)
        self.cross_attn = WanT2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)

        # ffn
        self.ffn = jit(nn.Sequential(
            nn.Linear(dim, ffn_dim), GELU(), nn.Linear(ffn_dim, dim)))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # jit helpers
        self._modulation_chunk = jit(modulation_chunk)
        self._modulated_norm_scale = jit(modulated_norm_scale)
        self._modulated_norm_shift = jit(modulated_norm_shift)
        self._modulated_residual = jit(modulated_residual)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        context,
        context_lens,
        updating_cache=False,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        num_valid_frames=None,
        shared_buffers=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(tuple): Python tuple (F, H, W)
            freqs_cos(Tensor): Rope cos, shape [1024, C / num_heads / 2]
            freqs_sin(Tensor): Rope sin, shape [1024, C / num_heads / 2]
        """
        _profiling = os.environ.get("PROFILE_PIPELINE", "0") == "1" and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) == 0 and self.layer_idx == 0
        if _profiling:
            _tb0 = time.perf_counter()

        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e0, e1, e2, e3, e4, e5 = self._modulation_chunk(self.modulation, e)

        if _profiling:
            _t_mod = (time.perf_counter() - _tb0) * 1000
            _tb1 = time.perf_counter()

        # self-attention
        norm_ones = torch.ones_like(e1)
        y = self.self_attn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(
                    self.norm1(x),
                    e1,
                    norm_ones,
                    num_frames,
                    frame_seqlen,
                ),
                e0,
            ),
            grid_sizes,
            freqs_cos,
            freqs_sin,
            kv_cache,
            current_start,
            cache_start,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
        )
        x = self._modulated_residual(x, y, e2, num_frames, frame_seqlen)

        if _profiling:
            _t_self = (time.perf_counter() - _tb1) * 1000
            _tb2 = time.perf_counter()

        # cross-attention
        x = x + self.cross_attn(
            self.norm3(x), context, context_lens,
            crossattn_cache=crossattn_cache)

        if _profiling:
            _t_cross = (time.perf_counter() - _tb2) * 1000
            _tb3 = time.perf_counter()

        # ffn
        y = self.ffn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(
                    self.norm2(x),
                    e4,
                    norm_ones,
                    num_frames,
                    frame_seqlen,
                ),
                e3,
            )
        )
        x = self._modulated_residual(x, y, e5, num_frames, frame_seqlen)

        if _profiling:
            _t_ffn = (time.perf_counter() - _tb3) * 1000
            _t_total = (time.perf_counter() - _tb0) * 1000
            print(f"        [block0] mod={_t_mod:.1f}ms  self_attn={_t_self:.1f}ms  cross_attn={_t_cross:.1f}ms  ffn={_t_ffn:.1f}ms  total={_t_total:.1f}ms")

        return x
