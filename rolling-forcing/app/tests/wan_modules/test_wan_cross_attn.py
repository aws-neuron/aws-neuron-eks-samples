import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_neuronx.jit import jit

from kernels.cross_attention import wan_cross_attn
from models.layers import WanRMSNorm
from models.layers import WanT2VCrossAttention


class RefCrossAttention(nn.Module):
    """CPU reference cross-attention (same structure as GPU WanT2VCrossAttention)."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        assert qk_norm is True
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x, context, context_lens, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        assert crossattn_cache is not None
        if not crossattn_cache["is_init"]:
            crossattn_cache["is_init"] = True
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
            crossattn_cache["k"] = k
            crossattn_cache["v"] = v
        else:
            k = crossattn_cache["k"]
            v = crossattn_cache["v"]

        x = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)

        x = x.flatten(2)
        x = self.o(x)
        return x


def test_cross_attn_q_proj():
    """Test q = self.norm_q(self.q(x)).view(b, -1, n, d)"""
    dtype = torch.bfloat16
    B, L_q, dim, num_heads = 1, 23400, 1536, 12
    head_dim = dim // num_heads  # 128

    x = torch.randn(B, L_q, dim, dtype=dtype)
    q_proj = nn.Linear(dim, dim).to(dtype)
    norm_q = WanRMSNorm(dim).to(dtype)

    # CPU reference
    expected = norm_q(q_proj(x)).view(B, -1, num_heads, head_dim)

    # Neuron: chain JIT modules
    q_proj_neuron = jit(q_proj).to("neuron")
    norm_q_neuron = norm_q.to("neuron")
    x_neuron = x.to("neuron")
    result = norm_q_neuron(q_proj_neuron(x_neuron)).view(B, -1, num_heads, head_dim)

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_cross_attn_kv_proj():
    """Test k = norm_k(k_proj(context)).view(...), v = v_proj(context).view(...)"""
    dtype = torch.bfloat16
    B, L_kv, dim, num_heads = 1, 512, 1536, 12
    head_dim = dim // num_heads  # 128

    context = torch.randn(B, L_kv, dim, dtype=dtype)
    k_proj = nn.Linear(dim, dim).to(dtype)
    norm_k = WanRMSNorm(dim).to(dtype)
    v_proj = nn.Linear(dim, dim).to(dtype)

    # CPU reference
    expected_k = norm_k(k_proj(context)).view(B, -1, num_heads, head_dim)
    expected_v = v_proj(context).view(B, -1, num_heads, head_dim)

    # Neuron: chain JIT modules
    k_proj_neuron = jit(k_proj).to("neuron")
    norm_k_neuron = norm_k.to("neuron")
    v_proj_neuron = jit(v_proj).to("neuron")
    ctx_neuron = context.to("neuron")

    result_k = norm_k_neuron(k_proj_neuron(ctx_neuron)).view(B, -1, num_heads, head_dim)
    result_v = v_proj_neuron(ctx_neuron).view(B, -1, num_heads, head_dim)

    torch.testing.assert_close(result_k.cpu(), expected_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(result_v.cpu(), expected_v, rtol=1e-2, atol=1e-2)


def test_cross_attn_out_proj():
    """Test x = x.flatten(2); x = self.o(x)"""
    dtype = torch.bfloat16
    B, L_q, num_heads, head_dim = 1, 23400, 12, 128
    dim = num_heads * head_dim  # 1536

    x = torch.randn(B, L_q, num_heads, head_dim, dtype=dtype)
    o_proj = nn.Linear(dim, dim).to(dtype)

    # CPU reference
    expected = o_proj(x.flatten(2))

    # Neuron
    o_proj_neuron = jit(o_proj).to("neuron")
    result = o_proj_neuron(x.to("neuron").flatten(2))

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_cross_attention_e2e_pure_compute():
    """Pure-computation test: manual sub-components + wan_cross_attn kernel."""
    dtype = torch.bfloat16
    dim, num_heads = 1536, 12
    head_dim = dim // num_heads  # 128
    seqlen_q, seqlen_k = 23400, 512

    # Create sub-components on CPU (raw, no JIT)
    q_proj = nn.Linear(dim, dim).to(dtype)
    k_proj = nn.Linear(dim, dim).to(dtype)
    v_proj = nn.Linear(dim, dim).to(dtype)
    o_proj = nn.Linear(dim, dim).to(dtype)
    norm_q = WanRMSNorm(dim).to(dtype)
    norm_k = WanRMSNorm(dim).to(dtype)

    # Test inputs
    x = torch.randn(1, seqlen_q, dim, dtype=dtype)
    context = torch.randn(1, seqlen_k, dim, dtype=dtype)

    # CPU reference: projections + norms
    q = norm_q(q_proj(x)).view(1, seqlen_q, num_heads, head_dim)
    k = norm_k(k_proj(context)).view(1, seqlen_k, num_heads, head_dim)
    v = v_proj(context).view(1, seqlen_k, num_heads, head_dim)

    # CPU reference: standard scaled dot-product attention
    q_ref = q.permute(0, 2, 1, 3)
    k_ref = k.permute(0, 2, 1, 3)
    v_ref = v.permute(0, 2, 1, 3)
    attn_out = F.scaled_dot_product_attention(q_ref, k_ref, v_ref)
    attn_out = attn_out.permute(0, 2, 1, 3)
    expected = o_proj(attn_out.flatten(2))

    # Neuron: JIT the same layers, run through wan_cross_attn kernel
    q_proj_n = jit(q_proj).to("neuron")
    k_proj_n = jit(k_proj).to("neuron")
    v_proj_n = jit(v_proj).to("neuron")
    o_proj_n = jit(o_proj).to("neuron")
    norm_q_n = norm_q.to("neuron")
    norm_k_n = norm_k.to("neuron")

    x_n = x.to("neuron")
    ctx_n = context.to("neuron")

    q_n = norm_q_n(q_proj_n(x_n)).view(1, seqlen_q, num_heads, head_dim)
    k_n = norm_k_n(k_proj_n(ctx_n)).view(1, seqlen_k, num_heads, head_dim)
    v_n = v_proj_n(ctx_n).view(1, seqlen_k, num_heads, head_dim)

    # Reshape for kernel
    q_kern = q_n[0].permute(1, 2, 0).contiguous()
    k_kern = k_n[0].permute(1, 2, 0).contiguous()
    v_kern = v_n[0].permute(1, 0, 2).contiguous()
    identity = torch.eye(head_dim, dtype=dtype).to("neuron")
    softmax_scale = 1.0 / math.sqrt(head_dim)

    attn_n = wan_cross_attn(q_kern, k_kern, v_kern, identity, softmax_scale=softmax_scale)
    # Kernel output: [L1, num_heads, head_dim] in q.dtype → [B, L1, C]
    result = o_proj_n(attn_n.unsqueeze(0).flatten(2))

    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


def test_cross_attention_e2e_pure_module():
    """End-to-end: RefCrossAttention (CPU) vs WanT2VCrossAttention (Neuron)."""
    dtype = torch.bfloat16
    dim, num_heads = 1536, 12
    seqlen_q, seqlen_k = 23400, 512

    x = torch.randn(1, seqlen_q, dim, dtype=dtype)
    context = torch.randn(1, seqlen_k, dim, dtype=dtype)

    # ── CPU reference ──
    cpu_module = RefCrossAttention(dim, num_heads).to(dtype)
    cpu_cache = {"is_init": False, "k": None, "v": None}
    expected = cpu_module(x, context, context_lens=None, crossattn_cache=cpu_cache)

    # ── Neuron ──
    neuron_module = WanT2VCrossAttention(dim, num_heads).to(dtype)

    cpu_sd = cpu_module.state_dict()
    neuron_module.load_state_dict(cpu_sd, strict=False)

    neuron_module = neuron_module.to("neuron")

    neuron_cache = {"is_init": False, "k": None, "v": None}
    result = neuron_module(
        x.to("neuron"), context.to("neuron"),
        context_lens=None, crossattn_cache=neuron_cache)

    # Compare output
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)
    # Compare crossattn_cache
    torch.testing.assert_close(neuron_cache["k"].cpu(), cpu_cache["k"], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(neuron_cache["v"].cpu(), cpu_cache["v"], rtol=1e-2, atol=1e-2)

    # ── Second call: cached path (is_init=True) ──
    # Use a different x to verify q recomputation while k/v come from cache
    x2 = torch.randn(1, seqlen_q, dim, dtype=dtype)

    expected2 = cpu_module(x2, context, context_lens=None, crossattn_cache=cpu_cache)
    result2 = neuron_module(
        x2.to("neuron"), context.to("neuron"),
        context_lens=None, crossattn_cache=neuron_cache)

    torch.testing.assert_close(result2.cpu(), expected2, rtol=1e-2, atol=1e-2)
