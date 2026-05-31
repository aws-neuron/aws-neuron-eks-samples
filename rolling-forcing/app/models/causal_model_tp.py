"""TP-aware CausalWanModel for Wan2.1-T2V-14B on Trainium.

This extends the Neuron-compatible CausalWanModel to support tensor parallelism.
Key differences from the single-rank version:
  - num_heads is the LOCAL head count (num_heads_total // tp_degree)
  - Q/K/V projections output dim//tp_degree features
  - O projection takes dim//tp_degree features and all-reduces to dim
  - FFN fc1 outputs ffn_dim//tp_degree, fc2 takes ffn_dim//tp_degree and all-reduces
  - KV cache is sized for local heads only
  - Cross-attention operates on local heads

The model is first loaded with full weights via from_pretrained(), then
shard_model_tp() is called to split weights and replace Linear layers with
ColumnParallelLinear/RowParallelLinear.
"""

import math
import os
import time

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from models.layers import (
    GELU,
    SiLU,
    WanPatchEmbed,
    WanLayerNorm,
    WanRMSNorm,
    WanFFN,
    CausalHead,
    CausalWanSelfAttention,
    WanT2VCrossAttention,
    rope_params,
    sinusoidal_embedding_1d,
    unpatchify,
    modulation_chunk,
    modulated_norm_scale,
    modulated_norm_shift,
    modulated_residual,
)
from models.tp_utils import all_reduce_sum, get_tp_rank, get_tp_world_size
from models import parallel_state as ps


def expand_e_for_sp_shard(e0, sp_rank, sp_degree, num_frames, frame_seqlen):
    """Expand per-frame time embedding to per-token, then slice to SP shard.

    e0: [B, num_frames, 6, dim] — per-frame embedding
    Returns: [B, num_frames_local, 6, dim] for this SP rank's token shard.

    With SP=2 and non-frame-aligned shards, we expand to all tokens
    then slice to the SP shard.
    """
    if sp_degree == 1:
        return e0
    B, F, I, C = e0.shape
    L = F * frame_seqlen
    shard_len = L // sp_degree
    sp_start = sp_rank * shard_len

    # Figure out which frames this shard covers
    start_frame = sp_start // frame_seqlen
    end_frame = (sp_start + shard_len + frame_seqlen - 1) // frame_seqlen
    end_frame = min(end_frame, F)
    F_sub = end_frame - start_frame
    start_off = sp_start - start_frame * frame_seqlen

    # Expand e to per-token for the covered frames
    e_sub = e0[:, start_frame:end_frame]  # [B, F_sub, I, C]
    e_exp = e_sub.unsqueeze(3).expand(B, F_sub, I, frame_seqlen, C)
    e_exp = e_exp.reshape(B, F_sub * frame_seqlen, I, C)

    # Slice to our shard
    e_shard = e_exp[:, start_off:start_off + shard_len]  # [B, shard_len, I, C]

    # Reshape back to [B, num_frames_local, I, C] for modulation functions
    # num_frames_local = shard_len // frame_seqlen (if evenly divisible)
    # Otherwise we keep per-token and modify the block to handle it
    return e_shard  # [B, shard_len, I, C] — per-token, not per-frame

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn=None, **kwargs):
    """No-op wrapper since torch_neuronx.jit is not available."""
    if fn is None:
        return lambda f: f
    return fn


def _init_rope_freqs(dim, num_heads):
    """Precompute 3D RoPE cos/sin frequencies.

    Note: dim and num_heads here are the FULL model values (not per-rank),
    since RoPE frequencies are determined by head_dim which doesn't change with TP.
    """
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads  # head_dim = 128 for 14B
    cos_0, sin_0 = rope_params(1024, d - 4 * (d // 6))
    cos_1, sin_1 = rope_params(1024, 2 * (d // 6))
    cos_2, sin_2 = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_0, cos_1, cos_2], dim=1), torch.cat([sin_0, sin_1, sin_2], dim=1)


class CausalWanAttentionBlockTP(nn.Module):
    """TP-aware attention block.

    After shard_model_tp() is called:
      - self_attn.q/k/v are ColumnParallelLinear (no comm)
      - self_attn.o is RowParallelLinear (all-reduce in forward)
      - cross_attn.q/k/v are ColumnParallelLinear
      - cross_attn.o is RowParallelLinear (all-reduce in forward)
      - ffn[0]/fc1 is ColumnParallelLinear
      - ffn[2]/fc2 is RowParallelLinear (all-reduce in forward)

    The all-reduces are embedded in RowParallelLinear.forward(), so this
    block's forward() is structurally identical to the non-TP version.
    """

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

        # norms (replicated — small parameters)
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)

        # attention (will be sharded by shard_model_tp)
        sp_degree = ps.get_world_size("attn-sp") if ps.is_registered("attn-sp") else 1
        if sp_degree > 1:
            from models.self_attn_sp import CausalWanSelfAttentionSP
            self.self_attn = CausalWanSelfAttentionSP(
                dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx, frame_length)
        else:
            self.self_attn = CausalWanSelfAttention(
                dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx, frame_length)
        self.cross_attn = WanT2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)

        # FFN (will be sharded by shard_model_tp)
        self.ffn = jit(nn.Sequential(
            nn.Linear(dim, ffn_dim), GELU(), nn.Linear(ffn_dim, dim)))

        # modulation (replicated — small, applied before attention)
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
        shared_buffers=None,
        current_start_frame_t=None,
        sp_mode=False,
        cache_update_start=None,
        nfpb_cu=None,
    ):
        if sp_mode:
            # SP mode: e is per-token [B, shard_len, 6, C], x is [B, shard_len, C]
            # Simple element-wise modulation (no unflatten/reshape)
            e0, e1, e2, e3, e4, e5 = e[:, :, 0], e[:, :, 1], e[:, :, 2], e[:, :, 3], e[:, :, 4], e[:, :, 5]
            ones = torch.ones_like(e1)

            # self-attention
            y = self.self_attn(
                self.norm1(x) * (ones + e1) + e0,
                grid_sizes, freqs_cos, freqs_sin,
                kv_cache, current_start, cache_start,
                updating_cache=updating_cache,
                num_valid_frames=num_valid_frames,
                shared_buffers=shared_buffers,
                current_start_frame_t=current_start_frame_t,
                cache_update_start=cache_update_start,
                nfpb_cu=nfpb_cu,
            )
            x = x + y * e2

            # cross-attention
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens,
                crossattn_cache=crossattn_cache)

            # FFN
            y = self.ffn(self.norm2(x) * (ones + e4) + e3)
            x = x + y * e5

            return x

        # Non-SP mode: original per-frame modulation
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        e0, e1, e2, e3, e4, e5 = self._modulation_chunk(self.modulation, e)

        # self-attention (all-reduce inside self_attn.o)
        norm_ones = torch.ones_like(e1)
        y = self.self_attn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(
                    self.norm1(x), e1, norm_ones, num_frames, frame_seqlen),
                e0),
            grid_sizes,
            freqs_cos,
            freqs_sin,
            kv_cache,
            current_start,
            cache_start,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
            current_start_frame_t=current_start_frame_t,
        )
        x = self._modulated_residual(x, y, e2, num_frames, frame_seqlen)

        # cross-attention (all-reduce inside cross_attn.o)
        x = x + self.cross_attn(
            self.norm3(x), context, context_lens,
            crossattn_cache=crossattn_cache)

        # FFN (all-reduce inside ffn[2] / fc2)
        y = self.ffn(
            self._modulated_norm_shift(
                self._modulated_norm_scale(
                    self.norm2(x), e4, norm_ones, num_frames, frame_seqlen),
                e3)
        )
        x = self._modulated_residual(x, y, e5, num_frames, frame_seqlen)

        return x


class CausalWanModelTP(ModelMixin, ConfigMixin):
    """TP-aware Wan diffusion backbone for Wan2.1-T2V-14B.

    This model is loaded with full weights via from_pretrained(), then
    shard_model_tp() splits the weights across TP ranks.

    After sharding:
      - Each rank has num_heads // tp_degree attention heads
      - KV cache is sized for local heads
      - All-reduces happen inside RowParallelLinear layers
    """

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['CausalWanAttentionBlockTP']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=5120,
                 ffn_dim=13824,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=40,
                 num_layers=40,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 frame_length=1560):
        super().__init__()

        assert model_type == 't2v'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings — replicated across all ranks (small)
        self.patch_embedding = WanPatchEmbed(in_dim, dim, patch_size)
        self.text_embedding = jit(nn.Sequential(
            nn.Linear(text_dim, dim), GELU(), nn.Linear(dim, dim)))
        self.time_embedding = jit(nn.Sequential(
            nn.Linear(freq_dim, dim), SiLU(), nn.Linear(dim, dim)))
        self.time_projection = jit(nn.Sequential(
            SiLU(), nn.Linear(dim, dim * 6)))

        # Transformer blocks (will be TP-sharded)
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlockTP(
                't2v_cross_attn', dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm,
                eps, layer_idx, frame_length)
            for layer_idx in range(num_layers)
        ])

        # Output head — replicated (small)
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # jit-wrapped standalone functions
        self._sinusoidal_embedding_1d = jit(sinusoidal_embedding_1d)
        self._unpatchify = jit(unpatchify)

        # RoPE frequencies — same for all ranks (head_dim unchanged by TP)
        self.freqs_cos, self.freqs_sin = _init_rope_freqs(dim, num_heads)

        # TP metadata (set by shard_model_tp)
        self.tp_degree = 1
        self.tp_rank = 0
        self.num_heads_per_rank = num_heads

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _update_frame_length(self, new_frame_length: int, num_frame_per_block: int = 3):
        """Update frame_length in all attention blocks after loading."""
        block_length = num_frame_per_block * new_frame_length
        for block in self.blocks:
            attn = block.self_attn
            attn.frame_length = new_frame_length
            attn.block_length = block_length
            attn.max_attention_size = 21 * new_frame_length
            attn.kv_cache_logical_size = 24 * new_frame_length
        print(f"[CausalWanModelTP] Updated frame_length={new_frame_length}, "
              f"block_length={block_length} in {len(self.blocks)} blocks")

    def _forward_inference(
        self,
        x,
        t,
        context,
        updating_cache=False,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        num_valid_frames: int = None,
        shared_buffers=None,
        cache_update_start: int = None,
        nfpb_cu: int = None,
    ):
        """Run the DiT forward pass with TP.

        When cache_update_start is provided (merged mode), x contains
        [cu_frames | dn_frames]. Process all through embeddings/blocks,
        but self_attn handles cache-update for first nfpb_cu frames
        and denoising for the rest.

        All ranks execute the same code in lockstep. Communication
        (all-reduce) happens inside RowParallelLinear layers.
        """
        assert self.model_type == 't2v'
        assert x.shape[0] == 1  # batch_size must be 1
        assert not torch.is_grad_enabled()

        device = self.patch_embedding.weight.device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)

        # Patch embedding (replicated)
        # .contiguous() needed: Neuron backend rejects non-contiguous tensors
        x = self.patch_embedding(x.contiguous())
        grid_sizes = tuple(int(d) for d in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)

        # Time embedding (replicated)
        e = self.time_embedding(
            self._sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x).contiguous())
        e0 = self.time_projection(e.contiguous()).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # Context embedding (replicated)
        context_lens = None
        assert context.size(1) == self.text_len
        context = self.text_embedding(context.contiguous())

        # Pre-compute cross-attention K/V for all layers (outside compiled blocks)
        # This avoids a graph-breaking `if not is_init` branch inside each block.
        for block_index, block in enumerate(self.blocks):
            cache = crossattn_cache[block_index]
            if not cache["is_init"]:
                b_ctx = context.size(0)
                n_heads = block.cross_attn.num_heads
                d_head = block.cross_attn.head_dim
                cache["k"] = block.cross_attn.norm_k(
                    block.cross_attn.k(context)).view(b_ctx, -1, n_heads, d_head)
                cache["v"] = block.cross_attn.v(context).view(b_ctx, -1, n_heads, d_head)
                cache["is_init"] = True

        # Transformer blocks (TP-sharded, optionally SP-sharded)
        sp_degree = ps.get_world_size("attn-sp") if ps.is_registered("attn-sp") else 1
        sp_mode = sp_degree > 1

        num_frames = e0.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        current_start_frame_t = torch.tensor(
            current_start // frame_seqlen, dtype=torch.int64, device=x.device)

        if sp_mode:
            sp_rank = ps.get_rank("attn-sp")
            L = x.shape[1]
            shard_len = L // sp_degree
            sp_start = sp_rank * shard_len

            # Shard x along sequence
            x = x[:, sp_start:sp_start + shard_len].contiguous()

            # Expand e0 to per-token and shard
            # e0: [B, num_frames, 6, dim] → [B, num_frames, 1, 6, dim]
            #   → expand to [B, num_frames, frame_seqlen, 6, dim]
            #   → reshape to [B, L, 6, dim] → shard
            e_expanded = e0.unsqueeze(2).expand(
                -1, -1, frame_seqlen, -1, -1).reshape(1, L, 6, self.dim)
            e_shard = e_expanded[:, sp_start:sp_start + shard_len].contiguous()

            # Also shard modulation bias per block (add once here)
            # modulation is [1, 6, dim] — needs to be added to e per-token
            # Do this per-block inside the loop (modulation is per-block param)

        kwargs = dict(
            grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos,
            freqs_sin=self.freqs_sin,
            context=context,
            context_lens=context_lens,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
            sp_mode=sp_mode,
            cache_update_start=cache_update_start,
            nfpb_cu=nfpb_cu,
        )

        if not sp_mode:
            kwargs["e"] = e0

        for block_index, block in enumerate(self.blocks):
            if sp_mode:
                # Add per-block modulation bias to per-token e shard
                mod = block.modulation  # [1, 6, dim]
                e_with_mod = e_shard + mod.unsqueeze(1)  # [1, shard_len, 6, dim]
                kwargs["e"] = e_with_mod

            kwargs.update({
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index],
                "current_start": current_start,
                "cache_start": cache_start,
                "current_start_frame_t": current_start_frame_t,
            })
            x = block(x, **kwargs)

        if sp_mode:
            # AllGather x back to full sequence for head + unpatchify
            x_full = torch.empty(1, L, self.dim, dtype=x.dtype, device=x.device)
            ps.all_gather_into_tensor(
                x_full.view(L, self.dim),
                x.view(shard_len, self.dim), "attn-sp")
            x = x_full

        # Head + unpatchify (replicated)
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = x.flatten(1, 2)
        result = self._unpatchify(x, self.out_dim, self.patch_size, grid_sizes).unsqueeze(0)
        return result

    def forward(self, *args, **kwargs):
        assert kwargs.get('kv_cache', None) is not None
        return self._forward_inference(*args, **kwargs)
