# Optimized version of causal_model.py:
# - Python int KV cache indices (no GPU scalar .item() calls)
# - Pre-allocated scratch buffers for cache eviction (no dynamic .clone())
# - Eliminated .clone() in updating_cache path (split anchor handling)
# - valid_q/valid_k masking via fake-batch-2 trick in flash_attn_varlen_b1
# - Debug prints removed
from wan.modules.attention_opt import flash_attn_varlen_b1
from wan.modules.model_opt import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math


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


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=1,
                 qk_norm=True,
                 eps=1e-6,
                 layer_idx=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_length = 1560
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.kv_cache_logical_size = 24 * self.frame_length  # 37440
        self.layer_idx = layer_idx

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        block_mask,
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
            seq_lens(Tensor): Shape [B]
            grid_sizes(tuple): Python tuple (F, H, W)
            freqs_cos(Tensor): Rope cos, shape [1024, C / num_heads / 2]
            freqs_sin(Tensor): Rope sin, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            num_valid_frames(int, optional): Number of valid (non-padding) frames
        """
        assert kv_cache is not None
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        assert b == 1, f"Batch size must be 1, got {b}"
        if cache_start is None:
            cache_start = current_start

        # ── Phase 1: QKV projection + RoPE ──────────────────────────────
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen
        current_start_frame_t = torch.tensor(current_start_frame, device=x.device)
        roped_query = causal_rope_apply(
            q, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t).type_as(v)
        roped_key = causal_rope_apply(
            k, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t).type_as(v)

        grid_sizes_one_block = (3, h, w)

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        # ── Phase 2: Cache management (write + eviction) ────────────────
        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        kv_cache_size = self.kv_cache_logical_size  # 37440 (logical, not tensor alloc)
        sink_tokens = self.block_length  # keep the first block (anchor) in cache

        # buffer_k/buffer_v: shared static-shaped buffers [B, max_buffer_size, N, D]
        # Used as scratch during eviction, then as assembled KV for attention.
        buffer_k, buffer_v = shared_buffers

        # Eviction: left-shift old entries when cache overflows
        num_evicted = 0
        if (num_new_tokens > 0) and (
                num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens  # static: 28080
            src_start = sink_tokens + num_evicted
            buffer_k[0, :evict_rolled].copy_(kv_cache["k"][0, src_start:src_start + evict_rolled])
            buffer_v[0, :evict_rolled].copy_(kv_cache["v"][0, src_start:src_start + evict_rolled])
            kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled].copy_(buffer_k[0, :evict_rolled])
            kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled].copy_(buffer_v[0, :evict_rolled])

        # Unified index computation
        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        # Write new block to cache
        if local_start_index == 0:
            kv_cache["k"][0, :self.block_length] = k[0, :self.block_length]  # anchor: un-roped
        else:
            kv_cache["k"][0, local_start_index:local_end_index] = roped_key[0, :self.block_length]
        kv_cache["v"][0, local_start_index:local_end_index] = v[0, :self.block_length]

        if num_new_tokens > 0:  # don't update indices when re-caching clean frame
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        # ── Phase 3: Assemble KV into buffers ────────────────────────────
        if updating_cache:
            # Cache-update call: attend over full cache
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)

            buffer_k[0, :cache_len].copy_(
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len])
            buffer_v[0, :cache_len].copy_(
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            # Overwrite anchor with RoPEd version if anchor is visible
            if cache_start_pos == 0:
                anchor_roped = causal_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin, start_frame=torch.tensor(0, device=v.device)).type_as(v)
                buffer_k[0, :self.block_length].copy_(anchor_roped[0])

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
                rope_start_frame = current_start_frame - wc_frame_length - 3
                anchor_roped = causal_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin, start_frame=torch.tensor(rope_start_frame, device=v.device)).type_as(v)
                buffer_k[0, :self.block_length].copy_(anchor_roped[0])
                buffer_v[0, :self.block_length].copy_(kv_cache["v"][0, :self.block_length])
                offset = self.block_length

                # Working cache
                buffer_k[0, offset:offset + wc_len].copy_(kv_cache["k"][0, wc_start:wc_start + wc_len])
                buffer_v[0, offset:offset + wc_len].copy_(kv_cache["v"][0, wc_start:wc_start + wc_len])
                offset += wc_len

            # Current tokens
            buffer_k[0, offset:offset + valid_tokens].copy_(roped_key[0, :valid_tokens])
            buffer_v[0, offset:offset + valid_tokens].copy_(v[0, :valid_tokens])
            k_len_int = offset + valid_tokens

        # ── Phase 4: Single attention call ──────────────────────────────
        x = flash_attn_varlen_b1(
            roped_query, buffer_k, buffer_v,
            valid_q=valid_tokens, valid_k=k_len_int)

        # ── Phase 5: Output projection ──────────────────────────────────
        x = x.flatten(2)
        x = self.o(x)
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
                 layer_idx=0):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs_cos,
        freqs_sin,
        context,
        context_lens,
        block_mask,
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
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(tuple): Python tuple (F, H, W)
            freqs_cos(Tensor): Rope cos, shape [1024, C / num_heads / 2]
            freqs_sin(Tensor): Rope sin, shape [1024, C / num_heads / 2]
            num_valid_frames(int, optional): Number of valid (non-padding) frames
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs_cos, freqs_sin, block_mask, kv_cache, current_start, cache_start,
            updating_cache=updating_cache, num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers)

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()

        assert model_type in ['t2v', 'i2v']
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

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps, layer_idx)
            for layer_idx in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        freqs_complex = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        self.freqs_cos = freqs_complex.real.clone()
        self.freqs_sin = freqs_complex.imag.clone()

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        updating_cache=False,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        num_valid_frames: int = None,
        shared_buffers=None,
    ):
        r"""
        Run the diffusion model with kv caching.
        """
        assert self.model_type == 't2v'
        assert x.shape[0] == 1  # batch_size must be 1
        assert not torch.is_grad_enabled()

        # params
        device = self.patch_embedding.weight.device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)

        # embeddings (batch_size=1: operate on tensor directly, no list comp)
        x = self.patch_embedding(x)                          # [1, dim, f, h, w]
        grid_sizes = tuple(int(d) for d in x.shape[2:])      # (f, h, w)
        x = x.flatten(2).transpose(1, 2)                     # [1, f*h*w, dim]
        seq_lens = torch.tensor([x.size(1)], dtype=torch.long)
        assert seq_lens.max() <= seq_len

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # context (batch_size=1: embed directly, tokenizer already pads to text_len)
        context_lens = None
        assert context.size(1) == self.text_len
        context = self.text_embedding(context)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos,
            freqs_sin=self.freqs_sin,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            updating_cache=updating_cache,
            num_valid_frames=num_valid_frames,
            shared_buffers=shared_buffers,
        )

        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index],
                    "current_start": current_start,
                    "cache_start": cache_start
                }
            )
            x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # Flatten from [B, F, seq, out_C] to [B, F*seq, out_C] so unpatchify
        # can slice the valid tokens correctly when there's padding
        x = x.flatten(1, 2)
        # unpatchify (batch_size=1: returns single [C, F, H, W] tensor)
        return self.unpatchify(x, grid_sizes).unsqueeze(0)

    def forward(
        self,
        *args,
        **kwargs
    ):
        assert kwargs.get('kv_cache', None) is not None
        return self._forward_inference(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensor from patch embeddings.
        grid_sizes: Python tuple (f, h, w). Assumes batch_size=1.
        """
        c = self.out_dim
        f, h, w = grid_sizes
        u = x.squeeze(0).view(f, h, w, *self.patch_size, c)
        u = u.permute(6, 0, 3, 1, 4, 2, 5).contiguous()
        u = u.reshape(c, f * self.patch_size[0], h * self.patch_size[1], w * self.patch_size[2])
        return u

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
