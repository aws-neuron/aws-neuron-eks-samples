"""SP-aware CausalWanSelfAttention — sequence parallelism support.

SP sharding:
  - Input x to the BLOCK is [1, L/SP, dim] (sequence-sharded at model level)
  - QKV projection on local shard: [1, L/SP, dim_local]
  - AllGather K/V across SP group: [1, L, dim_local] for cache + full attention
  - Q stays local: [1, L/SP, dim_local] → reduces attention compute
  - KV cache stores full sequence (not SP-sharded)
  - Output is [1, L/SP, dim] (stays sharded for next block)
"""
import math
import torch
import torch.nn as nn

from models.layers import (
    WanRMSNorm,
    ATTN_SEQLEN_MULTIPLE,
    causal_rope_rotation_nki,
    ROPE_NKI_AVAILABLE,
    wan_flash_self_attn_nki,
    SELF_ATTN_NKI_AVAILABLE,
    KV_CACHE_NKI_AVAILABLE,
    _nki_cache_copy,
    _nki_kv_cache_copy,
    jit,
)
from models import parallel_state as ps


class CausalWanSelfAttentionSP(nn.Module):

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
        assert qk_norm
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.eps = eps
        self.frame_length = frame_length
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.kv_cache_logical_size = 24 * self.frame_length
        self.layer_idx = layer_idx

        # SP/TP config
        self.tp_degree = ps.get_world_size("attn-tp")
        self.sp_degree = ps.get_world_size("attn-sp")
        self.tp_rank = ps.get_rank("attn-tp")
        self.sp_rank = ps.get_rank("attn-sp")
        self.heads_per_shard = num_heads // self.tp_degree

        # Layers — full size, shard_model_tp() replaces with Parallel variants
        self.q = jit(nn.Linear(dim, dim))
        self.k = jit(nn.Linear(dim, dim))
        self.v = jit(nn.Linear(dim, dim))
        self.o = jit(nn.Linear(dim, dim))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        # NKI kernels
        self._rope_kernel = causal_rope_rotation_nki
        self._self_attn_kernel = wan_flash_self_attn_nki

        # Buffers
        self.register_buffer('identity', torch.eye(self.head_dim), persistent=False)
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

        sign_pattern = torch.ones(self.head_dim, dtype=torch.float32)
        sign_pattern[0::2] = -1.0
        self.register_buffer(
            'sign_pattern',
            sign_pattern.unsqueeze(0).expand(128, -1).contiguous(),
            persistent=False)

    def _nki_rope_apply(self, x, grid_sizes, freqs_cos, freqs_sin, start_frame):
        """Same RoPE as base class."""
        b, s, n, d = x.shape
        f, h, w = grid_sizes
        seq_len = f * h * w
        c = d // 2
        s0 = c - 2 * (c // 3)
        s1 = c // 3

        frame_idx = start_frame + torch.arange(f, device=x.device)

        cos_half = torch.cat([
            torch.index_select(freqs_cos[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_cos[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_cos[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        sin_half = torch.cat([
            torch.index_select(freqs_sin[:, :s0], 0, frame_idx).view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_sin[:h, s0:s0 + s1].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_sin[:w, s0 + s1:].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, c)

        cos_expanded = cos_half.repeat_interleave(2, dim=-1)
        sin_expanded = sin_half.repeat_interleave(2, dim=-1)
        sign = torch.ones(d, device=x.device, dtype=sin_expanded.dtype)
        sign[0::2] = -1.0
        sin_signed = sin_expanded * sign.unsqueeze(0)
        cos_sin = torch.cat([cos_expanded, sin_signed], dim=-1).contiguous()

        P = 128
        pad = (P - seq_len % P) % P
        cos_sin = torch.nn.functional.pad(cos_sin, (0, 0, 0, pad))
        x_nki = torch.nn.functional.pad(x[0, :seq_len], (0, 0, 0, 0, 0, pad))

        out = self._rope_kernel(x_nki, cos_sin, num_heads=n, head_dim=d)
        return out[:seq_len].unsqueeze(0).type_as(x)

    def cache_copy_inplace(self, k_dst, k_src, v_dst=None, v_src=None):
        if KV_CACHE_NKI_AVAILABLE and v_dst is not None:
            _nki_kv_cache_copy(k_dst, k_src, v_dst, v_src)
        elif KV_CACHE_NKI_AVAILABLE:
            _nki_cache_copy(k_dst, k_src)
        else:
            k_dst.copy_(k_src)
            if v_dst is not None:
                v_dst.copy_(v_src)

    def _call_self_attn_nki(self, q, k, v, identity, mask, softmax_scale, num_sections):
        return self._self_attn_kernel(q, k, v, identity, mask,
                                      softmax_scale=softmax_scale,
                                      num_sections=num_sections)

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
        shared_buffers=None,
        current_start_frame_t=None
    ):
        """SP-aware self-attention.

        x is [1, L/SP, dim] — already sequence-sharded by the model.
        """
        b = 1
        s_local = x.shape[1]  # L / SP
        n_local = self.heads_per_shard
        d = self.head_dim
        f, h, w = grid_sizes
        frame_seqlen = h * w
        L = f * h * w  # full sequence length

        # ── Phase 1: QKV on local shard + AllGather K/V ─────────────────
        # QKV projection on SP shard (ColumnParallel: output dim_local)
        q_local = self.norm_q(self.q(x)).view(b, s_local, n_local, d)
        k_local = self.norm_k(self.k(x)).view(b, s_local, n_local, d)
        v_local = self.v(x).view(b, s_local, n_local, d)

        # AllGather K and V across SP to get full sequence
        if self.sp_degree > 1:
            k_full = torch.empty(b, L, n_local, d, dtype=k_local.dtype, device=k_local.device)
            v_full = torch.empty(b, L, n_local, d, dtype=v_local.dtype, device=v_local.device)
            ps.all_gather_into_tensor(
                k_full.view(L, n_local * d),
                k_local.view(s_local, n_local * d), "attn-sp")
            ps.all_gather_into_tensor(
                v_full.view(L, n_local * d),
                v_local.view(s_local, n_local * d), "attn-sp")
            k_full = k_full.view(b, L, n_local, d)
            v_full = v_full.view(b, L, n_local, d)
        else:
            k_full = k_local
            v_full = v_local

        # ── Phase 1b: RoPE on full sequence ─────────────────────────────
        # AllGather Q across SP too (for correct RoPE positions)
        if self.sp_degree > 1:
            q_full = torch.empty(b, L, n_local, d, dtype=q_local.dtype, device=q_local.device)
            ps.all_gather_into_tensor(
                q_full.view(L, n_local * d),
                q_local.view(s_local, n_local * d), "attn-sp")
            q_full = q_full.view(b, L, n_local, d)
        else:
            q_full = q_local

        # RoPE on full Q and K (correct 3D positions)
        roped_query_full = self._nki_rope_apply(
            q_full, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)
        roped_key = self._nki_rope_apply(
            k_full, grid_sizes, freqs_cos, freqs_sin, start_frame=current_start_frame_t)

        # Slice Q back to SP shard (SP savings: attend with fewer Q tokens)
        sp_start_token = self.sp_rank * s_local
        roped_query = roped_query_full[:, sp_start_token:sp_start_token + s_local]

        # K/V for cache (full sequence, local heads)
        k = k_full
        v = v_full

        num_frames_per_block = self.block_length // self.frame_length
        grid_sizes_one_block = (num_frames_per_block, h, w)

        # ── Phase 2: Cache management (operates on full sequence) ───────
        if cache_start is None:
            cache_start = current_start
        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = cache_end - global_end_index
        kv_cache_size = self.kv_cache_logical_size
        sink_tokens = self.block_length

        buffer_k, buffer_v = shared_buffers

        num_evicted = 0
        if (num_new_tokens > 0) and (
                num_new_tokens + local_end_index_current > kv_cache_size):
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
            evict_rolled = kv_cache_size - 2 * sink_tokens
            src_start = sink_tokens + num_evicted
            self.cache_copy_inplace(
                buffer_k[0, :evict_rolled], kv_cache["k"][0, src_start:src_start + evict_rolled],
                buffer_v[0, :evict_rolled], kv_cache["v"][0, src_start:src_start + evict_rolled])
            self.cache_copy_inplace(
                kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled], buffer_k[0, :evict_rolled],
                kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled], buffer_v[0, :evict_rolled])

        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        if local_start_index == 0:
            self.cache_copy_inplace(
                kv_cache["k"][0, :self.block_length], k[0, :self.block_length],
                kv_cache["v"][0, :self.block_length], v[0, :self.block_length])
        else:
            self.cache_copy_inplace(
                kv_cache["k"][0, local_start_index:local_end_index], roped_key[0, :self.block_length],
                kv_cache["v"][0, local_start_index:local_end_index], v[0, :self.block_length])

        if num_new_tokens > 0:
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        # ── Phase 3: Assemble KV (same as base — full sequence) ─────────
        if updating_cache:
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)

            self.cache_copy_inplace(
                buffer_k[0, :cache_len],
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len],
                buffer_v[0, :cache_len],
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            if cache_start_pos == 0:
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(0, device=v.device))
                self.cache_copy_inplace(
                    buffer_k[0, :self.block_length], anchor_roped[0])

            k_len_int = cache_len
        else:
            valid_tokens = num_valid_frames * frame_seqlen if num_valid_frames is not None else f * h * w
            offset = 0
            if local_start_index > 0:
                wc_max = self.max_attention_size - valid_tokens - self.block_length
                wc_end = local_start_index
                wc_start = max(self.block_length, wc_end - wc_max)
                wc_len = wc_end - wc_start

                wc_frame_length = wc_len // self.frame_length
                current_start_frame = current_start // frame_seqlen
                rope_start_frame = current_start_frame - wc_frame_length - num_frames_per_block
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(rope_start_frame, device=v.device))
                self.cache_copy_inplace(
                    buffer_k[0, :self.block_length], anchor_roped[0],
                    buffer_v[0, :self.block_length], kv_cache["v"][0, :self.block_length])
                offset = self.block_length

                if wc_len > 0:
                    self.cache_copy_inplace(
                        buffer_k[0, offset:offset + wc_len], kv_cache["k"][0, wc_start:wc_start + wc_len],
                        buffer_v[0, offset:offset + wc_len], kv_cache["v"][0, wc_start:wc_start + wc_len])
                offset += wc_len

            self.cache_copy_inplace(
                buffer_k[0, offset:offset + valid_tokens], roped_key[0, :valid_tokens],
                buffer_v[0, offset:offset + valid_tokens], v[0, :valid_tokens])
            k_len_int = offset + valid_tokens

        # ── Phase 4: Attention (Q is SP-local, K/V full from buffer) ────
        q_kern = roped_query[0].permute(1, 2, 0).contiguous()   # [n_local, d, L/SP]
        k_kern = buffer_k[0].permute(1, 2, 0).contiguous()       # [n_local, d, buf_size]
        v_kern = buffer_v[0].permute(1, 0, 2).contiguous()       # [n_local, buf_size, d]

        seqlen_k = k_kern.shape[2]
        seqlen_q_orig = q_kern.shape[2]

        P = 128
        pad_q = (P - seqlen_q_orig % P) % P
        q_kern = torch.nn.functional.pad(q_kern, (0, pad_q))

        mask = torch.zeros((P, seqlen_k), dtype=torch.bfloat16, device=q_kern.device)
        if k_len_int < seqlen_k:
            mask[:, k_len_int:] = float('-inf')

        num_sections = seqlen_k // ATTN_SEQLEN_MULTIPLE

        x = self._call_self_attn_nki(
            q_kern, k_kern, v_kern, self.identity, mask,
            softmax_scale=self.softmax_scale,
            num_sections=num_sections,
        )
        x = x[:seqlen_q_orig].unsqueeze(0).flatten(2)  # [1, L/SP, dim_local]

        # ── Phase 5: O projection (RowParallel handles all_reduce)
        x = self.o(x)
        return x  # [1, L/SP, dim]
