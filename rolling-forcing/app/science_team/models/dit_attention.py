# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Neuron Science Team, Amazon Annapurna Labs
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn

from kernels.kv_cache_copy import cache_copy, kv_cache_copy
from kernels.restore_layout import restore_layout
from kernels.rope import causal_rope_rotation, build_rope_grids
from kernels.self_attention import wan_flash_self_attn
from utils import _compile
from utils import parallel_state as ps

from models.dit_layers import (
    ATTN_SEQLEN_MULTIPLE,
    WanLayerNorm,
    WanRMSNorm,
    WanT2VCrossAttention,
)


def expand_e_shard(e, start_frame, end_frame, start_off, shard_len, frame_seqlen):
    B, _, I, C = e.shape
    F_sub = end_frame - start_frame
    e_sub = e[:, start_frame:end_frame]
    e_t = e_sub.transpose(1, 2)
    e_exp = e_t.unsqueeze(3).expand(B, I, F_sub, frame_seqlen, C).reshape(
        B, I, F_sub * frame_seqlen, C)
    return e_exp[:, :, start_off:start_off + shard_len]


def modulated_norm_scale_shard(norm_x, mod_slice, e_slice, ones):
    return norm_x * (ones + (mod_slice + e_slice))


def modulated_norm_shift_shard(y, mod_slice, e_slice):
    return y + (mod_slice + e_slice)


def modulated_residual_shard(x, y, mod_slice, e_slice):
    return x + y * (mod_slice + e_slice)


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
        assert qk_norm, "qk_norm must be True"
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.eps = eps
        self.frame_length = 1560
        self.max_attention_size = 21 * self.frame_length
        self.block_length = 3 * self.frame_length
        self.kv_cache_logical_size = 24 * self.frame_length
        self.layer_idx = layer_idx

        tp_degree = ps.get_world_size("attn-tp")
        sp_degree = ps.get_world_size("attn-sp")
        assert num_heads % tp_degree == 0, (
            f"num_heads ({num_heads}) must be divisible by tp_degree ({tp_degree})")
        self.sp_degree = sp_degree
        self.tp_degree = tp_degree
        self.world_size = sp_degree * tp_degree
        self.heads_per_shard = num_heads // tp_degree
        self.sp_rank = ps.get_rank("attn-sp")
        self.tp_rank = ps.get_rank("attn-tp")

        self.q = _compile(nn.Linear(dim, dim))
        self.k = _compile(nn.Linear(dim, dim))
        self.v = _compile(nn.Linear(dim, dim))
        if tp_degree > 1:
            self.o = _compile(nn.Linear(dim // tp_degree, dim))
        else:
            self.o = _compile(nn.Linear(dim, dim))
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

        sign_pattern = torch.ones(self.head_dim, dtype=torch.float32)
        sign_pattern[0::2] = -1.0
        self.register_buffer(
            'sign_pattern',
            sign_pattern.unsqueeze(0).expand(128, -1).contiguous(),
            persistent=False)

        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)

    @staticmethod
    def shard_state_dict(full_sd, dim, num_heads):
        tp_degree = ps.get_world_size("attn-tp")
        tp_rank = ps.get_rank("attn-tp")
        sd = {}
        head_dim = dim // num_heads
        heads_per_shard = num_heads // tp_degree
        shard_dim = heads_per_shard * head_dim

        for key, val in full_sd.items():
            if key == "o.weight":
                sd[key] = val[:, tp_rank * shard_dim:(tp_rank + 1) * shard_dim].clone()
            elif key == "o.bias":
                sd[key] = val.clone() if tp_rank == 0 else torch.zeros_like(val)
            else:
                sd[key] = val.clone()
        return sd

    def _local_qkv_norm(self, x):
        q = self.norm_q(self.q(x))[0]
        k = self.norm_k(self.k(x))[0]
        v = self.v(x)[0]
        return q, k, v

    def _gather_qkv(self, q_local, k_local, v_local, L):
        def _gather(t):
            if self.world_size == 1:
                return t
            out = torch.empty(L, self.dim, dtype=t.dtype, device=t.device)
            ps.all_gather_into_tensor(out, t, "world")
            return out

        return _gather(q_local), _gather(k_local), _gather(v_local)

    def _slice_heads(self, t):
        return self._slice_heads_2d(t).unsqueeze(0)

    def _slice_heads_2d(self, t):
        d = self.head_dim
        n = self.num_heads
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local
        L = t.shape[0]
        return t.view(L, n, d)[:, h_start:h_end]

    def _will_anchor_write(self, kv_cache, cache_start):
        cache_end = cache_start + self.block_length
        global_end_index = kv_cache["global_end_index"]
        local_end_index_current = kv_cache["local_end_index"]
        num_new_tokens = max(cache_end - global_end_index, 0)
        kv_cache_size = self.kv_cache_logical_size
        if num_new_tokens > 0 and num_new_tokens + local_end_index_current > kv_cache_size:
            num_evicted = num_new_tokens + local_end_index_current - kv_cache_size
        else:
            num_evicted = 0
        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        return local_end_index == self.block_length

    def _cache_copy_inplace(self, k_dst, k_src, v_dst=None, v_src=None):
        assert k_src.shape == k_dst.shape and k_src.numel() > 0
        assert v_dst is None or v_src.shape == v_dst.shape and v_src.numel() > 0
        """Device-dispatched cache copy: copy_ on CPU, NKI kernel on Neuron."""
        if v_dst is not None:
            kv_cache_copy(k_dst, k_src, v_dst, v_src)
        else:
            cache_copy(k_dst, k_src)

    def _nki_rope_apply(self, x, grid_sizes, freqs_cos, freqs_sin, start_frame,
                        rope_grid_cache=None, start_frame_int=None,
                        head_start=None, head_end=None):
        assert (head_start is None) == (head_end is None), (
            "head_start and head_end must be provided together")

        b, s, n, d = x.shape
        f, h, w = grid_sizes
        seq_len = f * h * w
        assert seq_len == s
        if head_start is None:
            head_start = 0
            head_end = n

        cache_key = None
        combined = None
        if rope_grid_cache is not None:
            assert start_frame_int is not None, (
                "start_frame_int must be provided with rope_grid_cache")
            cache_key = (grid_sizes, int(start_frame_int))
            combined = rope_grid_cache.get(cache_key)

        if combined is None:
            sf = start_frame.to(torch.int32).reshape(1, 1)
            combined = build_rope_grids(
                freqs_cos, freqs_sin, self.sign_pattern, sf,
                F=f, H=h, W=w, head_dim=d).view(seq_len, 2 * d)
            if cache_key is not None:
                rope_grid_cache[cache_key] = combined

        out = causal_rope_rotation(
            x[0, :seq_len], combined,
            head_start=head_start, head_end=head_end, head_dim=d)

        return out.unsqueeze(0)

    def _qkv_rope(self, x, grid_sizes, freqs_cos, freqs_sin, current_start,
                  rope_grid_cache=None, kv_cache=None):
        f, h, w = grid_sizes
        frame_seqlen = h * w
        L = f * h * w

        assert L % self.world_size == 0, (
            f"L ({L}) must be divisible by world_size ({self.world_size})")

        q_local, k_local, v_local = self._local_qkv_norm(x)
        q_full, k_full, v_full = self._gather_qkv(q_local, k_local, v_local, L)

        n = self.num_heads
        d = self.head_dim
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local
        q_full_4d = q_full.view(L, n, d).unsqueeze(0)
        k_full_4d = k_full.view(L, n, d).unsqueeze(0)
        v = self._slice_heads(v_full)

        start_frame_int = current_start // frame_seqlen
        start_frame_t = torch.tensor(start_frame_int, device=x.device)

        roped_q_full = self._nki_rope_apply(
            q_full_4d, grid_sizes, freqs_cos, freqs_sin,
            start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=start_frame_int,
            head_start=h_start, head_end=h_end)
        sp_shard_len = L // self.sp_degree
        sp_start = self.sp_rank * sp_shard_len
        roped_query = roped_q_full[:, sp_start:sp_start + sp_shard_len]

        roped_key = self._nki_rope_apply(
            k_full_4d, grid_sizes, freqs_cos, freqs_sin,
            start_frame=start_frame_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=start_frame_int,
            head_start=h_start, head_end=h_end)

        if kv_cache is None or self._will_anchor_write(kv_cache, current_start):
            k = self._slice_heads(k_full)
        else:
            k = None

        return roped_query, k, v, roped_key

    def _cache_write(self, k, v, roped_key, kv_cache, cache_start, shared_buffers):
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
            self._cache_copy_inplace(
                buffer_k[0, :evict_rolled], kv_cache["k"][0, src_start:src_start + evict_rolled],
                buffer_v[0, :evict_rolled], kv_cache["v"][0, src_start:src_start + evict_rolled])
            self._cache_copy_inplace(
                kv_cache["k"][0, sink_tokens:sink_tokens + evict_rolled], buffer_k[0, :evict_rolled],
                kv_cache["v"][0, sink_tokens:sink_tokens + evict_rolled], buffer_v[0, :evict_rolled])

        local_end_index = local_end_index_current + num_new_tokens - num_evicted
        local_start_index = local_end_index - self.block_length

        if local_start_index == 0:
            self._cache_copy_inplace(
                kv_cache["k"][0, :self.block_length], k[0, :self.block_length],
                kv_cache["v"][0, :self.block_length], v[0, :self.block_length])
        else:
            self._cache_copy_inplace(
                kv_cache["k"][0, local_start_index:local_end_index], roped_key[0, :self.block_length],
                kv_cache["v"][0, local_start_index:local_end_index], v[0, :self.block_length])

        if num_new_tokens > 0:
            kv_cache["global_end_index"] = cache_end
            kv_cache["local_end_index"] = local_end_index

        return local_end_index, local_start_index

    def _assemble_kv(self, roped_key, v, kv_cache, grid_sizes, freqs_cos, freqs_sin,
                     shared_buffers, updating_cache, valid_tokens,
                     current_start_frame, local_end_index, local_start_index,
                     rope_grid_cache=None):
        _, h, w = grid_sizes
        grid_sizes_one_block = (3, h, w)
        buffer_k, buffer_v = shared_buffers
        device = roped_key.device

        if updating_cache:
            cache_len = min(local_end_index, self.max_attention_size)
            cache_start_pos = max(0, local_end_index - self.max_attention_size)

            self._cache_copy_inplace(
                buffer_k[0, :cache_len],
                kv_cache["k"][0, cache_start_pos:cache_start_pos + cache_len],
                buffer_v[0, :cache_len],
                kv_cache["v"][0, cache_start_pos:cache_start_pos + cache_len])

            if cache_start_pos == 0:
                anchor_roped = self._nki_rope_apply(
                    kv_cache["k"][0, :self.block_length].unsqueeze(0),
                    grid_sizes_one_block, freqs_cos, freqs_sin,
                    start_frame=torch.tensor(0, device=device),
                    rope_grid_cache=rope_grid_cache, start_frame_int=0)
                self._cache_copy_inplace(
                    buffer_k[0, :self.block_length], anchor_roped[0])

            return cache_len

        offset = 0
        if local_start_index > 0:
            wc_max = self.max_attention_size - valid_tokens - self.block_length
            wc_end = local_start_index
            wc_start = max(self.block_length, wc_end - wc_max)
            wc_len = wc_end - wc_start

            wc_frame_length = wc_len // self.frame_length
            rope_start_frame = current_start_frame - wc_frame_length - 3
            anchor_roped = self._nki_rope_apply(
                kv_cache["k"][0, :self.block_length].unsqueeze(0),
                grid_sizes_one_block, freqs_cos, freqs_sin,
                start_frame=torch.tensor(rope_start_frame, device=device),
                rope_grid_cache=rope_grid_cache,
                start_frame_int=rope_start_frame)
            self._cache_copy_inplace(
                buffer_k[0, :self.block_length], anchor_roped[0],
                buffer_v[0, :self.block_length], kv_cache["v"][0, :self.block_length])
            offset = self.block_length

            if wc_len > 0:
                self._cache_copy_inplace(
                    buffer_k[0, offset:offset + wc_len], kv_cache["k"][0, wc_start:wc_start + wc_len],
                    buffer_v[0, offset:offset + wc_len], kv_cache["v"][0, wc_start:wc_start + wc_len])
            offset += wc_len

        self._cache_copy_inplace(
            buffer_k[0, offset:offset + valid_tokens], roped_key[0, :valid_tokens],
            buffer_v[0, offset:offset + valid_tokens], v[0, :valid_tokens])
        return offset + valid_tokens

    def _attend(self, roped_query, shared_buffers, k_len_int):
        buffer_k, buffer_v = shared_buffers
        q_kern = roped_query[0].permute(1, 2, 0).contiguous()
        k_kern = buffer_k[0].permute(1, 2, 0).contiguous()
        v_kern = buffer_v[0].permute(1, 0, 2).contiguous()

        assert k_kern.shape[2] % ATTN_SEQLEN_MULTIPLE == 0
        assert v_kern.shape[1] % ATTN_SEQLEN_MULTIPLE == 0
        out = wan_flash_self_attn(
            q_kern, k_kern, v_kern,
            softmax_scale=self.softmax_scale,
            actual_seqlen_k=k_len_int,
            use_dynamic_loop=True,
        )
        return out.unsqueeze(0).flatten(2)

    def _output_proj(self, out):
        out = self.o(out)
        if self.tp_degree > 1:
            seq_len = out.shape[1]
            out_flat = out.reshape(-1, self.dim)
            rs_out = torch.empty(
                seq_len // self.tp_degree, self.dim,
                dtype=out.dtype, device=out.device)
            ps.reduce_scatter_tensor(rs_out, out_flat, "attn-tp")
            out = rs_out.unsqueeze(0)
        return out

    def forward_merged(
        self,
        x,
        grid_sizes,
        freqs_cos, freqs_sin,
        kv_cache,
        cache_update_start, current_start,
        cu_shared_buffers, dn_shared_buffers,
        num_valid_frames_dn,
        nfpb_cu,
        rope_grid_cache=None,
    ):
        assert x.shape[0] == 1

        f_full, h, w = grid_sizes
        frame_seqlen = h * w
        grid_cu = (nfpb_cu, h, w)
        grid_dn = (f_full - nfpb_cu, h, w)
        L_cu = nfpb_cu * frame_seqlen
        L_dn = (f_full - nfpb_cu) * frame_seqlen
        L_full = L_cu + L_dn
        current_start_frame_dn = current_start // frame_seqlen

        sp = self.sp_degree
        tp = self.tp_degree
        N = self.world_size
        assert L_cu % N == 0, f"L_cu ({L_cu}) must be divisible by world_size ({N})"
        assert L_dn % N == 0, f"L_dn ({L_dn}) must be divisible by world_size ({N})"
        L_cu_sp = L_cu // sp
        L_dn_sp = L_dn // sp
        L_cu_N = L_cu // N
        L_dn_N = L_dn // N
        L_full_N = L_full // N

        q_local, k_local, v_local = self._local_qkv_norm(x)
        q_full, k_full, v_full = self._gather_qkv(
            q_local, k_local, v_local, L_full)

        n = self.num_heads
        d = self.head_dim
        n_local = self.heads_per_shard
        h_start = self.tp_rank * n_local
        h_end = h_start + n_local

        v_full_h = self._slice_heads_2d(v_full).contiguous()

        q_full_4d = q_full.view(L_full, n, d)
        k_full_4d = k_full.view(L_full, n, d)
        q_cu = q_full_4d[:L_cu].unsqueeze(0)
        q_dn = q_full_4d[L_cu:].unsqueeze(0)
        k_cu_full = k_full_4d[:L_cu].unsqueeze(0)
        k_dn_full = k_full_4d[L_cu:].unsqueeze(0)
        v_cu = v_full_h[:L_cu].unsqueeze(0)
        v_dn = v_full_h[L_cu:].unsqueeze(0)

        cu_sf_int = cache_update_start // frame_seqlen
        dn_sf_int = current_start // frame_seqlen
        cu_sf_t = torch.tensor(cu_sf_int, device=q_cu.device)
        dn_sf_t = torch.tensor(dn_sf_int, device=q_dn.device)

        rq_cu_full = self._nki_rope_apply(
            q_cu, grid_cu, freqs_cos, freqs_sin,
            start_frame=cu_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=cu_sf_int,
            head_start=h_start, head_end=h_end)
        rk_cu = self._nki_rope_apply(
            k_cu_full, grid_cu, freqs_cos, freqs_sin,
            start_frame=cu_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=cu_sf_int,
            head_start=h_start, head_end=h_end)
        rq_dn_full = self._nki_rope_apply(
            q_dn, grid_dn, freqs_cos, freqs_sin,
            start_frame=dn_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=dn_sf_int,
            head_start=h_start, head_end=h_end)
        rk_dn = self._nki_rope_apply(
            k_dn_full, grid_dn, freqs_cos, freqs_sin,
            start_frame=dn_sf_t, rope_grid_cache=rope_grid_cache,
            start_frame_int=dn_sf_int,
            head_start=h_start, head_end=h_end)

        if self._will_anchor_write(kv_cache, cache_update_start):
            k_cu = self._slice_heads_2d(k_full[:L_cu]).contiguous().unsqueeze(0)
        else:
            k_cu = None
        le_cu, ls_cu = self._cache_write(
            k_cu, v_cu, rk_cu, kv_cache, cache_update_start, cu_shared_buffers)

        if self._will_anchor_write(kv_cache, current_start):
            k_dn = self._slice_heads_2d(k_full[L_cu:]).contiguous().unsqueeze(0)
        else:
            k_dn = None
        le_dn, ls_dn = self._cache_write(
            k_dn, v_dn, rk_dn, kv_cache, current_start, dn_shared_buffers)

        klen_cu = self._assemble_kv(
            rk_cu, v_cu, kv_cache, grid_cu, freqs_cos, freqs_sin,
            cu_shared_buffers, True, nfpb_cu * frame_seqlen,
            cache_update_start // frame_seqlen, le_cu, ls_cu,
            rope_grid_cache=rope_grid_cache)
        klen_dn = self._assemble_kv(
            rk_dn, v_dn, kv_cache, grid_dn, freqs_cos, freqs_sin,
            dn_shared_buffers, False, num_valid_frames_dn * frame_seqlen,
            current_start_frame_dn, le_dn, ls_dn,
            rope_grid_cache=rope_grid_cache)

        q_cu_sp = rq_cu_full[:, self.sp_rank * L_cu_sp:(self.sp_rank + 1) * L_cu_sp]
        q_dn_sp = rq_dn_full[:, self.sp_rank * L_dn_sp:(self.sp_rank + 1) * L_dn_sp]
        y_cu = self._attend(q_cu_sp, cu_shared_buffers, klen_cu)
        y_dn = self._attend(q_dn_sp, dn_shared_buffers, klen_dn)

        y = self.o(torch.cat([y_cu, y_dn], dim=1))

        if tp > 1:
            y = y.reshape(L_full // sp, self.dim)
            y_cu_part = y[:L_cu_sp].reshape(tp, L_cu_N, self.dim)
            y_dn_part = y[L_cu_sp:].reshape(tp, L_dn_N, self.dim)
            rearranged = torch.cat([y_cu_part, y_dn_part], dim=1).reshape(-1, self.dim)
            rs_out = torch.empty(L_full_N, self.dim, dtype=y.dtype, device=y.device)
            ps.reduce_scatter_tensor(rs_out, rearranged, "attn-tp")
            cu_dn_sep = rs_out
        else:
            cu_dn_sep = y.reshape(L_full_N, self.dim)

        if N == 1:
            return cu_dn_sep.unsqueeze(0)

        gathered = torch.empty(N * L_full_N, self.dim, dtype=cu_dn_sep.dtype, device=cu_dn_sep.device)
        ps.all_gather_into_tensor(gathered, cu_dn_sep, "world")
        full = restore_layout(gathered, N=N)
        rank_world = ps.get_rank("world")
        out = full[rank_world * L_full_N:(rank_world + 1) * L_full_N]
        return out.unsqueeze(0)

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
        rope_grid_cache=None,
    ):
        assert kv_cache is not None
        assert x.shape[0] == 1, f"Batch size must be 1, got {x.shape[0]}"
        if cache_start is None:
            cache_start = current_start

        f, h, w = grid_sizes
        frame_seqlen = h * w
        current_start_frame = current_start // frame_seqlen

        roped_query, k, v, roped_key = self._qkv_rope(
            x, grid_sizes, freqs_cos, freqs_sin, current_start,
            rope_grid_cache=rope_grid_cache,
            kv_cache=kv_cache,
        )

        if num_valid_frames is not None:
            valid_tokens = num_valid_frames * frame_seqlen
        else:
            valid_tokens = f * h * w

        local_end_index, local_start_index = self._cache_write(
            k, v, roped_key, kv_cache, cache_start, shared_buffers)

        k_len_int = self._assemble_kv(
            roped_key, v, kv_cache, grid_sizes, freqs_cos, freqs_sin,
            shared_buffers, updating_cache, valid_tokens,
            current_start_frame, local_end_index, local_start_index,
            rope_grid_cache=rope_grid_cache)

        x = self._attend(roped_query, shared_buffers, k_len_int)

        x = self._output_proj(x)
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
        assert cross_attn_type == 't2v_cross_attn'
        assert cross_attn_norm
        self.layer_idx = layer_idx
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.norm2 = WanLayerNorm(dim, eps)

        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_idx)
        self.cross_attn = WanT2VCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, layer_idx=layer_idx)

        self.ffn = _compile(nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim)))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.world_size = ps.get_world_size("world")
        self.rank = ps.get_rank("world")

        self._modulated_norm_scale_shard = _compile(modulated_norm_scale_shard)
        self._modulated_norm_shift_shard = _compile(modulated_norm_shift_shard)
        self._modulated_residual_shard = _compile(modulated_residual_shard)

    @staticmethod
    def shard_state_dict(full_sd, dim, num_heads):
        sd = {}
        self_attn_full = {
            key[len("self_attn."):]: val
            for key, val in full_sd.items()
            if key.startswith("self_attn.")
        }
        self_attn_sharded = CausalWanSelfAttention.shard_state_dict(
            self_attn_full, dim, num_heads)
        for key, val in self_attn_sharded.items():
            sd[f"self_attn.{key}"] = val
        for key, val in full_sd.items():
            if not key.startswith("self_attn."):
                sd[key] = val.clone()
        return sd

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
        mode="denoise",
        cache_update_start=None,
        cu_shared_buffers=None,
        nfpb_cu=None,
        rope_grid_cache=None,
    ):
        e0s, e1s, e2s, e3s, e4s, e5s = (
            e[:, 0], e[:, 1], e[:, 2], e[:, 3], e[:, 4], e[:, 5])
        m0, m1, m2, m3, m4, m5 = (
            self.modulation[:, 0], self.modulation[:, 1], self.modulation[:, 2],
            self.modulation[:, 3], self.modulation[:, 4], self.modulation[:, 5])
        ones_shard = torch.ones_like(e0s)

        attn_in = self._modulated_norm_shift_shard(
            self._modulated_norm_scale_shard(self.norm1(x), m1, e1s, ones_shard),
            m0, e0s,
        )

        if mode == "merged":
            assert cache_update_start is not None and nfpb_cu is not None
            y = self.self_attn.forward_merged(
                attn_in,
                grid_sizes,
                freqs_cos, freqs_sin,
                kv_cache,
                cache_update_start, current_start,
                cu_shared_buffers, shared_buffers,
                num_valid_frames_dn=num_valid_frames,
                nfpb_cu=nfpb_cu,
                rope_grid_cache=rope_grid_cache,
            )
        else:
            y = self.self_attn(
                attn_in,
                grid_sizes,
                freqs_cos,
                freqs_sin,
                kv_cache,
                current_start,
                cache_start,
                updating_cache=updating_cache,
                num_valid_frames=num_valid_frames,
                shared_buffers=shared_buffers,
                rope_grid_cache=rope_grid_cache,
            )
        x = self._modulated_residual_shard(x, y, m2, e2s)

        x = x + self.cross_attn(
            self.norm3(x), context, context_lens,
            crossattn_cache=crossattn_cache)

        y = self.ffn(
            self._modulated_norm_shift_shard(
                self._modulated_norm_scale_shard(self.norm2(x), m4, e4s, ones_shard),
                m3, e3s,
            )
        )
        x = self._modulated_residual_shard(x, y, m5, e5s)

        return x
