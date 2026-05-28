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
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn

from utils import parallel_state as ps
from utils import _compile
from utils.tokenizers import HuggingfaceTokenizer


SEQ_LEN = 512
_TP_GROUP = "t5-tp"




def init_t5_parallel_group():
    ps.register_group(_TP_GROUP, dist.group.WORLD)


def destroy_t5_parallel_group():
    ps.destroy_group(_TP_GROUP)


def _tp_degree():
    return ps.get_world_size(_TP_GROUP) if ps.is_registered(_TP_GROUP) else 1


def _tp_rank():
    return ps.get_rank(_TP_GROUP) if ps.is_registered(_TP_GROUP) else 0


def _shard_attention(full_sd):
    tp = _tp_degree()
    rank = _tp_rank()
    shard = full_sd["q.weight"].shape[0] // tp
    sd = {}
    for k, v in full_sd.items():
        if k in ("q.weight", "k.weight", "v.weight"):
            sd[k] = v[rank * shard:(rank + 1) * shard, :].clone()
        elif k == "o.weight":
            sd[k] = v[:, rank * shard:(rank + 1) * shard].clone()
        else:
            sd[k] = v.clone()
    return sd


def _shard_ffn(full_sd):
    tp = _tp_degree()
    rank = _tp_rank()
    shard = full_sd["fc1.weight"].shape[0] // tp
    sd = {}
    for k, v in full_sd.items():
        if k in ("gate.0.weight", "fc1.weight"):
            sd[k] = v[rank * shard:(rank + 1) * shard, :].clone()
        elif k == "fc2.weight":
            sd[k] = v[:, rank * shard:(rank + 1) * shard].clone()
        else:
            sd[k] = v.clone()
    return sd


def _shard_block(full_sd):
    attn_sd = {k[len("attn."):]: v for k, v in full_sd.items() if k.startswith("attn.")}
    ffn_sd = {k[len("ffn."):]: v for k, v in full_sd.items() if k.startswith("ffn.")}
    sa = _shard_attention(attn_sd)
    sf = _shard_ffn(ffn_sd)
    sd = {}
    for k, v in full_sd.items():
        if k.startswith("attn."):
            sd[k] = sa[k[len("attn."):]]
        elif k.startswith("ffn."):
            sd[k] = sf[k[len("ffn."):]]
        else:
            sd[k] = v.clone()
    return sd


def shard_encoder_state_dict(full_sd):
    if _tp_degree() == 1:
        return dict(full_sd)
    blocks_by_idx, passthrough = {}, {}
    for k, v in full_sd.items():
        if k.startswith("blocks."):
            idx_str, subkey = k[len("blocks."):].split(".", 1)
            blocks_by_idx.setdefault(int(idx_str), {})[subkey] = v
        else:
            passthrough[k] = v
    sd = {k: v.clone() for k, v in passthrough.items()}
    for idx, block_sd in blocks_by_idx.items():
        for subkey, v in _shard_block(block_sd).items():
            sd[f"blocks.{idx}.{subkey}"] = v
    return sd




def _relative_position_bucket(rel_pos, num_buckets, max_dist=128):
    num_buckets = num_buckets // 2
    rel_buckets = (rel_pos > 0).long() * num_buckets
    rel_pos = torch.abs(rel_pos)

    max_exact = num_buckets // 2
    rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                 math.log(max_dist / max_exact) *
                                 (num_buckets - max_exact)).long()
    rel_pos_large = torch.min(
        rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
    rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
    return rel_buckets


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@_compile
class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
        rms = sq_sum / x_f.shape[-1]
        x_normed = x_f * torch.rsqrt(rms + self.eps)
        return (self.weight * x_normed).to(x.dtype)


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn):
        super().__init__()
        tp = _tp_degree()
        assert dim_ffn % tp == 0
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.tp_degree = tp
        shard_ffn = dim_ffn // tp

        self.gate = nn.Sequential(nn.Linear(dim, shard_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, shard_ffn, bias=False)
        self.fc2 = nn.Linear(shard_ffn, dim, bias=False)

    def forward(self, x):
        x = self.fc2(self.fc1(x) * self.gate(x))
        if self.tp_degree > 1:
            ps.all_reduce(x, _TP_GROUP)
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, rel_pos_buckets):
        super().__init__()
        tp = _tp_degree()
        assert num_heads % tp == 0
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.tp_degree = tp
        self.heads_per_shard = num_heads // tp
        self.embedding = nn.Embedding(num_buckets, num_heads)
        self.rel_pos_buckets = rel_pos_buckets
        self.register_buffer("_cached_bias", None, persistent=False)

    def precompute(self):
        with torch.no_grad():
            bias = self.embedding(self.rel_pos_buckets).permute(2, 0, 1).unsqueeze(0).contiguous()
            if self.tp_degree > 1:
                rank = _tp_rank()
                h = self.heads_per_shard
                bias = bias[:, rank * h:(rank + 1) * h, :, :].contiguous()
            self._cached_bias = bias
        del self.embedding
        del self.rel_pos_buckets

    def forward(self):
        return self._cached_bias


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads):
        super().__init__()
        assert dim_attn % num_heads == 0
        tp = _tp_degree()
        assert dim_attn % tp == 0 and num_heads % tp == 0

        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.tp_degree = tp
        self.heads_per_shard = num_heads // tp
        shard_dim = dim_attn // tp

        self.q = nn.Linear(dim, shard_dim, bias=False)
        self.k = nn.Linear(dim, shard_dim, bias=False)
        self.v = nn.Linear(dim, shard_dim, bias=False)
        self.o = nn.Linear(shard_dim, dim, bias=False)

    def forward(self, x, mask, pos_bias):
        b = x.shape[0]
        n = self.heads_per_shard
        c = self.head_dim

        q = self.q(x).reshape(b, -1, n, c).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, -1, n, c).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, -1, n, c).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) + pos_bias
        attn = attn + (1.0 - mask.reshape(b, 1, 1, -1).float()) * (-1e9)

        attn_f = attn.float()
        attn_f = torch.exp(attn_f - torch.amax(attn_f, dim=-1, keepdim=True))
        attn = (attn_f / torch.sum(attn_f, dim=-1, keepdim=True)).to(x.dtype)

        x = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, -1, n * c)
        x = self.o(x)

        if self.tp_degree > 1:
            ps.all_reduce(x, _TP_GROUP)

        return x


@_compile
class T5SelfAttention(nn.Module):

    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets,
                 rel_pos_buckets):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads,
                                                  rel_pos_buckets)

    def forward(self, x, mask):
        e = self.pos_embedding()
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class T5Encoder(nn.Module):

    def __init__(self, vocab_size, dim, dim_attn, dim_ffn, num_heads,
                 num_layers, num_buckets):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.token_embedding = nn.Embedding(vocab_size, dim)

        rel_pos = torch.arange(SEQ_LEN).unsqueeze(0) - \
            torch.arange(SEQ_LEN).unsqueeze(1)
        rel_pos_buckets = _relative_position_bucket(rel_pos, num_buckets)

        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            rel_pos_buckets)
            for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def precompute(self):
        for block in self.blocks:
            block.pos_embedding.precompute()

    def forward(self, ids, mask):
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return x


def umt5_xxl(*, dtype=torch.float32, device='cpu', **overrides):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        num_layers=24,
        num_buckets=32,
    )
    cfg.update(**overrides)

    with torch.device(device):
        model = T5Encoder(**cfg)
    return model.to(dtype=dtype, device=device)


class WanTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.text_encoder = umt5_xxl(
            dtype=torch.float32,
            device=torch.device('cpu'),
        ).eval().requires_grad_(False)

        full_sd = torch.load(
            "wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            map_location='cpu', weights_only=False,
        )
        if _tp_degree() > 1:
            full_sd = shard_encoder_state_dict(full_sd)
        self.text_encoder.load_state_dict(full_sd, strict=True)
        self.text_encoder.precompute()

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/",
            seq_len=512)

    def forward(self, text_prompts: List[str]) -> dict:
        assert len(text_prompts) == 1
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        device = self.text_encoder.token_embedding.weight.device
        ids = ids.to(device)
        mask = mask.to(device)
        context = self.text_encoder(ids, mask)
        return {"prompt_embeds": context * mask.unsqueeze(-1).to(context.dtype)}


def build_text_encoder(device="neuron", dtype=torch.bfloat16):
    return (WanTextEncoder().eval().requires_grad_(False)
            .to(device=device, dtype=dtype))


def encode_one_prompt(text_encoder, prompt: str) -> torch.Tensor:
    return text_encoder([prompt])["prompt_embeds"]
