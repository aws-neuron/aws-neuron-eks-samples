import os
import time

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

# torch_neuronx.jit doesn't exist in private-torch-neuronx; use identity function
def jit(fn):
    """No-op wrapper since torch_neuronx.jit is not available."""
    return fn

from models.layers import (
    GELU,
    SiLU,
    WanPatchEmbed,
    CausalHead,
    CausalWanAttentionBlock,
    rope_params,
    sinusoidal_embedding_1d,
    unpatchify,
)


def _init_rope_freqs(dim, num_heads):
    """Precompute 3D RoPE cos/sin frequencies for (frame, height, width).

    Returns (cos, sin) each [1024, d//2] in float32, where d = dim // num_heads.
    Neuron rope_params returns (cos, sin) instead of complex (no float64/complex).
    """
    assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
    d = dim // num_heads
    cos_0, sin_0 = rope_params(1024, d - 4 * (d // 6))
    cos_1, sin_1 = rope_params(1024, 2 * (d // 6))
    cos_2, sin_2 = rope_params(1024, 2 * (d // 6))
    return torch.cat([cos_0, cos_1, cos_2], dim=1), torch.cat([sin_0, sin_1, sin_2], dim=1)


class CausalWanModel(ModelMixin, ConfigMixin):
    """Neuron-compatible CausalWanModel for causal video generation.

    using Neuron-compatible components from models.layers.
    """

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['CausalWanAttentionBlock']

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

        # Embeddings — WanPatchEmbed replaces nn.Conv3d (unsupported on Neuron),
        # jit-wrapped Sequentials use Neuron-compatible GELU/SiLU activations
        self.patch_embedding = WanPatchEmbed(in_dim, dim, patch_size)
        self.text_embedding = jit(nn.Sequential(
            nn.Linear(text_dim, dim), GELU(), nn.Linear(dim, dim)))
        self.time_embedding = jit(nn.Sequential(
            nn.Linear(freq_dim, dim), SiLU(), nn.Linear(dim, dim)))
        self.time_projection = jit(nn.Sequential(
            SiLU(), nn.Linear(dim, dim * 6)))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(
                't2v_cross_attn', dim, ffn_dim, num_heads,
                local_attn_size, sink_size, qk_norm, cross_attn_norm,
                eps, layer_idx, frame_length)
            for layer_idx in range(num_layers)
        ])

        # Output head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # jit-wrapped standalone functions
        self._sinusoidal_embedding_1d = jit(sinusoidal_embedding_1d)
        self._unpatchify = jit(unpatchify)

        # RoPE frequencies — not buffers (to() would change dtype)
        self.freqs_cos, self.freqs_sin = _init_rope_freqs(dim, num_heads)

    def _update_frame_length(self, new_frame_length: int, num_frame_per_block: int = 3):
        """Update frame_length in all attention blocks after loading.
        
        Required because from_pretrained() ignores runtime kwargs and loads
        frame_length from the saved config (default 1560 for full res).
        
        Args:
            new_frame_length: tokens per frame (H * W after patch embed)
            num_frame_per_block: frames per block from config (default 3)
        """
        block_length = num_frame_per_block * new_frame_length
        for block in self.blocks:
            attn = block.self_attn
            attn.frame_length = new_frame_length
            attn.block_length = block_length
            attn.max_attention_size = 21 * new_frame_length
            attn.kv_cache_logical_size = 24 * new_frame_length
        print(f"[CausalWanModel] Updated frame_length={new_frame_length}, block_length={block_length} in {len(self.blocks)} blocks")

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
    ):
        assert self.model_type == 't2v'
        assert x.shape[0] == 1
        assert not torch.is_grad_enabled()
        _profile = os.environ.get("PROFILE_PIPELINE", "0") == "1" and int(os.environ.get("NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS", "0")) == 0

        device = self.patch_embedding.weight.device
        if self.freqs_cos.device != device:
            self.freqs_cos = self.freqs_cos.to(device)
            self.freqs_sin = self.freqs_sin.to(device)

        # Patch embedding
        if _profile:
            _t = time.perf_counter()
        x = self.patch_embedding(x)
        grid_sizes = tuple(int(d) for d in x.shape[2:])
        x = x.flatten(2).transpose(1, 2)
        if _profile:
            _t_patch = (time.perf_counter() - _t) * 1000

        # Time embedding
        if _profile:
            _t = time.perf_counter()
        e = self.time_embedding(
            self._sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        if _profile:
            _t_time = (time.perf_counter() - _t) * 1000

        # Context embedding
        if _profile:
            _t = time.perf_counter()
        context_lens = None
        assert context.size(1) == self.text_len
        context = self.text_embedding(context)
        if _profile:
            _t_text = (time.perf_counter() - _t) * 1000

        # Transformer blocks
        if _profile:
            _t = time.perf_counter()
        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            freqs_cos=self.freqs_cos,
            freqs_sin=self.freqs_sin,
            context=context,
            context_lens=context_lens,
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
                    "cache_start": cache_start,
                }
            )
            x = block(x, **kwargs)
        if _profile:
            _t_blocks = (time.perf_counter() - _t) * 1000

        # Head + unpatchify
        if _profile:
            _t = time.perf_counter()
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        if _profile:
            _t_head = (time.perf_counter() - _t) * 1000
            _t = time.perf_counter()
        x = x.flatten(1, 2)
        result = self._unpatchify(x, self.out_dim, self.patch_size, grid_sizes).unsqueeze(0)
        if _profile:
            _t_unpatchify = (time.perf_counter() - _t) * 1000
            print(f"      [model] patch_embed={_t_patch:.1f}ms  time_embed={_t_time:.1f}ms  text_embed={_t_text:.1f}ms")
            print(f"      [model] blocks={_t_blocks:.1f}ms  head={_t_head:.1f}ms  unpatchify={_t_unpatchify:.1f}ms")
        return result

    def forward(self, *args, **kwargs):
        assert kwargs.get('kv_cache', None) is not None
        return self._forward_inference(*args, **kwargs)
