# Optimized version: WanT2VCrossAttention for batch_size=1.
# Imports unchanged classes from original model.py.
from wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    rope_params,
    rope_apply,
    MLPProj,
    sinusoidal_embedding_1d
)
from wan.modules.attention_opt import flash_attn_varlen_b1


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens: unused (kept for API compat), all K tokens are valid
            crossattn_cache (dict, *optional*): Cached key/value tensors for context.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # All K tokens valid (context_lens is always None), no masking needed
        x = flash_attn_varlen_b1(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
}
