import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 128


def _compile(mod_or_fn):
    return torch.compile(mod_or_fn, backend="neuron", dynamic=False, fullgraph=True)


def w_shard(tensor, rank, world):
    W = tensor.shape[-1]
    assert W % world == 0, f"W={W} not divisible by world={world}"
    s = W // world
    return tensor[..., rank * s:(rank + 1) * s].contiguous()
