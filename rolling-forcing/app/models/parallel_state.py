"""Parallel state registry for TP and SP process groups.

Manages named process groups for tensor parallelism (attn-tp) and
sequence parallelism (attn-sp). Used by the SP-aware attention module.

Usage:
    # At init (in inference_neuron_tp.py):
    init_parallel_groups(sp_degree=2, tp_degree=4)  # 8 ranks total

    # In model code:
    tp_degree = parallel_state.get_world_size("attn-tp")
    sp_rank = parallel_state.get_rank("attn-sp")
    parallel_state.all_gather_into_tensor(output, input, "attn-sp")
"""
import torch.distributed as dist


_GROUPS = {}


def register_group(name, group):
    assert name not in _GROUPS, f"group {name!r} already registered"
    _GROUPS[name] = group


def destroy_group(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    del _GROUPS[name]


def is_registered(name):
    return name in _GROUPS


def _get(name):
    assert name in _GROUPS, f"group {name!r} is not registered"
    return _GROUPS[name]


def get_group(name):
    return _get(name)


def get_world_size(name):
    return dist.get_world_size(_get(name))


def get_rank(name):
    return dist.get_rank(_get(name))


def all_gather_into_tensor(output, input, group_name):
    dist.all_gather_into_tensor(output, input, group=_get(group_name))


def reduce_scatter_tensor(output, input, group_name):
    dist.reduce_scatter_tensor(output, input, group=_get(group_name))


def all_reduce(tensor, group_name, op=dist.ReduceOp.SUM):
    dist.all_reduce(tensor, op=op, group=_get(group_name))


def init_parallel_groups(sp_degree, tp_degree):
    """Initialize TP and SP process groups.

    With 8 ranks, tp_degree=4, sp_degree=2:
      - attn-tp groups: [0,1,2,3] and [4,5,6,7]
      - attn-sp groups: [0,4], [1,5], [2,6], [3,7]
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert sp_degree * tp_degree == world_size, (
        f"sp_degree * tp_degree ({sp_degree * tp_degree}) must equal "
        f"world_size ({world_size})")

    sp_rank = rank // tp_degree
    tp_rank = rank % tp_degree

    register_group("world", dist.group.WORLD)

    tp_group = None
    for sp_i in range(sp_degree):
        ranks = list(range(sp_i * tp_degree, (sp_i + 1) * tp_degree))
        grp = dist.new_group(ranks)
        if sp_i == sp_rank:
            tp_group = grp
    register_group("attn-tp", tp_group)

    sp_group = None
    for tp_i in range(tp_degree):
        ranks = list(range(tp_i, world_size, tp_degree))
        grp = dist.new_group(ranks)
        if tp_i == tp_rank:
            sp_group = grp
    register_group("attn-sp", sp_group)


def destroy_parallel_groups():
    destroy_group("attn-tp")
    destroy_group("attn-sp")
    destroy_group("world")
