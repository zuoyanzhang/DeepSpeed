# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from TorchTitan graph_trainer/simple_fsdp.py for the DeepCompile
# benchmark harness. This is the compiler-friendly SimpleFSDP core: parameters
# are stored as DTensor shards, and parameter access performs a traceable
# redistribute to replicated local tensors.

from __future__ import annotations

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement

_active_parametrization = True


@contextmanager
def disable_active_parametrization() -> Generator[None, None, None]:
    global _active_parametrization
    old = _active_parametrization
    try:
        _active_parametrization = False
        yield
    finally:
        _active_parametrization = old


@dataclass(frozen=True)
class SimpleFSDPMixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


def _distribute_dtensor(
    tensor: DTensor,
    device_mesh: DeviceMesh,
    dp_placements: Sequence[Placement],
) -> DTensor:
    inner_spec = tensor._spec
    outer_mesh, inner_mesh = device_mesh, inner_spec.mesh
    spanned_mesh = DeviceMesh._concatenate([outer_mesh, inner_mesh])

    if len(dp_placements) == 1:
        assert dp_placements[0].is_replicate() or dp_placements[0].is_shard()
        if dp_placements[0].is_shard():
            assert len(inner_spec.placements) in (1, 2)
            shard_dim = dp_placements[0].dim
            split_factor = inner_spec.num_shards_map[shard_dim]
            tensor_placement = (
                _StridedShard(shard_dim, split_factor=split_factor)
                if split_factor > 1
                else dp_placements[0],
            ) + inner_spec.placements
        else:
            assert len(inner_spec.placements) == 1
            tensor_placement = (dp_placements[0], inner_spec.placements[0])
    elif len(dp_placements) == 2:
        assert dp_placements[0].is_replicate() and dp_placements[1].is_shard()
        assert len(inner_spec.placements) in (1, 2)
        shard_dim = dp_placements[1].dim
        split_factor = inner_spec.num_shards_map[shard_dim]
        tensor_placement = (
            dp_placements[0],
            _StridedShard(shard_dim, split_factor=split_factor)
            if split_factor > 1
            else dp_placements[1],
        ) + inner_spec.placements
    else:
        raise ValueError(f"Unsupported SimpleFSDP placement {dp_placements}")

    current_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=(Replicate(),) * len(dp_placements),
        tensor_meta=inner_spec.tensor_meta,
    )
    target_spec = DTensorSpec(
        mesh=outer_mesh,
        placements=tuple(dp_placements),
        tensor_meta=inner_spec.tensor_meta,
    )
    result_tensor = redistribute_local_tensor(
        tensor._local_tensor,
        current_spec=current_spec,
        target_spec=target_spec,
    )
    return DTensor(
        result_tensor.requires_grad_(tensor.requires_grad),
        DTensorSpec(
            mesh=spanned_mesh,
            placements=tensor_placement,
            tensor_meta=inner_spec.tensor_meta,
        ),
        requires_grad=tensor.requires_grad,
    )


_wrap_class_cache: dict[tuple[type, frozenset[str]], type] = {}


def _register_parametrization(
    module: nn.Module, param_names: list[str], parametrization: nn.Module
) -> None:
    if not param_names:
        return
    param_name_to_property = {
        param_name: property(
            lambda self, pn=param_name: parametrization(self._parameters[pn])
        )
        for param_name in param_names
    }
    cache_key = (module.__class__, frozenset(param_names))
    if cache_key in _wrap_class_cache:
        module_cls = _wrap_class_cache[cache_key]
    else:
        module_cls = type(
            f"SimpleFSDP{module.__class__.__name__}",
            (module.__class__,),
            param_name_to_property,
        )
        sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls
        _wrap_class_cache[cache_key] = module_cls
    module.__class__ = module_cls


class ReplicateComputation(nn.Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: SimpleFSDPMixedPrecisionPolicy | None,
        full_dtensor: bool = False,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.param_sharding = param_sharding
        self.mode = mode
        self.compute_placements: list[Placement] = [Replicate()] * self.device_mesh.ndim
        self.grad_placements: list[Placement] = [Partial(reduce_op="sum")] * self.device_mesh.ndim
        mp_policy = mp_policy or SimpleFSDPMixedPrecisionPolicy()
        self.param_dtype = mp_policy.param_dtype
        self.reduce_dtype = mp_policy.reduce_dtype
        self.full_dtensor = full_dtensor

    def replicate_compute(self, x: DTensor) -> torch.Tensor:
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        assert non_dp_mesh_dims <= 2, "Only DP + EP/TP/EP+TP is supported"
        if self.param_dtype is not None and x.dtype != self.param_dtype:
            # Cast while the parameter is still sharded so the following
            # all-gather materializes and communicates compute-dtype params.
            x = x.to(self.param_dtype)
        if non_dp_mesh_dims > 0:
            if self.full_dtensor:
                raise NotImplementedError("full_dtensor not implemented for nD parallelisms")
            dp_mesh = self.device_mesh
            sharded_local_tensor = x.to_local()
            sharded_dtensor = DTensor.from_local(sharded_local_tensor, dp_mesh, self.param_sharding)
            replicated_dtensor = sharded_dtensor.redistribute(
                placements=self.compute_placements,
            )
            replicated_local_tensor = replicated_dtensor.to_local(
                grad_placements=self.grad_placements
            )
            non_dp_placements = tuple(x._spec.placements[-non_dp_mesh_dims:])
            non_dp_mesh_dim_names = tuple(x._spec.mesh.mesh_dim_names[-non_dp_mesh_dims:])
            non_dp_mesh = x._spec.mesh[non_dp_mesh_dim_names]
            return DTensor.from_local(replicated_local_tensor, non_dp_mesh, non_dp_placements)

        output = x.redistribute(placements=self.compute_placements)
        if not self.full_dtensor:
            output = output.to_local(grad_placements=self.grad_placements)
        return output

    def forward(self, x: DTensor) -> torch.Tensor:
        if not _active_parametrization:
            return x
        return self.replicate_compute(x)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "fully_shard",
    mp_policy: SimpleFSDPMixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    full_dtensor: bool = False,
    replicate_numel_threshold: int = 0,
) -> nn.Module:
    if mode == "replicate":
        param_sharding: tuple[Placement, ...] = (Replicate(),)
    elif mode == "fully_shard":
        param_sharding = (Shard(shard_dim),)
    elif mode == "hybrid_shard":
        param_sharding = (Replicate(), Shard(shard_dim))
        assert device_mesh.ndim == 2, "hybrid sharded data parallel requires 2D DeviceMesh"
    else:
        raise ValueError(f"Unsupported SimpleFSDP mode {mode}")

    for mod in list(model.modules()):
        params_dict = dict(mod.named_parameters(recurse=False))
        if "SimpleFSDP" in mod.__class__.__name__:
            continue
        wrapped_param_names = []
        for p_name, p in params_dict.items():
            if p is None or p.numel() == 0:
                continue
            param_placements = param_sharding
            if replicate_numel_threshold > 0 and int(p.numel()) <= replicate_numel_threshold:
                param_placements = (Replicate(),)
            distribute_tensor_func = _distribute_dtensor if isinstance(p, DTensor) else distribute_tensor
            dtensor_param = distribute_tensor_func(p, device_mesh, param_placements)
            mod.register_parameter(p_name, nn.Parameter(dtensor_param))
            wrapped_param_names.append(p_name)
        _register_parametrization(
            mod,
            wrapped_param_names,
            ReplicateComputation(
                device_mesh,
                param_sharding,
                mode,
                mp_policy=mp_policy,
                full_dtensor=full_dtensor,
            ),
        )
    return model


def summarize_simplefsdp_parameters(model: nn.Module) -> dict:
    dtensor_params = 0
    replicated_params = 0
    replicated_global_numel = 0
    sharded_params = 0
    local_numel = 0
    global_numel = 0
    wrapped_modules = 0
    for module in model.modules():
        if "SimpleFSDP" in module.__class__.__name__:
            wrapped_modules += 1
        for param in module.parameters(recurse=False):
            if isinstance(param, DTensor):
                dtensor_params += 1
                local_numel += int(param.to_local().numel())
                param_numel = int(param.numel())
                global_numel += param_numel
                placements = tuple(param._spec.placements)
                if placements and placements[0].is_replicate():
                    replicated_params += 1
                    replicated_global_numel += param_numel
                elif placements and placements[0].is_shard():
                    sharded_params += 1
    return {
        "dtensor_params": dtensor_params,
        "replicated_params": replicated_params,
        "replicated_global_numel": replicated_global_numel,
        "sharded_params": sharded_params,
        "local_numel": local_numel,
        "global_numel": global_numel,
        "wrapped_modules": wrapped_modules,
    }



def _node_nbytes(node) -> int:
    val = node.meta.get("val") if hasattr(node, "meta") else None
    if isinstance(val, (tuple, list)) and val:
        val = val[0]
    if hasattr(val, "numel") and hasattr(val, "element_size"):
        try:
            return int(val.numel() * val.element_size())
        except Exception:
            return 0
    return 0


def _bucket_collective_nodes(nodes, max_bucket_bytes: int):
    buckets = []
    current = []
    current_bytes = 0
    current_key = None
    for node in nodes:
        node_bytes = max(1, _node_nbytes(node.args[0]))
        if (
            current
            and current_key == node.args[1:]
            and current_bytes + node_bytes <= max_bucket_bytes
        ):
            current.append(node)
            current_bytes += node_bytes
            continue
        if current:
            buckets.append(current)
        current = [node]
        current_bytes = node_bytes
        current_key = node.args[1:]
    if current:
        buckets.append(current)
    return buckets


def _iter_fx_nodes(arg):
    if hasattr(arg, "op") and hasattr(arg, "users"):
        yield arg
    elif isinstance(arg, (tuple, list)):
        for item in arg:
            yield from _iter_fx_nodes(item)
    elif isinstance(arg, dict):
        for item in arg.values():
            yield from _iter_fx_nodes(item)


def _coalesce_all_gather_bucket(graph, bucket) -> bool:
    if len(bucket) < 2:
        return False
    group_size = bucket[0].args[1]
    group_name = bucket[0].args[2]
    inputs = [node.args[0] for node in bucket]

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    input_nodes = [node for inp in inputs for node in _iter_fx_nodes(inp)]
    latest_input = max(input_nodes, key=lambda node: node_order[node], default=None)
    latest_input_idx = node_order[latest_input] if latest_input is not None else -1
    users = [user for node in bucket for user in node.users if user not in bucket]
    if not users:
        return False
    earliest_user_idx = min(node_order[user] for user in users)

    # Some Inductor schedules raise all-gathers before later parameter casts.
    # Coalescing those nodes would either break FX topological order or delay an
    # already-consumed all-gather. Skip that bucket and leave the native nodes.
    if latest_input_idx >= earliest_user_idx:
        return False

    if latest_input is None:
        with graph.inserting_before(bucket[0]):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
                args=(inputs, group_size, group_name),
            )
    else:
        with graph.inserting_after(latest_input):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
                args=(inputs, group_size, group_name),
            )

    items = []
    insert_after = coalesced
    for idx, old_node in enumerate(bucket):
        with graph.inserting_after(insert_after):
            item = graph.call_function(__import__("operator").getitem, args=(coalesced, idx))
        items.append(item)
        insert_after = item

    for old_node, item in zip(bucket, items):
        old_node.replace_all_uses_with(item)
    for old_node in bucket:
        graph.erase_node(old_node)
    return True



def simplefsdp_coalesce_collectives_graph_pass(graph, max_bucket_bytes: int = 128 * 1024 * 1024):
    """Coalesce SimpleFSDP functional all-gather nodes in a PyTorch 2.7 FX graph.

    TorchTitan GraphTrainer uses newer bucketing/overlap passes that are not
    present in this environment. This local pass keeps the same core idea for
    SimpleFSDP: once DTensor makes collectives visible in the graph, pack nearby
    parameter-level all-gathers into c10d coalesced collectives so Inductor sees
    fewer, larger communication nodes to schedule and overlap.

    Reduce-scatter coalescing needs a backward-specific scheduler: inserting a
    coalesced reduce-scatter at the first node in a bucket can reference gradient
    tensors that are produced later in the backward graph. Leave RS nodes in their
    original order unless TorchTitan's newer scheduling passes are available.
    """
    max_bucket_bytes = int(max_bucket_bytes)
    if max_bucket_bytes <= 0:
        return graph

    ag_nodes = []
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            ag_nodes.append(node)

    changed = False
    for bucket in _bucket_collective_nodes(ag_nodes, max_bucket_bytes):
        changed = _coalesce_all_gather_bucket(graph, bucket) or changed
    if changed:
        graph.lint()
    return graph
