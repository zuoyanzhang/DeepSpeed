# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from TorchTitan graph_trainer/simple_fsdp.py for the DeepCompile
# benchmark harness. This is the compiler-friendly SimpleFSDP core: parameters
# are stored as DTensor shards, and parameter access performs a traceable
# redistribute to replicated local tensors.

from __future__ import annotations

import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import operator

import torch
import torch.nn as nn

from torch.distributed._tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._redistribute import redistribute_local_tensor
from torch.distributed.tensor.placement_types import _StridedShard, Placement

_active_parametrization = True
_LATEST_CHORUS_GRAPH_STATS: dict[str, int | float | str] = {}
_CHORUS_GRAPH_HISTORY: list[dict[str, int | float | str]] = []
_CHORUS_RETAINED_TENSORS: dict[int, torch.Tensor] = {}
_CHORUS_RETAINED_BYTES_BY_KEY: dict[int, int] = {}
_CHORUS_FORWARD_RETAIN_SIGNATURE_TO_KEY: dict[tuple, int] = {}
_CHORUS_FORWARD_RETAIN_BYTES: dict[tuple, int] = {}
_CHORUS_NEXT_RETAIN_KEY = 1
_CHORUS_RUNTIME_MAX_LIVE_BYTES = 0
_CHORUS_RUNTIME_USABLE_FRACTION = 0.90
_CHORUS_RUNTIME_STATIC_MARGIN_BYTES = 1024 * 1024 * 1024
_CHORUS_RUNTIME_RETAINED_BYTES = 0
_CHORUS_RUNTIME_RETAIN_CLONE = True
_CHORUS_RUNTIME_STATS: dict[str, int | float] = {}


_chorus_lib = torch.library.Library("simplefsdp_chorus", "DEF")
_chorus_lib.define("retain_put(Tensor x, int key) -> Tensor")
_chorus_lib.define("retain_get(Tensor local_shard, int key, int group_size, str group_name) -> Tensor")


def _new_chorus_runtime_stats() -> dict[str, int | float]:
    return {
        "configured": 0,
        "max_live_bytes": 0,
        "usable_fraction": 0.0,
        "static_margin_bytes": 0,
        "safe_total_bytes": 0,
        "total_mem_bytes": 0,
        "attempted_puts": 0,
        "admitted_puts": 0,
        "admitted_bytes": 0,
        "retain_aliases": 0,
        "retain_clones": 0,
        "clone_mode": 0,
        "skipped_disabled": 0,
        "skipped_live_budget": 0,
        "skipped_memory_pressure": 0,
        "skipped_oom": 0,
        "hit_gets": 0,
        "fallback_gets": 0,
        "evicted_keys": 0,
        "current_retained_bytes": 0,
        "current_retained_tensors": 0,
        "peak_retained_bytes": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "peak_driver_used_bytes": 0,
        "last_required_bytes": 0,
        "last_available_live_budget_bytes": 0,
        "last_driver_used_bytes": 0,
        "last_reusable_cache_bytes": 0,
        "last_driver_growth_bytes": 0,
        "last_projected_used_bytes": 0,
    }


_CHORUS_RUNTIME_STATS = _new_chorus_runtime_stats()


def _chorus_tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _drop_chorus_retained_key(key: int) -> None:
    global _CHORUS_RUNTIME_RETAINED_BYTES
    key = int(key)
    old = _CHORUS_RETAINED_TENSORS.pop(key, None)
    old_bytes = int(_CHORUS_RETAINED_BYTES_BY_KEY.pop(key, 0))
    if old is not None:
        _CHORUS_RUNTIME_STATS["evicted_keys"] = int(_CHORUS_RUNTIME_STATS.get("evicted_keys", 0)) + 1
    if old_bytes > 0:
        _CHORUS_RUNTIME_RETAINED_BYTES = max(0, int(_CHORUS_RUNTIME_RETAINED_BYTES) - old_bytes)


def clear_simplefsdp_chorus_cross_graph_cache() -> None:
    global _CHORUS_RUNTIME_RETAINED_BYTES
    _CHORUS_RETAINED_TENSORS.clear()
    _CHORUS_RETAINED_BYTES_BY_KEY.clear()
    _CHORUS_RUNTIME_RETAINED_BYTES = 0
    _CHORUS_RUNTIME_STATS["current_retained_bytes"] = 0
    _CHORUS_RUNTIME_STATS["current_retained_tensors"] = 0


def configure_simplefsdp_chorus_runtime_memory_budget(
    max_live_bytes: int,
    usable_fraction: float = 0.90,
    static_margin_mb: int = 1024,
) -> dict[str, int | float]:
    """Configure runtime admission for cross-graph retained all-gathers.

    The graph-level MILP decides which all-gathers are worth retaining, but this
    guard is the runtime safety loop: each retain_put is admitted only if the
    real CUDA allocator/driver state still has enough room. If not, retain_get
    falls back to the original all-gather, preserving correctness.
    """
    global _CHORUS_RUNTIME_MAX_LIVE_BYTES
    global _CHORUS_RUNTIME_USABLE_FRACTION
    global _CHORUS_RUNTIME_STATIC_MARGIN_BYTES
    global _CHORUS_RUNTIME_RETAINED_BYTES
    global _CHORUS_RUNTIME_RETAIN_CLONE
    global _CHORUS_RUNTIME_STATS
    global _CHORUS_NEXT_RETAIN_KEY

    _CHORUS_RUNTIME_MAX_LIVE_BYTES = max(0, int(max_live_bytes))
    _CHORUS_RUNTIME_USABLE_FRACTION = max(0.0, min(1.0, float(usable_fraction)))
    _CHORUS_RUNTIME_STATIC_MARGIN_BYTES = max(0, int(static_margin_mb)) * 1024 * 1024
    _CHORUS_RUNTIME_RETAIN_CLONE = os.getenv("SIMPLEFSDP_CHORUS_RETAIN_CLONE", "1") == "1"
    _CHORUS_RUNTIME_RETAINED_BYTES = 0
    _CHORUS_RUNTIME_STATS = _new_chorus_runtime_stats()
    _CHORUS_RUNTIME_STATS.update({
        "configured": 1,
        "max_live_bytes": int(_CHORUS_RUNTIME_MAX_LIVE_BYTES),
        "usable_fraction": float(_CHORUS_RUNTIME_USABLE_FRACTION),
        "static_margin_bytes": int(_CHORUS_RUNTIME_STATIC_MARGIN_BYTES),
        "clone_mode": int(bool(_CHORUS_RUNTIME_RETAIN_CLONE)),
    })
    _CHORUS_RETAINED_TENSORS.clear()
    _CHORUS_RETAINED_BYTES_BY_KEY.clear()
    _CHORUS_FORWARD_RETAIN_SIGNATURE_TO_KEY.clear()
    _CHORUS_FORWARD_RETAIN_BYTES.clear()
    _CHORUS_NEXT_RETAIN_KEY = 1
    _CHORUS_GRAPH_HISTORY.clear()

    if torch.cuda.is_available():
        try:
            total_mem = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
            _CHORUS_RUNTIME_STATS["total_mem_bytes"] = int(total_mem)
            _CHORUS_RUNTIME_STATS["safe_total_bytes"] = int(float(total_mem) * _CHORUS_RUNTIME_USABLE_FRACTION)
        except Exception:
            pass
    return dict(_CHORUS_RUNTIME_STATS)


def summarize_simplefsdp_chorus_runtime_retention() -> dict[str, int | float]:
    stats = dict(_CHORUS_RUNTIME_STATS)
    stats["current_retained_bytes"] = int(_CHORUS_RUNTIME_RETAINED_BYTES)
    stats["current_retained_tensors"] = int(len(_CHORUS_RETAINED_TENSORS))
    return stats


def _chorus_runtime_memory_snapshot(tensor: torch.Tensor | None = None) -> tuple[int, int, int, int, int]:
    total_mem = int(_CHORUS_RUNTIME_STATS.get("total_mem_bytes", 0))
    allocated = 0
    reserved = 0
    driver_used = 0
    safe_total = int(_CHORUS_RUNTIME_STATS.get("safe_total_bytes", 0))
    if not torch.cuda.is_available():
        return total_mem, safe_total, allocated, reserved, driver_used
    try:
        device = tensor.device if tensor is not None and tensor.is_cuda else torch.cuda.current_device()
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        if total_mem <= 0:
            total_mem = int(torch.cuda.get_device_properties(device).total_memory)
        try:
            free_mem, mem_total = torch.cuda.mem_get_info(device)
        except TypeError:
            with torch.cuda.device(device):
                free_mem, mem_total = torch.cuda.mem_get_info()
        total_mem = int(mem_total or total_mem)
        driver_used = max(0, int(total_mem) - int(free_mem))
        if safe_total <= 0 and total_mem > 0:
            safe_total = int(float(total_mem) * _CHORUS_RUNTIME_USABLE_FRACTION)
    except Exception:
        pass
    _CHORUS_RUNTIME_STATS["total_mem_bytes"] = int(total_mem)
    _CHORUS_RUNTIME_STATS["safe_total_bytes"] = int(safe_total)
    _CHORUS_RUNTIME_STATS["peak_allocated_bytes"] = max(
        int(_CHORUS_RUNTIME_STATS.get("peak_allocated_bytes", 0)), int(allocated)
    )
    _CHORUS_RUNTIME_STATS["peak_reserved_bytes"] = max(
        int(_CHORUS_RUNTIME_STATS.get("peak_reserved_bytes", 0)), int(reserved)
    )
    _CHORUS_RUNTIME_STATS["peak_driver_used_bytes"] = max(
        int(_CHORUS_RUNTIME_STATS.get("peak_driver_used_bytes", 0)), int(driver_used)
    )
    return int(total_mem), int(safe_total), int(allocated), int(reserved), int(driver_used)


def _chorus_can_admit_retain(required_bytes: int, tensor: torch.Tensor, *, will_allocate: bool) -> bool:
    required_bytes = max(0, int(required_bytes))
    _CHORUS_RUNTIME_STATS["last_required_bytes"] = int(required_bytes)
    if _CHORUS_RUNTIME_MAX_LIVE_BYTES <= 0:
        _CHORUS_RUNTIME_STATS["skipped_disabled"] = int(_CHORUS_RUNTIME_STATS.get("skipped_disabled", 0)) + 1
        return False
    live_remaining = int(_CHORUS_RUNTIME_MAX_LIVE_BYTES) - int(_CHORUS_RUNTIME_RETAINED_BYTES)
    _CHORUS_RUNTIME_STATS["last_available_live_budget_bytes"] = int(live_remaining)
    if required_bytes <= 0 or required_bytes > live_remaining:
        _CHORUS_RUNTIME_STATS["skipped_live_budget"] = int(_CHORUS_RUNTIME_STATS.get("skipped_live_budget", 0)) + 1
        return False

    _, safe_total, allocated, reserved, driver_used = _chorus_runtime_memory_snapshot(tensor)
    _CHORUS_RUNTIME_STATS["last_driver_used_bytes"] = int(driver_used)
    if safe_total > 0:
        margin = int(_CHORUS_RUNTIME_STATIC_MARGIN_BYTES)
        allocator_floor = max(int(allocated), int(reserved))
        extra_allocation = int(required_bytes) if bool(will_allocate) else 0
        _CHORUS_RUNTIME_STATS["last_reusable_cache_bytes"] = max(0, int(reserved) - int(allocated))
        _CHORUS_RUNTIME_STATS["last_driver_growth_bytes"] = int(extra_allocation)
        _CHORUS_RUNTIME_STATS["last_projected_used_bytes"] = int(driver_used) + int(extra_allocation)
        # Clone-retention is the safe default for Inductor-compiled graphs.
        # Alias-retention can be enabled for experiments with
        # SIMPLEFSDP_CHORUS_RETAIN_CLONE=0; in that mode the already materialized
        # all-gather output is kept alive instead of allocating another copy.
        if driver_used + extra_allocation + margin > safe_total:
            _CHORUS_RUNTIME_STATS["skipped_memory_pressure"] = int(
                _CHORUS_RUNTIME_STATS.get("skipped_memory_pressure", 0)
            ) + 1
            return False
        if allocator_floor + extra_allocation + margin > safe_total:
            _CHORUS_RUNTIME_STATS["skipped_memory_pressure"] = int(
                _CHORUS_RUNTIME_STATS.get("skipped_memory_pressure", 0)
            ) + 1
            return False
    return True


@torch.library.impl(_chorus_lib, "retain_put", "CUDA")
def _chorus_retain_put_cuda(x: torch.Tensor, key: int) -> torch.Tensor:
    waited = torch.ops._c10d_functional.wait_tensor.default(x)
    key = int(key)
    nbytes = _chorus_tensor_nbytes(waited)
    _CHORUS_RUNTIME_STATS["attempted_puts"] = int(_CHORUS_RUNTIME_STATS.get("attempted_puts", 0)) + 1
    _drop_chorus_retained_key(key)
    retain_clone = bool(_CHORUS_RUNTIME_RETAIN_CLONE)
    if not _chorus_can_admit_retain(nbytes, waited, will_allocate=retain_clone):
        return waited
    try:
        retained = waited.detach().clone() if retain_clone else waited.detach()
    except torch.OutOfMemoryError:
        _CHORUS_RUNTIME_STATS["skipped_oom"] = int(_CHORUS_RUNTIME_STATS.get("skipped_oom", 0)) + 1
        _drop_chorus_retained_key(key)
        return waited
    _CHORUS_RETAINED_TENSORS[key] = retained
    _CHORUS_RETAINED_BYTES_BY_KEY[key] = int(nbytes)
    global _CHORUS_RUNTIME_RETAINED_BYTES
    _CHORUS_RUNTIME_RETAINED_BYTES += int(nbytes)
    _CHORUS_RUNTIME_STATS["admitted_puts"] = int(_CHORUS_RUNTIME_STATS.get("admitted_puts", 0)) + 1
    _CHORUS_RUNTIME_STATS["admitted_bytes"] = int(_CHORUS_RUNTIME_STATS.get("admitted_bytes", 0)) + int(nbytes)
    if retain_clone:
        _CHORUS_RUNTIME_STATS["retain_clones"] = int(_CHORUS_RUNTIME_STATS.get("retain_clones", 0)) + 1
    else:
        _CHORUS_RUNTIME_STATS["retain_aliases"] = int(_CHORUS_RUNTIME_STATS.get("retain_aliases", 0)) + 1
    _CHORUS_RUNTIME_STATS["current_retained_bytes"] = int(_CHORUS_RUNTIME_RETAINED_BYTES)
    _CHORUS_RUNTIME_STATS["current_retained_tensors"] = int(len(_CHORUS_RETAINED_TENSORS))
    _CHORUS_RUNTIME_STATS["peak_retained_bytes"] = max(
        int(_CHORUS_RUNTIME_STATS.get("peak_retained_bytes", 0)),
        int(_CHORUS_RUNTIME_RETAINED_BYTES),
    )
    return waited


@torch.library.impl(_chorus_lib, "retain_put", "CPU")
def _chorus_retain_put_cpu(x: torch.Tensor, key: int) -> torch.Tensor:
    key = int(key)
    _drop_chorus_retained_key(key)
    _CHORUS_RETAINED_TENSORS[key] = x.detach().clone()
    _CHORUS_RETAINED_BYTES_BY_KEY[key] = _chorus_tensor_nbytes(x)
    return x


@torch.library.impl(_chorus_lib, "retain_put", "Meta")
def _chorus_retain_put_meta(x: torch.Tensor, key: int) -> torch.Tensor:
    return x.new_empty_strided(tuple(x.shape), tuple(x.stride()))


def _chorus_all_gather_fallback(local_shard: torch.Tensor, group_size: int, group_name: str) -> torch.Tensor:
    gathered = torch.ops._c10d_functional.all_gather_into_tensor.default(
        local_shard,
        int(group_size),
        str(group_name),
    )
    return torch.ops._c10d_functional.wait_tensor.default(gathered)


@torch.library.impl(_chorus_lib, "retain_get", "CUDA")
def _chorus_retain_get_cuda(
    local_shard: torch.Tensor,
    key: int,
    group_size: int,
    group_name: str,
) -> torch.Tensor:
    retained = _CHORUS_RETAINED_TENSORS.get(int(key))
    if retained is None:
        _CHORUS_RUNTIME_STATS["fallback_gets"] = int(_CHORUS_RUNTIME_STATS.get("fallback_gets", 0)) + 1
        return _chorus_all_gather_fallback(local_shard, int(group_size), str(group_name))
    _CHORUS_RUNTIME_STATS["hit_gets"] = int(_CHORUS_RUNTIME_STATS.get("hit_gets", 0)) + 1
    if os.getenv("SIMPLEFSDP_CHORUS_VALIDATE_RETAIN", "0") == "1":
        expected = _chorus_all_gather_fallback(local_shard, int(group_size), str(group_name))
        try:
            diff = (expected - retained).abs().max().item()
            if diff != 0.0:
                print(f"[simplefsdp-chorus-validate] key={int(key)} max_diff={diff}")
        except Exception as exc:
            print(f"[simplefsdp-chorus-validate] key={int(key)} compare_failed={exc}")
    return retained


@torch.library.impl(_chorus_lib, "retain_get", "CPU")
def _chorus_retain_get_cpu(
    local_shard: torch.Tensor,
    key: int,
    group_size: int,
    group_name: str,
) -> torch.Tensor:
    retained = _CHORUS_RETAINED_TENSORS.get(int(key))
    if retained is None:
        _CHORUS_RUNTIME_STATS["fallback_gets"] = int(_CHORUS_RUNTIME_STATS.get("fallback_gets", 0)) + 1
        return _chorus_all_gather_fallback(local_shard, int(group_size), str(group_name))
    _CHORUS_RUNTIME_STATS["hit_gets"] = int(_CHORUS_RUNTIME_STATS.get("hit_gets", 0)) + 1
    return retained


@torch.library.impl(_chorus_lib, "retain_get", "Meta")
def _chorus_retain_get_meta(
    local_shard: torch.Tensor,
    key: int,
    group_size: int,
    group_name: str,
) -> torch.Tensor:
    shape = list(local_shard.shape)
    if shape:
        shape[0] = int(shape[0]) * int(group_size)
    return local_shard.new_empty(tuple(shape))


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
            lambda self, pn=param_name: self._simplefsdp_parametrization(self._parameters[pn], pn)
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
    module._simplefsdp_parametrization = parametrization
    module.__class__ = module_cls


class ReplicateComputation(nn.Module):
    def __init__(
        self,
        device_mesh: DeviceMesh,
        param_sharding: tuple[Placement, ...],
        mode: str,
        mp_policy: SimpleFSDPMixedPrecisionPolicy | None,
        full_dtensor: bool = False,
        persistent_param_names: Sequence[str] | None = None,
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
        self.persistent_param_names = frozenset(str(name) for name in (persistent_param_names or ()))
        self._chorus_persistent_cache: dict[str, tuple[int | None, torch.Tensor]] = {}
        self._chorus_trace_cache_hits = 0
        self._chorus_trace_cache_misses = 0
        self._chorus_eager_cache_hits = 0
        self._chorus_eager_cache_misses = 0
        self._chorus_no_grad_bypass = 0

    @staticmethod
    def _is_compiling() -> bool:
        try:
            import torch._dynamo
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False

    def clear_chorus_cache(self) -> None:
        self._chorus_persistent_cache.clear()

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

    def persistent_replicate_compute(self, x: DTensor, param_name: str) -> torch.Tensor:
        # Keep the canonical parameter sharded. Chorus persistent retention only
        # reuses the full-parameter all-gather result, matching ZeRO3 persistent
        # semantics without replicating optimizer state.
        if self.full_dtensor:
            return self.replicate_compute(x)
        non_dp_mesh_dims = x._spec.mesh.ndim - self.device_mesh.ndim
        if non_dp_mesh_dims > 0:
            return self.replicate_compute(x)

        cache_key = str(param_name)
        if not torch.is_grad_enabled():
            self._chorus_no_grad_bypass += 1
            return self.replicate_compute(x)
        if self._is_compiling():
            cached = self._chorus_persistent_cache.get(cache_key)
            if cached is not None:
                self._chorus_trace_cache_hits += 1
                return cached[1]
            self._chorus_trace_cache_misses += 1
            output = self.replicate_compute(x)
            self._chorus_persistent_cache[cache_key] = (None, output)
            return output

        local_shard = x.to_local()
        version = int(getattr(local_shard, "_version", 0))
        cached = self._chorus_persistent_cache.get(cache_key)
        if cached is not None and cached[0] == version:
            self._chorus_eager_cache_hits += 1
            return cached[1]
        self._chorus_eager_cache_misses += 1
        output = self.replicate_compute(x)
        self._chorus_persistent_cache[cache_key] = (version, output)
        return output

    def forward(self, x: DTensor, param_name: str | None = None) -> torch.Tensor:
        if not _active_parametrization:
            return x
        if param_name is not None and str(param_name) in self.persistent_param_names:
            return self.persistent_replicate_compute(x, str(param_name))
        return self.replicate_compute(x)


def data_parallel(
    model: nn.Module,
    device_mesh: DeviceMesh,
    mode: str = "fully_shard",
    mp_policy: SimpleFSDPMixedPrecisionPolicy | None = None,
    shard_dim: int = 0,
    full_dtensor: bool = False,
    replicate_numel_threshold: int = 0,
    replicate_module_attr: str = "_simplefsdp_chorus_global_retain",
    replicate_param_names_attr: str = "_simplefsdp_chorus_global_retain_params",
    persistent_param_names_attr: str = "_simplefsdp_chorus_persistent_params",
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
        persistent_param_names = set(str(name) for name in getattr(mod, persistent_param_names_attr, ()))
        for p_name, p in params_dict.items():
            if p is None or p.numel() == 0:
                continue
            param_placements = param_sharding
            retained_param_names = getattr(mod, replicate_param_names_attr, ())
            if bool(getattr(mod, replicate_module_attr, False)) or p_name in retained_param_names:
                param_placements = (Replicate(),)
            elif replicate_numel_threshold > 0 and int(p.numel()) <= replicate_numel_threshold:
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
                persistent_param_names=persistent_param_names,
            ),
        )
    return model


def clear_simplefsdp_chorus_persistent_cache(model: nn.Module) -> None:
    seen: set[int] = set()
    roots = [model]
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None:
        roots.append(orig_mod)
    for root in roots:
        for module in root.modules():
            obj_id = id(module)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            parametrization = getattr(module, "_simplefsdp_parametrization", None)
            if hasattr(parametrization, "clear_chorus_cache"):
                parametrization.clear_chorus_cache()


def summarize_simplefsdp_chorus_persistent_cache(model: nn.Module) -> dict:
    seen: set[int] = set()
    stats = {
        "trace_hits": 0,
        "trace_misses": 0,
        "eager_hits": 0,
        "eager_misses": 0,
        "no_grad_bypass": 0,
        "cache_entries": 0,
        "persistent_params": 0,
        "modules": 0,
    }
    roots = [model]
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None:
        roots.append(orig_mod)
    for root in roots:
        for module in root.modules():
            parametrization = getattr(module, "_simplefsdp_parametrization", None)
            if parametrization is None or id(parametrization) in seen:
                continue
            seen.add(id(parametrization))
            stats["modules"] += 1
            stats["trace_hits"] += int(getattr(parametrization, "_chorus_trace_cache_hits", 0))
            stats["trace_misses"] += int(getattr(parametrization, "_chorus_trace_cache_misses", 0))
            stats["eager_hits"] += int(getattr(parametrization, "_chorus_eager_cache_hits", 0))
            stats["eager_misses"] += int(getattr(parametrization, "_chorus_eager_cache_misses", 0))
            stats["no_grad_bypass"] += int(getattr(parametrization, "_chorus_no_grad_bypass", 0))
            stats["cache_entries"] += int(len(getattr(parametrization, "_chorus_persistent_cache", {})))
            stats["persistent_params"] += int(len(getattr(parametrization, "persistent_param_names", ())))
    return stats


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


def _debug_node_name(arg) -> str:
    if hasattr(arg, "name"):
        return str(getattr(arg, "name"))
    return type(arg).__name__


def _maybe_dump_chorus_ag_signatures(graph, ag_nodes) -> None:
    if os.getenv("SIMPLEFSDP_CHORUS_DUMP_SIGNATURES", "0") != "1":
        return
    graph_idx = len(_CHORUS_GRAPH_HISTORY)
    print(f"[simplefsdp-chorus-dump] graph={graph_idx} ag_nodes={len(ag_nodes)}")
    for idx, node in enumerate(ag_nodes[:32]):
        inp = node.args[0] if node.args else None
        meta = getattr(node, "meta", {}) or {}
        inp_meta = getattr(inp, "meta", {}) or {}
        print(
            "[simplefsdp-chorus-dump] "
            f"idx={idx} node={getattr(node, 'name', '')} input={_debug_node_name(inp)} "
            f"nbytes={_node_nbytes(inp)} args_tail={node.args[1:]} "
            f"meta_keys={list(meta.keys())[:8]} input_meta_keys={list(inp_meta.keys())[:8]}"
        )
        stack = str(meta.get("stack_trace", "") or inp_meta.get("stack_trace", ""))
        if stack:
            print("[simplefsdp-chorus-dump-stack] " + stack.splitlines()[-1][:240])


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


@dataclass
class _CollectiveBucketInfo:
    nodes: list
    nbytes: int
    latest_input: object | None
    latest_input_idx: int
    earliest_user: object | None
    earliest_user_idx: int


@dataclass
class _ChorusBucketSchedule:
    anchors: list[int]
    retained_parent: dict[int, int]
    method: str
    objective: float
    solve_time_s: float
    status: int


def _collective_bucket_info(graph, bucket) -> _CollectiveBucketInfo | None:
    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    inputs = [node.args[0] for node in bucket]
    input_nodes = [node for inp in inputs for node in _iter_fx_nodes(inp)]
    latest_input = max(input_nodes, key=lambda node: node_order[node], default=None)
    latest_input_idx = node_order[latest_input] if latest_input is not None else -1
    users = [user for node in bucket for user in node.users if user not in bucket]
    if not users:
        return None
    earliest_user = min(users, key=lambda node: node_order[node])
    earliest_user_idx = node_order[earliest_user]
    if latest_input_idx >= earliest_user_idx:
        return None
    return _CollectiveBucketInfo(
        nodes=list(bucket),
        nbytes=sum(max(1, _node_nbytes(node.args[0])) for node in bucket),
        latest_input=latest_input,
        latest_input_idx=latest_input_idx,
        earliest_user=earliest_user,
        earliest_user_idx=earliest_user_idx,
    )


def _collective_bucket_infos_with_fallback(graph, buckets) -> list[_CollectiveBucketInfo]:
    infos: list[_CollectiveBucketInfo] = []
    for bucket in buckets:
        info = _collective_bucket_info(graph, bucket)
        if info is not None:
            infos.append(info)
            continue
        # A large communication bucket can be unschedulable as a unit because
        # earlier all-gathers are consumed before later bucket inputs exist.
        # Chorus still needs those operations as scheduling units, so fall back
        # to single-node units instead of silently dropping the whole bucket.
        if len(bucket) <= 1:
            continue
        for node in bucket:
            single_info = _collective_bucket_info(graph, [node])
            if single_info is not None:
                infos.append(single_info)
    return infos


def _hashable_fx_arg(arg):
    if hasattr(arg, "op") and hasattr(arg, "users"):
        return ("node", getattr(arg, "name", repr(arg)))
    if isinstance(arg, (tuple, list)):
        return tuple(_hashable_fx_arg(item) for item in arg)
    if isinstance(arg, dict):
        return tuple(sorted((key, _hashable_fx_arg(value)) for key, value in arg.items()))
    try:
        hash(arg)
        return arg
    except Exception:
        return repr(arg)


def _all_gather_retention_key(node):
    return (
        node.target,
        _hashable_fx_arg(node.args),
        _hashable_fx_arg(node.kwargs),
    )


def _bucket_retention_key(info: _CollectiveBucketInfo):
    return tuple(_all_gather_retention_key(node) for node in info.nodes)


def _tensor_signature(arg):
    meta = getattr(arg, "meta", {}) or {}
    val = meta.get("val") if "val" in meta else meta.get("tensor_meta")
    shape = ()
    dtype = ""
    if hasattr(val, "shape"):
        try:
            shape = tuple(int(dim) for dim in val.shape)
        except Exception:
            shape = tuple(str(dim) for dim in val.shape)
    elif hasattr(val, "shape"):
        shape = tuple(val.shape)
    if hasattr(val, "dtype"):
        dtype = str(val.dtype)
    return (shape, dtype)


def _all_gather_group_size(node) -> int:
    try:
        return int(node.args[1])
    except Exception:
        return 1


def _all_gather_group_name(node) -> str:
    try:
        return str(node.args[2])
    except Exception:
        return ""


def _all_gather_output_nbytes(node) -> int:
    output_nbytes = int(_node_nbytes(node))
    if output_nbytes > 0:
        return output_nbytes
    input_nbytes = int(_node_nbytes(node.args[0])) if node.args else 0
    return max(0, input_nbytes * max(1, _all_gather_group_size(node)))


def _all_gather_cross_graph_signature(node):
    inp = node.args[0] if node.args else None
    return (
        _debug_node_name(inp),
        _tensor_signature(inp),
        int(_node_nbytes(inp)),
        int(_all_gather_output_nbytes(node)),
        int(_all_gather_group_size(node)),
        str(_all_gather_group_name(node)),
    )


def _is_activation_recompute_all_gather(node) -> bool:
    meta = getattr(node, "meta", {}) or {}
    inp = node.args[0] if node.args else None
    inp_meta = getattr(inp, "meta", {}) or {}
    return bool(
        meta.get("recompute", False)
        or inp_meta.get("recompute", False)
        or "ac_graph_id" in meta
        or "ac_graph_id" in inp_meta
    )


def _unique_cross_graph_candidates(ag_nodes) -> list[tuple[tuple, object, int]]:
    candidates = []
    seen = set()
    for node in ag_nodes:
        if not _is_activation_recompute_all_gather(node):
            continue
        signature = _all_gather_cross_graph_signature(node)
        if signature in seen:
            continue
        seen.add(signature)
        nbytes = int(_all_gather_output_nbytes(node))
        if nbytes <= 0:
            continue
        candidates.append((signature, node, nbytes))
    return candidates


def _plan_cross_graph_retention(
    candidates: list[tuple[tuple, object, int]],
    max_live_bytes: int,
    milp_time_limit_s: float,
) -> tuple[set[tuple], dict[str, int | float | str]]:
    if not candidates:
        return set(), {"method": "empty", "status": 0, "solve_time_s": 0.0, "selected_bytes": 0, "selected": 0}
    budget = int(max_live_bytes)
    if budget <= 0:
        return set(), {
            "method": "disabled_no_live_budget",
            "status": -1,
            "solve_time_s": 0.0,
            "selected_bytes": 0,
            "selected": 0,
            "budget_bytes": 0,
        }

    selected: list[int] = []
    method = "greedy_cross_graph"
    status = -1
    solve_time_s = 0.0
    try:
        import time as _time
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        costs = np.array([float(nbytes) for _, _, nbytes in candidates], dtype=float)
        # The MILP objective models saved backward/recompute all-gather traffic.
        # Larger full parameters are preferred under the live-retention budget.
        values = costs.copy()
        matrix = coo_matrix((costs, (np.zeros(len(candidates), dtype=int), np.arange(len(candidates), dtype=int))),
                            shape=(1, len(candidates))).tocsr()
        start = _time.perf_counter()
        result = milp(
            -values,
            integrality=np.ones(len(candidates), dtype=int),
            bounds=Bounds(np.zeros(len(candidates), dtype=float), np.ones(len(candidates), dtype=float)),
            constraints=LinearConstraint(matrix, np.array([0.0]), np.array([float(budget)])),
            options={"time_limit": float(milp_time_limit_s), "mip_rel_gap": 0.001, "presolve": True},
        )
        solve_time_s = float(_time.perf_counter() - start)
        if result.x is not None and int(getattr(result, "status", -1)) in (0, 1):
            selected = [idx for idx, value in enumerate(result.x) if float(value) >= 0.5]
            method = "milp_cross_graph"
            status = int(getattr(result, "status", -1))
    except Exception:
        pass

    if not selected:
        used = 0
        for idx in sorted(range(len(candidates)), key=lambda i: candidates[i][2], reverse=True):
            nbytes = int(candidates[idx][2])
            if used + nbytes > budget:
                continue
            selected.append(idx)
            used += nbytes

    selected_signatures = {candidates[idx][0] for idx in selected}
    selected_bytes = int(sum(candidates[idx][2] for idx in selected))
    return selected_signatures, {
        "method": method,
        "status": int(status),
        "solve_time_s": float(solve_time_s),
        "selected_bytes": int(selected_bytes),
        "selected": int(len(selected_signatures)),
        "budget_bytes": int(budget),
    }


def _chorus_retain_key_for_signature(signature: tuple) -> int:
    global _CHORUS_NEXT_RETAIN_KEY
    key = _CHORUS_FORWARD_RETAIN_SIGNATURE_TO_KEY.get(signature)
    if key is not None:
        return int(key)
    key = int(_CHORUS_NEXT_RETAIN_KEY)
    _CHORUS_NEXT_RETAIN_KEY += 1
    _CHORUS_FORWARD_RETAIN_SIGNATURE_TO_KEY[signature] = key
    return key


def _insert_cross_graph_retain_puts(graph, ag_nodes, selected_signatures: set[tuple]) -> tuple[int, int]:
    puts = 0
    retained_bytes = 0
    for node in list(ag_nodes):
        signature = _all_gather_cross_graph_signature(node)
        if signature not in selected_signatures:
            continue
        key = _chorus_retain_key_for_signature(signature)
        nbytes = int(_all_gather_output_nbytes(node))
        _CHORUS_FORWARD_RETAIN_BYTES[signature] = nbytes
        with graph.inserting_after(node):
            retained = graph.call_function(
                torch.ops.simplefsdp_chorus.retain_put.default,
                args=(node, int(key)),
            )
        retained.meta = dict(getattr(node, "meta", {}))
        for user in list(node.users):
            if user is retained:
                continue
            user.replace_input_with(node, retained)
        puts += 1
        retained_bytes += nbytes
    return puts, retained_bytes


def _replace_cross_graph_retain_gets(graph, ag_nodes) -> tuple[int, int]:
    gets = 0
    retained_bytes = 0
    for node in list(ag_nodes):
        signature = _all_gather_cross_graph_signature(node)
        key = _CHORUS_FORWARD_RETAIN_SIGNATURE_TO_KEY.get(signature)
        if key is None:
            continue
        local_shard = node.args[0]
        group_size = int(_all_gather_group_size(node))
        group_name = str(_all_gather_group_name(node))
        with graph.inserting_before(node):
            retained = graph.call_function(
                torch.ops.simplefsdp_chorus.retain_get.default,
                args=(local_shard, int(key), group_size, group_name),
            )
        retained.meta = dict(getattr(node, "meta", {}))
        nbytes = int(_all_gather_output_nbytes(node))
        node.replace_all_uses_with(retained)
        if len(node.users) == 0:
            graph.erase_node(node)
        gets += 1
        retained_bytes += nbytes
    return gets, retained_bytes


def _apply_cross_graph_retention(
    graph,
    ag_nodes,
    has_reduce_scatter: bool,
    max_live_bytes: int,
    milp_time_limit_s: float,
) -> tuple[bool, dict[str, int | float | str]]:
    stats: dict[str, int | float | str] = {
        "cross_graph_puts": 0,
        "cross_graph_gets": 0,
        "cross_graph_selected": 0,
        "cross_graph_selected_bytes": 0,
        "cross_graph_get_bytes": 0,
        "cross_graph_budget_bytes": int(max_live_bytes),
        "cross_graph_method": "none",
        "cross_graph_status": -1,
        "cross_graph_solve_time_s": 0.0,
    }
    if not ag_nodes:
        return False, stats

    if not has_reduce_scatter:
        candidates = _unique_cross_graph_candidates(ag_nodes)
        selected_signatures, plan_stats = _plan_cross_graph_retention(
            candidates,
            max_live_bytes=max_live_bytes,
            milp_time_limit_s=milp_time_limit_s,
        )
        puts, selected_bytes = _insert_cross_graph_retain_puts(graph, ag_nodes, selected_signatures)
        stats.update({
            "cross_graph_puts": int(puts),
            "cross_graph_selected": int(plan_stats.get("selected", 0)),
            "cross_graph_selected_bytes": int(selected_bytes),
            "cross_graph_budget_bytes": int(plan_stats.get("budget_bytes", max_live_bytes)),
            "cross_graph_method": str(plan_stats.get("method", "none")),
            "cross_graph_status": int(plan_stats.get("status", -1)),
            "cross_graph_solve_time_s": float(plan_stats.get("solve_time_s", 0.0)),
        })
        return puts > 0, stats

    gets, get_bytes = _replace_cross_graph_retain_gets(graph, ag_nodes)
    stats.update({
        "cross_graph_gets": int(gets),
        "cross_graph_get_bytes": int(get_bytes),
        "cross_graph_method": "matched_cross_graph",
        "cross_graph_status": 0 if gets > 0 else -1,
    })
    return gets > 0, stats


def _retain_all_gather_bucket(graph, source: _CollectiveBucketInfo, target: _CollectiveBucketInfo) -> bool:
    if len(source.nodes) != len(target.nodes):
        return False
    for src_node, dst_node in zip(source.nodes, target.nodes):
        if _all_gather_retention_key(src_node) != _all_gather_retention_key(dst_node):
            return False
    for src_node, dst_node in zip(source.nodes, target.nodes):
        dst_node.replace_all_uses_with(src_node)
    for dst_node in target.nodes:
        if len(dst_node.users) == 0:
            graph.erase_node(dst_node)
    return True


def _greedy_chorus_bucket_schedule(
    infos: list[_CollectiveBucketInfo],
    prefetch_groups: int,
    max_live_bytes: int,
) -> _ChorusBucketSchedule:
    anchors = []
    for idx, info in enumerate(infos):
        target_idx = idx
        live_bytes = 0
        min_idx = max(0, idx - prefetch_groups)
        for candidate_idx in range(idx, min_idx - 1, -1):
            live_bytes += int(infos[candidate_idx].nbytes)
            if max_live_bytes > 0 and live_bytes > max_live_bytes and candidate_idx < idx:
                break
            target_idx = candidate_idx
        anchors.append(target_idx)
    return _ChorusBucketSchedule(
        anchors=anchors,
        retained_parent={},
        method="greedy_prefetch",
        objective=0.0,
        solve_time_s=0.0,
        status=-1,
    )


def _plan_chorus_bucket_schedule(
    infos: list[_CollectiveBucketInfo],
    prefetch_groups: int,
    max_live_bytes: int,
    milp_time_limit_s: float,
) -> _ChorusBucketSchedule:
    num_buckets = len(infos)
    prefetch_groups = max(0, int(prefetch_groups))
    max_live_bytes = max(0, int(max_live_bytes))
    if num_buckets <= 0:
        return _ChorusBucketSchedule([], {}, "empty", 0.0, 0.0, 0)
    if max_live_bytes <= 0:
        return _ChorusBucketSchedule(
            anchors=list(range(num_buckets)),
            retained_parent={},
            method="disabled_no_live_budget",
            objective=0.0,
            solve_time_s=0.0,
            status=-1,
        )

    # Build a bounded-window MILP over graph-visible FSDP collectives.
    # x(i,j)=1 means all-gather bucket i is materialized at anchor bucket j.
    # r(i,p)=1 means bucket i is locally retained from identical bucket p, saving
    # the second communication while paying live memory from p to i.
    variables = []
    objective = []
    by_bucket = [[] for _ in range(num_buckets)]
    live_terms: list[list[tuple[int, float]]] = [[] for _ in range(num_buckets)]

    for bucket_idx, info in enumerate(infos):
        min_anchor = max(0, bucket_idx - prefetch_groups)
        for anchor_idx in range(min_anchor, bucket_idx + 1):
            var_idx = len(variables)
            variables.append(("prefetch", bucket_idx, anchor_idx))
            by_bucket[bucket_idx].append(var_idx)
            distance = bucket_idx - anchor_idx
            objective.append(float(info.nbytes) * float(distance))
            if distance > 0 and max_live_bytes > 0:
                for live_idx in range(anchor_idx, bucket_idx):
                    live_terms[live_idx].append((var_idx, float(info.nbytes)))

    last_by_key = {}
    retention_candidates = []
    for bucket_idx, info in enumerate(infos):
        key = _bucket_retention_key(info)
        parents = last_by_key.get(key, [])
        # Keep the MILP compact: retaining from the nearest same-parameter
        # producers gives almost all of the benefit and avoids long live ranges.
        for parent_idx in parents[-2:]:
            if parent_idx >= bucket_idx:
                continue
            var_idx = len(variables)
            variables.append(("retain", bucket_idx, parent_idx))
            by_bucket[bucket_idx].append(var_idx)
            distance = bucket_idx - parent_idx
            objective.append(float(info.nbytes) * float(prefetch_groups + 1 + distance))
            retention_candidates.append((var_idx, bucket_idx, parent_idx))
            if max_live_bytes > 0:
                for live_idx in range(parent_idx, bucket_idx):
                    live_terms[live_idx].append((var_idx, float(info.nbytes)))
        parents.append(bucket_idx)
        last_by_key[key] = parents

    if not retention_candidates and prefetch_groups <= 0:
        return _greedy_chorus_bucket_schedule(infos, prefetch_groups, max_live_bytes)

    try:
        import time as _time
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        rows = []
        cols = []
        data = []
        lower = []
        upper = []
        row = 0

        for bucket_vars in by_bucket:
            for var_idx in bucket_vars:
                rows.append(row)
                cols.append(var_idx)
                data.append(1.0)
            lower.append(1.0)
            upper.append(1.0)
            row += 1

        if max_live_bytes > 0:
            for terms in live_terms:
                if not terms:
                    continue
                for var_idx, coeff in terms:
                    rows.append(row)
                    cols.append(var_idx)
                    data.append(coeff)
                lower.append(0.0)
                upper.append(float(max_live_bytes))
                row += 1

        matrix = coo_matrix(
            (
                np.array(data, dtype=float),
                (np.array(rows, dtype=int), np.array(cols, dtype=int)),
            ),
            shape=(row, len(variables)),
        ).tocsr()
        start = _time.perf_counter()
        result = milp(
            -np.array(objective, dtype=float),
            integrality=np.ones(len(variables), dtype=int),
            bounds=Bounds(
                np.zeros(len(variables), dtype=float),
                np.ones(len(variables), dtype=float),
            ),
            constraints=LinearConstraint(matrix, np.array(lower, dtype=float), np.array(upper, dtype=float)),
            options={"time_limit": float(milp_time_limit_s), "mip_rel_gap": 0.001, "presolve": True},
        )
        solve_time_s = float(_time.perf_counter() - start)
        if result.x is None or int(getattr(result, "status", -1)) not in (0, 1):
            return _greedy_chorus_bucket_schedule(infos, prefetch_groups, max_live_bytes)

        anchors = list(range(num_buckets))
        retained_parent = {}
        selected_objective = 0.0
        for var_idx, value in enumerate(result.x):
            if float(value) < 0.5:
                continue
            kind, bucket_idx, target_idx = variables[var_idx]
            selected_objective += float(objective[var_idx])
            if kind == "prefetch":
                anchors[int(bucket_idx)] = int(target_idx)
            elif kind == "retain":
                retained_parent[int(bucket_idx)] = int(target_idx)

        # Retention chains are legal in the MILP but the graph rewrite needs a
        # concrete producer node. Collapse every chain to its earliest live root.
        for bucket_idx in sorted(list(retained_parent)):
            seen = set()
            root = retained_parent[bucket_idx]
            while root in retained_parent and root not in seen:
                seen.add(root)
                root = retained_parent[root]
            retained_parent[bucket_idx] = root

        return _ChorusBucketSchedule(
            anchors=anchors,
            retained_parent=retained_parent,
            method="milp",
            objective=selected_objective,
            solve_time_s=solve_time_s,
            status=int(getattr(result, "status", -1)),
        )
    except Exception:
        return _greedy_chorus_bucket_schedule(infos, prefetch_groups, max_live_bytes)


def _iter_fx_nodes(arg):
    if hasattr(arg, "op") and hasattr(arg, "users"):
        yield arg
    elif isinstance(arg, (tuple, list)):
        for item in arg:
            yield from _iter_fx_nodes(item)
    elif isinstance(arg, dict):
        for item in arg.values():
            yield from _iter_fx_nodes(item)


def _move_single_all_gather_node(graph, node, insert_before=None) -> bool:
    info = _collective_bucket_info(graph, [node])
    if info is None:
        return False

    node_order = {n: idx for idx, n in enumerate(graph.nodes)}
    anchor = insert_before or info.earliest_user
    anchor_idx = node_order.get(anchor, info.earliest_user_idx)
    if info.latest_input_idx >= anchor_idx or anchor_idx > info.earliest_user_idx:
        anchor = info.earliest_user
        anchor_idx = info.earliest_user_idx
    if info.latest_input_idx >= anchor_idx:
        return False

    with graph.inserting_before(anchor):
        moved = graph.call_function(node.target, args=node.args, kwargs=node.kwargs)
    moved.meta = dict(getattr(node, "meta", {}))
    node.replace_all_uses_with(moved)
    graph.erase_node(node)
    return True


def _coalesce_all_gather_bucket(graph, bucket, insert_before=None) -> bool:
    if len(bucket) < 2:
        if len(bucket) == 1 and insert_before is not None:
            return _move_single_all_gather_node(graph, bucket[0], insert_before=insert_before)
        return False
    group_size = bucket[0].args[1]
    group_name = bucket[0].args[2]
    inputs = [node.args[0] for node in bucket]

    info = _collective_bucket_info(graph, bucket)
    if info is None:
        return False
    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    anchor = insert_before
    if anchor is not None:
        anchor_idx = node_order.get(anchor, info.earliest_user_idx)
        if info.latest_input_idx >= anchor_idx or anchor_idx > info.earliest_user_idx:
            anchor = None

    # Some Inductor schedules raise all-gathers before later parameter casts.
    # Coalescing those nodes would either break FX topological order or delay an
    # already-consumed all-gather. Skip that bucket and leave the native nodes.
    if info.latest_input_idx >= info.earliest_user_idx:
        return False

    if anchor is not None:
        with graph.inserting_before(anchor):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
                args=(inputs, group_size, group_name),
            )
    elif info.latest_input is None:
        with graph.inserting_before(bucket[0]):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
                args=(inputs, group_size, group_name),
            )
    else:
        with graph.inserting_after(info.latest_input):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.all_gather_into_tensor_coalesced.default,
                args=(inputs, group_size, group_name),
            )
    coalesced.meta = dict(getattr(bucket[0], "meta", {}))

    items = []
    insert_after = coalesced
    for idx, old_node in enumerate(bucket):
        with graph.inserting_after(insert_after):
            item = graph.call_function(operator.getitem, args=(coalesced, idx))
        item.meta = dict(getattr(old_node, "meta", {}))
        items.append(item)
        insert_after = item

    for old_node, item in zip(bucket, items):
        old_node.replace_all_uses_with(item)
    for old_node in bucket:
        graph.erase_node(old_node)
    return True



def _coalesce_reduce_scatter_bucket(graph, bucket) -> bool:
    if len(bucket) < 2:
        return False
    info = _collective_bucket_info(graph, bucket)
    if info is None:
        return False

    reduce_op = bucket[0].args[1]
    group_size = bucket[0].args[2]
    group_name = bucket[0].args[3]
    inputs = [node.args[0] for node in bucket]

    if info.latest_input is None:
        with graph.inserting_before(bucket[0]):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default,
                args=(inputs, reduce_op, group_size, group_name),
            )
    else:
        with graph.inserting_after(info.latest_input):
            coalesced = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor_coalesced.default,
                args=(inputs, reduce_op, group_size, group_name),
            )
    coalesced.meta = dict(getattr(bucket[0], "meta", {}))

    items = []
    insert_after = coalesced
    for idx, old_node in enumerate(bucket):
        with graph.inserting_after(insert_after):
            item = graph.call_function(operator.getitem, args=(coalesced, idx))
        item.meta = dict(getattr(old_node, "meta", {}))
        items.append(item)
        insert_after = item

    for old_node, item in zip(bucket, items):
        old_node.replace_all_uses_with(item)
    for old_node in bucket:
        if len(old_node.users) == 0:
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


def simplefsdp_chorus_collectives_graph_pass(
    graph,
    max_bucket_bytes: int = 128 * 1024 * 1024,
    prefetch_groups: int = 2,
    max_live_bytes: int = 4 * 1024 * 1024 * 1024,
    milp_time_limit_s: float = 2.0,
):
    """Apply the SimpleFSDP+Chorus graph schedule to visible collectives.

    SimpleFSDP exposes parameter all-gathers as functional collectives in the FX
    graph. Chorus first solves the graph-local MILP for prefetch placement and
    intra-graph local retention. For activation-checkpointed training, PyTorch
    emits separate forward and backward/recompute graphs, so this pass also adds
    graph-visible cross-graph retention: forward all-gathers selected by a MILP
    write their full parameters to retained buffers, while matching backward
    all-gathers read those buffers and are removed from the graph.
    """
    max_bucket_bytes = int(max_bucket_bytes)
    prefetch_groups = max(0, int(prefetch_groups))
    max_live_bytes = max(0, int(max_live_bytes))
    if max_live_bytes <= 0:
        prefetch_groups = 0
    global _LATEST_CHORUS_GRAPH_STATS
    _LATEST_CHORUS_GRAPH_STATS = {
        "enabled": 1,
        "graph_index": len(_CHORUS_GRAPH_HISTORY),
        "max_bucket_bytes": int(max_bucket_bytes),
        "prefetch_groups": int(prefetch_groups),
        "max_live_bytes": int(max_live_bytes),
        "ag_before": 0,
        "ag_buckets": 0,
        "ag_retained": 0,
        "ag_coalesced_or_moved": 0,
        "rs_before": 0,
        "rs_coalesced": 0,
        "cross_graph_puts": 0,
        "cross_graph_gets": 0,
        "cross_graph_selected": 0,
        "cross_graph_selected_bytes": 0,
        "cross_graph_get_bytes": 0,
        "cross_graph_budget_bytes": int(max_live_bytes),
        "cross_graph_method": "none",
        "cross_graph_status": -1,
        "cross_graph_solve_time_s": 0.0,
        "method": "none",
        "status": -1,
    }

    changed = False
    if max_bucket_bytes <= 0:
        _CHORUS_GRAPH_HISTORY.append(dict(_LATEST_CHORUS_GRAPH_STATS))
        return graph

    ag_nodes = []
    rs_nodes = []
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            ag_nodes.append(node)
        elif node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
            rs_nodes.append(node)
    _LATEST_CHORUS_GRAPH_STATS["ag_before"] = int(len(ag_nodes))
    _LATEST_CHORUS_GRAPH_STATS["rs_before"] = int(len(rs_nodes))
    _maybe_dump_chorus_ag_signatures(graph, ag_nodes)

    cross_changed, cross_stats = _apply_cross_graph_retention(
        graph,
        ag_nodes,
        has_reduce_scatter=bool(rs_nodes),
        max_live_bytes=max_live_bytes,
        milp_time_limit_s=float(milp_time_limit_s),
    )
    _LATEST_CHORUS_GRAPH_STATS.update(cross_stats)
    changed = cross_changed or changed

    if cross_changed:
        ag_nodes = []
        for node in list(graph.nodes):
            if node.op == "call_function" and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default:
                ag_nodes.append(node)

    if ag_nodes:
        raw_buckets = _bucket_collective_nodes(ag_nodes, max_bucket_bytes)
        infos = _collective_bucket_infos_with_fallback(graph, raw_buckets)
        _LATEST_CHORUS_GRAPH_STATS["ag_buckets"] = int(len(infos))
        if infos:
            schedule = _plan_chorus_bucket_schedule(
                infos,
                prefetch_groups=prefetch_groups,
                max_live_bytes=max_live_bytes,
                milp_time_limit_s=float(milp_time_limit_s),
            )

            _LATEST_CHORUS_GRAPH_STATS["method"] = str(schedule.method)
            _LATEST_CHORUS_GRAPH_STATS["status"] = int(schedule.status)
            retained_indices = set()
            for bucket_idx in sorted(schedule.retained_parent, reverse=True):
                parent_idx = schedule.retained_parent[bucket_idx]
                if not (0 <= parent_idx < len(infos)) or parent_idx == bucket_idx:
                    continue
                if _bucket_retention_key(infos[parent_idx]) != _bucket_retention_key(infos[bucket_idx]):
                    continue
                retained = _retain_all_gather_bucket(graph, infos[parent_idx], infos[bucket_idx])
                if retained:
                    retained_indices.add(bucket_idx)
                    _LATEST_CHORUS_GRAPH_STATS["ag_retained"] = int(_LATEST_CHORUS_GRAPH_STATS.get("ag_retained", 0)) + len(infos[bucket_idx].nodes)
                    changed = True

            for idx, info in enumerate(infos):
                if idx in retained_indices:
                    continue
                anchor_idx = idx
                if idx < len(schedule.anchors):
                    anchor_idx = schedule.anchors[idx]
                if not (0 <= anchor_idx < len(infos)):
                    anchor_idx = idx
                anchor = infos[anchor_idx].earliest_user if anchor_idx < idx else None
                bucket_changed = _coalesce_all_gather_bucket(graph, info.nodes, insert_before=anchor)
                if bucket_changed:
                    _LATEST_CHORUS_GRAPH_STATS["ag_coalesced_or_moved"] = int(_LATEST_CHORUS_GRAPH_STATS.get("ag_coalesced_or_moved", 0)) + len(info.nodes)
                changed = bucket_changed or changed

    rs_nodes = []
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target == torch.ops._c10d_functional.reduce_scatter_tensor.default:
            rs_nodes.append(node)
    _LATEST_CHORUS_GRAPH_STATS["rs_before"] = int(len(rs_nodes))
    for bucket in _bucket_collective_nodes(rs_nodes, max_bucket_bytes):
        bucket_changed = _coalesce_reduce_scatter_bucket(graph, bucket)
        if bucket_changed:
            _LATEST_CHORUS_GRAPH_STATS["rs_coalesced"] = int(_LATEST_CHORUS_GRAPH_STATS.get("rs_coalesced", 0)) + len(bucket)
        changed = bucket_changed or changed

    if changed:
        graph.lint()
    _CHORUS_GRAPH_HISTORY.append(dict(_LATEST_CHORUS_GRAPH_STATS))
    return graph
