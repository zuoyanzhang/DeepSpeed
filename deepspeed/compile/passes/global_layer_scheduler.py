# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# zuoyan

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..graph_param import DSGraphParamManager
from ..profilers.comm_profile import create_predictor
from ..util import (get_deepcompile_handle, get_tracked_step_peak_memory_bytes, get_tracked_step_peak_reserved_memory_bytes,
                    log_rank0)

NAME = "global_layer_scheduler"

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except Exception:
    unset_fake_temporarily = None


@dataclass(frozen=True)
class _SchedulerConfig:
    layer_regexes: Tuple[str, ...]
    lookahead_blocks: int
    fuse_max_bytes: int
    fuse_deadline_window_blocks: int
    fuse_factor: float
    inner_refine_max_iters: int
    inner_refine_min_gain_ms: float
    max_tasks_per_block: int
    dump_schedule: bool
    dump_dir: Optional[str]
    milp_time_limit_s: float
    milp_node_limit: int
    milp_rel_gap: float
    milp_presolve: bool
    separate_allgather_communicator: bool
    topology_policy: str
    low_comm_effective_bw_gib_per_s: float
    low_comm_pressure_ratio: float
    low_comm_keep_opportunity_factor: float
    low_comm_keep_cap_fraction: float
    low_comm_persistent_budget_mode: str
    low_comm_persistent_usable_fraction: float
    low_comm_persistent_starvation_fraction: float
    low_comm_graph_rewrite_mode: str
    low_comm_prefetch_fuse_slack: float
    low_comm_prefetch_fuse_max_bytes: int
    low_comm_prefetch_buffer_max_bytes: int
    low_comm_elide_persistent_releases: bool
    low_comm_elide_persistent_waits: bool
    low_comm_elide_persistent_prefetches: bool
    low_comm_persistent_value_mode: str
    low_comm_comm_value_weight: float
    low_comm_state_op_ms: float
    low_comm_recompute_relief: bool
    low_comm_recompute_pressure_ratio: float
    low_comm_post_step_refresh: bool


@dataclass(frozen=True)
class _LayerMapping:
    mapping_hash: str
    L: int
    layer_to_ds_ids: Tuple[Tuple[int, ...], ...]
    ds_id_to_layer: Dict[int, int]


_CFG: Optional[_SchedulerConfig] = None
_LAYER_MAPPING: Optional[_LayerMapping] = None
_LATEST_SCHEDULE: Optional[dict] = None
_PERSISTENT_SET_DONE: bool = False
_PERSISTENT_SET_LOCK = threading.Lock()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _schedule_hash(schedule_wo_hash: dict) -> str:
    schedule = dict(schedule_wo_hash)
    schedule.pop("schedule_hash", None)
    return _sha256_hex(_canonical_json(schedule).encode("utf-8"))


def _get_ds_id_from_wait(node: Node) -> int:
    assert node.target == torch.ops.dc.wait_allgather.default
    return int(node.args[2])


def _get_ds_id_from_release(node: Node) -> int:
    assert node.target == torch.ops.dc.release_param.default
    return int(node.args[2])


def _infer_layer_idx(param_name: str, layer_regexes: Sequence[str]) -> Optional[int]:
    for pattern in layer_regexes:
        m = re.search(pattern, param_name)
        if m is None:
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue
    return None



def _has_checkpointing_enabled(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if bool(getattr(module, "gradient_checkpointing", False)):
            return True
    return False


def _maybe_init_recompute_relief(model: torch.nn.Module, compile_config, schedule) -> None:
    if _CFG is None or not bool(_CFG.low_comm_recompute_relief):
        return
    if not _has_checkpointing_enabled(model):
        log_rank0(f"[{NAME}] Recompute relief disabled: model checkpointing is not enabled.", enable=True)
        return

    class _ChorusRecomputeConfigProxy:
        def __init__(self, base, pressure_ratio: float):
            self._base = base
            self._pressure_ratio = float(pressure_ratio)

        def __getattr__(self, key):
            if key == "selective_activation_recompute":
                return True
            if key == "global_layer_scheduler":
                return False
            if key == "selective_activation_recompute_pressure_ratio":
                return self._pressure_ratio
            return getattr(self._base, key)

    try:
        from . import selective_activation_recompute as recompute
        proxy = _ChorusRecomputeConfigProxy(compile_config, float(_CFG.low_comm_recompute_pressure_ratio))
        # Pass an empty schedule so the standalone pass does not see global_layer_scheduler;
        # this initializes its planner for Chorus-internal use only.
        recompute.maybe_init_layer_mapping(model, proxy, [])
        log_rank0(
            f"[{NAME}] Initialized Chorus recompute relief pressure_ratio={float(_CFG.low_comm_recompute_pressure_ratio)}",
            enable=True,
        )
    except Exception as exc:
        log_rank0(f"[{NAME}] Recompute relief initialization skipped: {exc}", enable=True)

def maybe_init_layer_mapping(model: torch.nn.Module, compile_config, schedule) -> None:
    global _CFG, _LAYER_MAPPING

    should_enable = bool(getattr(compile_config, "global_layer_scheduler", False))
    if not should_enable and schedule is not None:
        for _, passes in schedule:
            for p in passes:
                mod = getattr(p, "__module__", "")
                if mod.endswith(".global_layer_scheduler"):
                    should_enable = True
                    break
            if should_enable:
                break

    if not should_enable:
        return

    default_regexes = (
        r"\.layers\.(\d+)\.",
        r"\.model\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.transformer\.h\.(\d+)\.",
        r"\.decoder\.layers\.(\d+)\.",
    )

    cfg = _SchedulerConfig(
        layer_regexes=tuple(getattr(compile_config, "global_layer_scheduler_layer_regexes", default_regexes)),
        lookahead_blocks=0,
        fuse_max_bytes=int(getattr(compile_config, "global_layer_scheduler_fuse_max_bytes", 256 * 1024 * 1024)),
        fuse_deadline_window_blocks=int(
            getattr(compile_config, "global_layer_scheduler_fuse_deadline_window_blocks", 1)),
        fuse_factor=float(getattr(compile_config, "global_layer_scheduler_fuse_factor", 0.8)),
        inner_refine_max_iters=int(getattr(compile_config, "global_layer_scheduler_inner_refine_max_iters", 512)),
        inner_refine_min_gain_ms=float(getattr(compile_config, "global_layer_scheduler_inner_refine_min_gain_ms", 0.01)),
        max_tasks_per_block=16,
        dump_schedule=bool(getattr(compile_config, "global_layer_scheduler_dump_schedule", False)),
        dump_dir=(str(getattr(compile_config, "global_layer_scheduler_dump_dir", "")).strip() or None),
        milp_time_limit_s=float(getattr(compile_config, "global_layer_scheduler_milp_time_limit_s", 2.0)),
        milp_node_limit=int(getattr(compile_config, "global_layer_scheduler_milp_node_limit", 0)),
        milp_rel_gap=float(getattr(compile_config, "global_layer_scheduler_milp_rel_gap", 0.01)),
        milp_presolve=bool(getattr(compile_config, "global_layer_scheduler_milp_presolve", True)),
        separate_allgather_communicator=bool(getattr(compile_config, "separate_allgather_communicator", True)),
        topology_policy=str(getattr(compile_config, "global_layer_scheduler_topology_policy", "auto")).strip().lower(),
        low_comm_effective_bw_gib_per_s=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_effective_bw_gib_per_s", 120.0)),
        low_comm_pressure_ratio=float(getattr(compile_config, "global_layer_scheduler_low_comm_pressure_ratio", 0.35)),
        low_comm_keep_opportunity_factor=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_keep_opportunity_factor", 0.5)),
        low_comm_keep_cap_fraction=float(getattr(compile_config, "global_layer_scheduler_low_comm_keep_cap_fraction", 0.10)),
        low_comm_persistent_budget_mode=str(
            getattr(compile_config, "global_layer_scheduler_low_comm_persistent_budget_mode", "selective")).strip().lower(),
        low_comm_persistent_usable_fraction=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_persistent_usable_fraction", 0.90)),
        low_comm_persistent_starvation_fraction=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_persistent_starvation_fraction", 0.75)),
        low_comm_graph_rewrite_mode=str(
            getattr(compile_config, "global_layer_scheduler_low_comm_graph_rewrite_mode", "local_prefetch")).strip().lower(),
        low_comm_prefetch_fuse_slack=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_prefetch_fuse_slack", 1.9)),
        low_comm_prefetch_fuse_max_bytes=int(
            getattr(compile_config, "global_layer_scheduler_low_comm_prefetch_fuse_max_bytes", 1850000000)),
        low_comm_prefetch_buffer_max_bytes=int(
            getattr(compile_config, "global_layer_scheduler_low_comm_prefetch_buffer_max_bytes", 5300000000)),
        low_comm_elide_persistent_releases=bool(
            getattr(compile_config, "global_layer_scheduler_low_comm_elide_persistent_releases", True)),
        low_comm_elide_persistent_waits=bool(
            getattr(compile_config, "global_layer_scheduler_low_comm_elide_persistent_waits", True)),
        low_comm_elide_persistent_prefetches=bool(
            getattr(compile_config, "global_layer_scheduler_low_comm_elide_persistent_prefetches", True)),
        low_comm_persistent_value_mode=str(
            getattr(compile_config, "global_layer_scheduler_low_comm_persistent_value_mode", "event_density")).strip().lower(),
        low_comm_comm_value_weight=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_comm_value_weight", 0.25)),
        low_comm_state_op_ms=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_state_op_ms", 0.02)),
        low_comm_recompute_relief=bool(
            getattr(compile_config, "global_layer_scheduler_low_comm_recompute_relief", False)),
        low_comm_recompute_pressure_ratio=float(
            getattr(compile_config, "global_layer_scheduler_low_comm_recompute_pressure_ratio", 0.72)),
        low_comm_post_step_refresh=bool(
            getattr(compile_config, "global_layer_scheduler_low_comm_post_step_refresh", False)),
    )
    _CFG = cfg
    _maybe_init_recompute_relief(model, compile_config, schedule)

    name_and_ds = []
    ds_id_to_layer: Dict[int, int] = {}
    layer_to_ds_ids: Dict[int, List[int]] = {}
    for name, p in model.named_parameters():
        if not hasattr(p, "ds_id"):
            continue
        ds_id = int(p.ds_id)
        layer_idx = _infer_layer_idx(name, cfg.layer_regexes)
        if layer_idx is None:
            continue
        ds_id_to_layer[ds_id] = layer_idx
        layer_to_ds_ids.setdefault(layer_idx, []).append(ds_id)
        name_and_ds.append((name, ds_id, layer_idx))

    mapping_hash = _sha256_hex(_canonical_json(sorted(name_and_ds)).encode("utf-8"))

    if dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [mapping_hash] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        if obj_list[0] != mapping_hash:
            raise RuntimeError(
                f"[{NAME}] param->layer mapping hash mismatch across ranks: local={mapping_hash} rank0={obj_list[0]}")

    if not layer_to_ds_ids:
        log_rank0(f"[{NAME}] No transformer layers detected from parameter names; scheduler disabled.", enable=True)
        _LAYER_MAPPING = None
        return

    L = max(layer_to_ds_ids.keys()) + 1
    layer_to_ds_ids_tuple: List[Tuple[int, ...]] = []
    for i in range(L):
        ids = sorted(set(layer_to_ds_ids.get(i, [])))
        layer_to_ds_ids_tuple.append(tuple(ids))

    _LAYER_MAPPING = _LayerMapping(
        mapping_hash=mapping_hash,
        L=L,
        layer_to_ds_ids=tuple(layer_to_ds_ids_tuple),
        ds_id_to_layer=ds_id_to_layer,
    )

    log_rank0(f"[{NAME}] Initialized layer mapping: L={L} layers, mapped_ds_ids={len(ds_id_to_layer)}", enable=True)


def _last_backward_graph_id(graph_order: List[Tuple[int, bool]]) -> Optional[int]:
    last = None
    for g_id, needs_bwd in graph_order:
        if needs_bwd:
            last = g_id
            break
    return last


def _lookahead_blocks_for_num_layers(L: int) -> int:
    L = int(L)
    return int(L) if int(L) < 8 else min(int(L), 8)


def _max_tasks_per_block_for_num_layers(L: int) -> int:
    L = int(L)
    return int(L) if int(L) < 16 else min(int(L), 16)


def _peak_mem_bytes(profiling_results) -> int:
    peak = 0
    for prof in profiling_results.values():
        if getattr(prof, "fwd_mem", None):
            peak = max(peak, max(m[3] for m in prof.fwd_mem))
        if getattr(prof, "bwd_mem", None):
            peak = max(peak, max(m[3] for m in prof.bwd_mem))
    return int(peak)


def _min_total_mem_bytes_across_ranks() -> int:
    total_mem = int(get_accelerator().total_memory())
    if dist.is_initialized() and dist.get_world_size() > 1:
        device = get_accelerator().device(get_accelerator().current_device())
        if unset_fake_temporarily is None:
            vals = torch.tensor([total_mem], device=device, dtype=torch.int64)
            dist.all_reduce(vals, dist.ReduceOp.MIN)
            total_mem = int(vals[0].item())
        else:
            with unset_fake_temporarily():
                vals = torch.tensor([total_mem], device=device, dtype=torch.int64)
                dist.all_reduce(vals, dist.ReduceOp.MIN)
                total_mem = int(vals[0].item())
    return total_mem


def _estimate_non_torch_mem_bytes() -> int:
    """Best-effort estimate of non-PyTorch (or non-caching-allocator) GPU memory in use.

    This uses `torch.cuda.mem_get_info()` to read device-wide free/total bytes and
    subtracts the current caching allocator reservation.
    """
    try:
        accelerator = get_accelerator()
        if accelerator.device_name() != "cuda":
            return 0
        if not hasattr(torch.cuda, "mem_get_info"):
            return 0
        device_index = int(accelerator.current_device())

        def _read() -> int:
            free, total = torch.cuda.mem_get_info(device_index)
            reserved = accelerator.memory_reserved(device_index) or 0
            return max(0, int(total) - int(free) - int(reserved))

        if unset_fake_temporarily is None:
            return int(_read())
        with unset_fake_temporarily():
            return int(_read())
    except Exception:
        return 0


def _allocator_margin_bytes(total_mem_bytes: Optional[int] = None) -> int:
    explicit_margin = os.getenv("DS_DEEPCOMPILE_GLS_ALLOCATOR_MARGIN_BYTES", "").strip()
    if explicit_margin:
        return max(0, int(explicit_margin))

    total_mem = int(total_mem_bytes) if total_mem_bytes is not None else _min_total_mem_bytes_across_ranks()
    usable_fraction = float(os.getenv("DS_DEEPCOMPILE_GLS_USABLE_MEMORY_FRACTION", "0.9"))
    usable_fraction = max(0.0, min(1.0, usable_fraction))
    return max(0, int(float(total_mem) * (1.0 - usable_fraction)))


def _build_ds_id_to_size_bytes(graph_id: int, profiling_results, param_manager: Dict[int, DSGraphParamManager]) -> Dict[int, int]:
    ds_id_to_size: Dict[int, int] = {}

    for _, pm in param_manager.items():
        for param_name, ds_param in pm.params.items():
            ds_id = int(pm.ds_ids[param_name])
            # Fallback: true numel (unpadded) is good enough for scheduling.
            ds_id_to_size[ds_id] = int(ds_param.param.numel() * ds_param.param.element_size())

    prof = profiling_results[graph_id]
    for g in (getattr(prof, "fwd_graph", None), getattr(prof, "bwd_graph", None)):
        if g is None:
            continue
        for n in g.nodes:
            if n.target != torch.ops.dc.allgather_param.default:
                continue
            ds_id = int(n.args[2])
            # Prefer alloc_mem (actual padded buffer allocation) over tensor_size (logical view size).
            alloc_mem = int(n.meta.get("alloc_mem", 0))
            if alloc_mem > 0:
                ds_id_to_size[ds_id] = max(int(ds_id_to_size.get(ds_id, 0)), alloc_mem)
                continue
            if "tensor_size" in n.meta:
                ds_id_to_size[ds_id] = max(int(ds_id_to_size.get(ds_id, 0)), int(n.meta["tensor_size"]))

    return ds_id_to_size


def _build_ds_id_to_allgather_ms(profiling_results) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for prof in profiling_results.values():
        for g in (getattr(prof, "fwd_graph", None), getattr(prof, "bwd_graph", None)):
            if g is None:
                continue
            for n in g.nodes:
                if n.target == torch.ops.dc.allgather_param.default and "device_time" in n.meta:
                    ds_id = int(n.args[2])
                    out[ds_id] = out.get(ds_id, 0.0) + float(n.meta["device_time"])
    return out


def _build_ds_id_state_op_counts(profiling_results) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for prof in profiling_results.values():
        for g in (getattr(prof, "fwd_graph", None), getattr(prof, "bwd_graph", None)):
            if g is None:
                continue
            for n in g.nodes:
                if n.target == torch.ops.dc.wait_allgather.default:
                    ds_id = int(n.args[2])
                    counts[ds_id] = counts.get(ds_id, 0) + 1
                elif n.target == torch.ops.dc.release_param.default:
                    ds_id = int(n.args[2])
                    counts[ds_id] = counts.get(ds_id, 0) + 1
    return counts


def _rank_persistent_candidates(ds_id_to_size: Dict[int, int], ds_id_to_allgather_ms: Dict[int, float]) -> List[int]:
    def _time_per_byte(ds_id: int) -> float:
        size = int(ds_id_to_size.get(int(ds_id), 0))
        if size <= 0:
            return 0.0
        return float(ds_id_to_allgather_ms.get(int(ds_id), 0.0)) / float(size)

    candidates = sorted(
        [int(ds_id) for ds_id in ds_id_to_size.keys()],
        key=lambda d: (_time_per_byte(int(d)), float(ds_id_to_allgather_ms.get(int(d), 0.0)),
                       int(ds_id_to_size.get(int(d), 0)), int(d)),
        reverse=True,
    )
    return candidates


def _select_persistent_ds_ids(candidates: Sequence[int], ds_id_to_size: Dict[int, int],
                              budget_bytes: int) -> Tuple[List[int], int]:
    budget = max(0, int(budget_bytes))
    persistent_ds_ids: List[int] = []
    persistent_mem_bytes = 0
    if budget <= 0:
        return persistent_ds_ids, 0
    for ds_id in candidates:
        ds_id = int(ds_id)
        size = int(ds_id_to_size.get(int(ds_id), 0))
        if size <= 0:
            continue
        if persistent_mem_bytes + size > budget:
            continue
        persistent_ds_ids.append(int(ds_id))
        persistent_mem_bytes += int(size)
    return persistent_ds_ids, int(persistent_mem_bytes)


def _select_persistent_ds_ids_knapsack(candidates: Sequence[int],
                                       ds_id_to_size: Dict[int, int],
                                       ds_id_to_value: Dict[int, float],
                                       budget_bytes: int,
                                       *,
                                       block_increments: Optional[Dict[int, Tuple[int, ...]]] = None,
                                       base_reserved: Optional[Sequence[int]] = None,
                                       cap_bytes: Optional[Sequence[int]] = None) -> Tuple[List[int], int, float, dict]:
    budget = max(0, int(budget_bytes))
    if budget <= 0:
        return [], 0, 0.0, {"method": "none", "status": "empty_budget"}

    items: List[Tuple[int, int, float, Tuple[int, ...]]] = []
    cap_list = [int(x) for x in cap_bytes] if cap_bytes is not None else []
    base_reserved_list = [int(x) for x in base_reserved] if base_reserved is not None else []
    use_block_caps = bool(block_increments) and bool(cap_list) and len(cap_list) == len(base_reserved_list)

    for ds_id in candidates:
        ds_id = int(ds_id)
        size = int(ds_id_to_size.get(int(ds_id), 0))
        if size <= 0 or size > budget:
            continue
        incr_blocks = tuple(block_increments.get(int(ds_id), ())) if use_block_caps and block_increments is not None else ()
        individually_feasible = True
        for k in incr_blocks:
            if int(k) < 0 or int(k) >= len(cap_list):
                continue
            if int(base_reserved_list[int(k)]) + int(size) > int(cap_list[int(k)]):
                individually_feasible = False
                break
        if not individually_feasible:
            continue
        items.append((int(ds_id), int(size), max(0.0, float(ds_id_to_value.get(int(ds_id), 0.0))), incr_blocks))

    if not items:
        return [], 0, 0.0, {"method": "none", "status": "no_candidates"}

    if not any(float(value) > 0.0 for _, _, value, _ in items):
        return [], 0, 0.0, {"method": "none", "status": "no_positive_value"}

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except Exception as exc:
        greedy_selected: List[int] = []
        greedy_mem = 0
        greedy_value = 0.0
        extra_reserved = [0 for _ in range(len(cap_list))]
        for ds_id, size, value, incr_blocks in items:
            if greedy_mem + int(size) > budget:
                continue
            feasible = True
            for k in incr_blocks:
                if int(k) < 0 or int(k) >= len(cap_list):
                    continue
                if int(base_reserved_list[int(k)]) + int(extra_reserved[int(k)]) + int(size) > int(cap_list[int(k)]):
                    feasible = False
                    break
            if not feasible:
                continue
            greedy_selected.append(int(ds_id))
            greedy_mem += int(size)
            greedy_value += float(value)
            for k in incr_blocks:
                if 0 <= int(k) < len(extra_reserved):
                    extra_reserved[int(k)] += int(size)
        return greedy_selected, int(greedy_mem), float(greedy_value), {
            "method": "greedy_fallback",
            "status": f"import_error: {exc}",
        }

    n_items = len(items)
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    ub_rows: List[float] = []
    row = 0

    def add_row(coeffs: Sequence[Tuple[int, float]], rhs: float) -> None:
        nonlocal row
        for j, v in coeffs:
            if float(v) == 0.0:
                continue
            rows.append(int(row))
            cols.append(int(j))
            data.append(float(v))
        ub_rows.append(float(rhs))
        row += 1

    add_row([(idx, float(size)) for idx, (_, size, _, _) in enumerate(items)], float(budget))

    if use_block_caps:
        for k in range(len(cap_list)):
            coeffs = [(idx, float(size)) for idx, (_, size, _, incr_blocks) in enumerate(items) if int(k) in incr_blocks]
            if not coeffs:
                continue
            rhs = float(int(cap_list[int(k)]) - int(base_reserved_list[int(k)]))
            if rhs < 0.0:
                return [], 0, 0.0, {"method": "none", "status": f"infeasible_block_{k}"}
            add_row(coeffs, rhs)

    A = coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                   shape=(row, n_items)).tocsr()
    constraint = LinearConstraint(A, -np.inf * np.ones(row, dtype=float), np.array(ub_rows, dtype=float))

    c = np.array(
        [-(float(value) + 1e-9 * float(n_items - idx)) for idx, (_, _, value, _) in enumerate(items)],
        dtype=float,
    )
    integrality = np.ones(n_items, dtype=int)
    lb = np.zeros(n_items, dtype=float)
    ub = np.ones(n_items, dtype=float)

    milp_time_limit_s = 2.0
    milp_rel_gap = 0.01
    if _CFG is not None:
        try:
            milp_time_limit_s = float(getattr(_CFG, "milp_time_limit_s", milp_time_limit_s))
            milp_rel_gap = float(getattr(_CFG, "milp_rel_gap", milp_rel_gap))
        except Exception:
            pass

    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=constraint,
        options={
            "disp": False,
            "time_limit": float(milp_time_limit_s),
            "mip_rel_gap": float(max(0.0, milp_rel_gap)),
        },
    )

    x = getattr(res, "x", None)
    if x is None:
        return [], 0, 0.0, {
            "method": "milp",
            "status": f"no_solution: {getattr(res, 'message', 'unknown')}",
            "solver_status": int(getattr(res, "status", -1)),
        }

    chosen_indices = [idx for idx, value in enumerate(x) if float(value) >= 0.5]
    selected = [int(items[idx][0]) for idx in chosen_indices]
    selected_mem = int(sum(int(items[idx][1]) for idx in chosen_indices))
    selected_value = float(sum(float(items[idx][2]) for idx in chosen_indices))
    return selected, int(selected_mem), float(selected_value), {
        "method": "milp",
        "status": str(getattr(res, "message", "ok")),
        "solver_status": int(getattr(res, "status", 0)),
        "mip_gap": float(getattr(res, "mip_gap", float("nan"))) if hasattr(res, "mip_gap") else float("nan"),
        "objective": float(-res.fun) if getattr(res, "fun", None) is not None else None,
        "num_items": int(n_items),
    }


def _persistent_increment_blocks_for_ds_id(ds_id: int, sched: dict, ds_id_to_layer: Dict[int, int]) -> Tuple[int, ...]:
    layer = ds_id_to_layer.get(int(ds_id))
    if layer is None:
        return ()

    L = int(sched.get("L", 0))
    K = 2 * L
    if L <= 0:
        return ()

    per_task_start = {str(k): int(v) for k, v in sched.get("per_task_start_k", {}).items()}
    start_f = per_task_start.get(f"layer.{int(layer)}.FWD")
    if start_f is None:
        return ()

    f_deadline = int(layer)
    b_deadline = int(K - 1 - int(layer))
    keep_layers = set(map(int, sched.get("keep_layers", [])))

    if int(layer) in keep_layers:
        extra_start = int(b_deadline + 1)
        if extra_start > int(K - 1):
            return ()
        return tuple(range(extra_start, K))

    start_b = per_task_start.get(f"layer.{int(layer)}.BWD")
    blocks: List[int] = []
    if start_b is None:
        if int(f_deadline + 1) <= int(K - 1):
            blocks.extend(range(int(f_deadline + 1), K))
        return tuple(blocks)

    if int(f_deadline + 1) <= int(start_b - 1):
        blocks.extend(range(int(f_deadline + 1), int(start_b)))
    if int(b_deadline + 1) <= int(K - 1):
        blocks.extend(range(int(b_deadline + 1), K))
    return tuple(blocks)


def _select_persistent_ds_ids_block_safe(candidates: Sequence[int], ds_id_to_size: Dict[int, int], budget_bytes: int,
                                         sched: dict, ds_id_to_layer: Dict[int, int],
                                         ds_id_to_value: Optional[Dict[int, float]] = None) -> Tuple[List[int], int, float, dict]:
    budget = max(0, int(budget_bytes))
    if budget <= 0:
        return [], 0, 0.0, {"method": "none", "status": "empty_budget"}

    base_reserved = [int(x) for x in sched.get("predicted_reserved_mem", [])]
    block_profiles = sched.get("block_profiles", {})
    cap_bytes = [int(x) for x in block_profiles.get("cap_bytes", [])]
    values = {int(k): max(0.0, float(v)) for k, v in (ds_id_to_value or {}).items()}
    if not base_reserved or len(base_reserved) != len(cap_bytes) or not ds_id_to_layer:
        selected, mem, value, meta = _select_persistent_ds_ids_knapsack(candidates,
                                                                        ds_id_to_size,
                                                                        values,
                                                                        budget)
        return selected, int(mem), float(value), meta

    block_increments = {
        int(ds_id): _persistent_increment_blocks_for_ds_id(int(ds_id), sched, ds_id_to_layer) for ds_id in candidates
    }
    selected, mem, value, meta = _select_persistent_ds_ids_knapsack(candidates,
                                                                    ds_id_to_size,
                                                                    values,
                                                                    budget,
                                                                    block_increments=block_increments,
                                                                    base_reserved=base_reserved,
                                                                    cap_bytes=cap_bytes)
    return selected, int(mem), float(value), meta


def _predicted_peak_mem_bytes_from_schedule(sched: dict,
                                            ds_id_to_size: Optional[Dict[int, int]] = None,
                                            ds_id_to_layer: Optional[Dict[int, int]] = None) -> int:
    reserved = [int(x) for x in sched.get("predicted_reserved_mem", [])]
    floors = [int(x) for x in sched.get("block_profiles", {}).get("non_param_floor_bytes", [])]
    if not reserved or not floors:
        return 0

    K = min(len(reserved), len(floors))
    persistent_extra = [0 for _ in range(K)]
    if ds_id_to_size and ds_id_to_layer:
        for ds_id in sched.get("persistent_ds_ids", []):
            ds_id = int(ds_id)
            size = int(ds_id_to_size.get(ds_id, 0))
            if size <= 0:
                continue
            for k in _persistent_increment_blocks_for_ds_id(ds_id, sched, ds_id_to_layer):
                if 0 <= int(k) < K:
                    persistent_extra[int(k)] += size

    return max((int(reserved[i]) + int(floors[i]) + int(persistent_extra[i]) for i in range(K)), default=0)


def _build_ds_id_to_wait_pos(graph: Optional[Graph]) -> Dict[int, int]:
    # Approximate first-use order for each ds_id using wait_allgather positions (fallback to allgather positions).
    out: Dict[int, int] = {}
    if graph is None:
        return out

    for idx, n in enumerate(graph.nodes):
        if n.target == torch.ops.dc.wait_allgather.default:
            ds_id = _get_ds_id_from_wait(n)
            prev = out.get(ds_id)
            out[ds_id] = int(idx) if prev is None else min(int(prev), int(idx))
        elif n.target == torch.ops.dc.allgather_param.default:
            # Fallback for graphs that have no explicit wait node for a ds_id (rare).
            ds_id = int(n.args[2])
            out.setdefault(int(ds_id), int(idx))
    return out


@dataclass
class _Task:
    kind: str  # "FWD" or "BWD"
    layer_idx: int
    ds_ids: Tuple[int, ...]
    size_bytes: int
    min_start_k: int
    deadline_k: int
    release_end_k: int
    start_k: Optional[int] = None


@dataclass(frozen=True)
class _BlockProfile:
    k: int
    phase: str  # "FWD" or "BWD"
    layer_idx: int
    compute_ms: float
    cap_bytes: int
    explicit_temp_reserve_bytes: int
    non_param_floor_bytes: int
    fixed_comm_start_offset_ms: float
    fixed_comm_ms: float


@dataclass(frozen=True)
class _CommModel:
    alpha_ms: float
    beta_ms_per_byte: float

    def predict_ms(self, size_bytes: int) -> float:
        if size_bytes <= 0:
            return 0.0
        return max(0.0, float(self.alpha_ms) + float(self.beta_ms_per_byte) * float(size_bytes))


@dataclass(frozen=True)
class _CommGroup:
    start_k: int
    deadline_k: int
    ds_ids: Tuple[int, ...]
    # (task_key, ready_offset_ms) where ready_offset_ms is measured from the group's start time.
    task_ready_offsets_ms: Tuple[Tuple[Tuple[int, str], float], ...]
    total_bytes: int
    comm_ms: float


def _layer_size_and_ops(L: int, layer_to_ds_ids: Sequence[Sequence[int]],
                        ds_id_to_size: Dict[int, int]) -> Tuple[List[int], List[int]]:
    layer_sizes: List[int] = []
    layer_ops: List[int] = []
    for li in range(int(L)):
        ids = [int(ds_id) for ds_id in layer_to_ds_ids[int(li)] if int(ds_id) in ds_id_to_size]
        ids = sorted(set(ids))
        layer_sizes.append(int(sum(int(ds_id_to_size[int(ds_id)]) for ds_id in ids)))
        layer_ops.append(int(len(ids)))
    return layer_sizes, layer_ops


def _comm_coeffs_for_layers(layer_sizes: Sequence[int], layer_ops: Sequence[int], cfg: _SchedulerConfig,
                            comm_model: _CommModel) -> List[float]:
    alpha_ms = float(comm_model.alpha_ms)
    alpha_per_op_fused = float(alpha_ms) * max(0.0, min(1.0, float(cfg.fuse_factor)))
    beta = float(comm_model.beta_ms_per_byte)
    coeffs: List[float] = []
    for size, ops in zip(layer_sizes, layer_ops):
        alpha_eff = float(alpha_per_op_fused) if int(ops) > 1 else float(alpha_ms)
        coeffs.append(float(alpha_eff) * float(ops) + float(beta) * float(size))
    return coeffs


def _effective_bw_gib_per_s(comm_model: _CommModel) -> float:
    beta = float(comm_model.beta_ms_per_byte)
    if beta <= 0.0:
        return float("inf")
    bytes_per_ms = 1.0 / beta
    return float(bytes_per_ms * 1000.0 / float(1024**3))


def _resolve_comm_backend_policy(cfg: _SchedulerConfig, blocks: Sequence[_BlockProfile], layer_sizes: Sequence[int],
                                 layer_ops: Sequence[int], comm_model: _CommModel) -> dict:
    requested = str(getattr(cfg, "topology_policy", "auto") or "auto").strip().lower()
    if requested in {"nvlink", "low_comm", "low-comm"}:
        mode = "low_comm"
    elif requested in {"generic", "default", "none", "off"}:
        mode = "generic"
    else:
        comm_coeffs = _comm_coeffs_for_layers(layer_sizes, layer_ops, cfg, comm_model)
        compute_ms = float(sum(float(b.compute_ms) for b in blocks))
        # Count FWD and BWD param all-gathers. This is raw communication pressure, not exposed stall.
        param_comm_ms = 2.0 * float(sum(comm_coeffs))
        pressure = float(param_comm_ms / max(compute_ms, 1e-9))
        bw_gib = _effective_bw_gib_per_s(comm_model)
        low_by_bw = bool(math.isfinite(float(bw_gib)) and bw_gib >= float(cfg.low_comm_effective_bw_gib_per_s))
        low_by_pressure = bool(pressure <= float(cfg.low_comm_pressure_ratio))
        mode = "low_comm" if bool(cfg.separate_allgather_communicator) and (low_by_bw or low_by_pressure) else "generic"
        return {
            "requested": requested or "auto",
            "mode": mode,
            "effective_bw_gib_per_s": float(bw_gib),
            "param_comm_ms": float(param_comm_ms),
            "compute_ms": float(compute_ms),
            "comm_pressure_ratio": float(pressure),
            "low_by_bw": bool(low_by_bw),
            "low_by_pressure": bool(low_by_pressure),
            "separate_allgather_communicator": bool(cfg.separate_allgather_communicator),
        }

    comm_coeffs = _comm_coeffs_for_layers(layer_sizes, layer_ops, cfg, comm_model)
    compute_ms = float(sum(float(b.compute_ms) for b in blocks))
    param_comm_ms = 2.0 * float(sum(comm_coeffs))
    pressure = float(param_comm_ms / max(compute_ms, 1e-9))
    return {
        "requested": requested or "auto",
        "mode": mode,
        "effective_bw_gib_per_s": float(_effective_bw_gib_per_s(comm_model)),
        "param_comm_ms": float(param_comm_ms),
        "compute_ms": float(compute_ms),
        "comm_pressure_ratio": float(pressure),
        "low_by_bw": bool(mode == "low_comm"),
        "low_by_pressure": bool(mode == "low_comm"),
        "separate_allgather_communicator": bool(cfg.separate_allgather_communicator),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0)
    if not vals:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(vals) - 1) * q))
    return float(vals[idx])


def _build_keep_opportunity_penalties(*,
                                      L: int,
                                      layer_to_ds_ids: Sequence[Sequence[int]],
                                      layer_sizes: Sequence[int],
                                      ds_id_to_size: Dict[int, int],
                                      ds_id_to_allgather_ms: Dict[int, float],
                                      cfg: _SchedulerConfig,
                                      backend_policy: dict) -> Dict[int, float]:
    if str(backend_policy.get("mode", "generic")) != "low_comm":
        return {}

    factor = max(0.0, float(cfg.low_comm_keep_opportunity_factor))
    if factor <= 0.0:
        return {}

    densities = []
    for ds_id, size in ds_id_to_size.items():
        size = int(size)
        if size <= 0:
            continue
        value = max(0.0, float(ds_id_to_allgather_ms.get(int(ds_id), 0.0)))
        if value > 0.0:
            densities.append(float(value) / float(size))
    ref_density = _percentile(densities, 0.75)

    pressure = max(1e-9, float(backend_policy.get("comm_pressure_ratio", 1.0)))
    threshold = max(1e-9, float(cfg.low_comm_pressure_ratio))
    pressure_scale = max(1.0, min(2.0, threshold / pressure))

    penalties: Dict[int, float] = {}
    K = max(1, 2 * int(L))
    for li in range(int(L)):
        size = int(layer_sizes[int(li)]) if int(li) < len(layer_sizes) else 0
        if size <= 0:
            continue
        layer_value = 0.0
        for ds_id in layer_to_ds_ids[int(li)]:
            layer_value += max(0.0, float(ds_id_to_allgather_ms.get(int(ds_id), 0.0)))
        ref_value = float(ref_density) * float(size) if ref_density > 0.0 else 0.0
        opportunity_ms = max(float(layer_value), float(ref_value))
        if opportunity_ms <= 0.0:
            continue
        f_deadline = int(li)
        b_deadline = int(2 * int(L) - 1 - int(li))
        live_span = max(0, int(b_deadline) - int(f_deadline))
        live_span_scale = 0.5 + float(live_span) / float(K)
        penalties[int(li)] = float(factor) * float(pressure_scale) * float(live_span_scale) * float(opportunity_ms)
    return penalties


def _is_param_comm_node(n: Node) -> bool:
    return n.target in {
        torch.ops.dc.allgather_param.default,
        torch.ops.dc.wait_allgather.default,
        torch.ops.dc.release_param.default,
        torch.ops.dc.prefetch_params_fused.default,
    }


def _is_comm_node(n: Node) -> bool:
    return bool(getattr(n, "meta", {}).get("comm", False))


def _is_fixed_comm_marker_node(n: Node) -> bool:
    # Comm nodes that are not part of param-prefetch scheduling (e.g., grad reduction).
    return _is_comm_node(n) and not _is_param_comm_node(n)


def _node_explicit_temp_reserve_bytes(n: Node) -> int:
    try:
        return max(0, int(getattr(n, "meta", {}).get("explicit_temp_reserve_bytes", 0)))
    except Exception:
        return 0


def _build_mem_peak_by_node_name(mem_list, graph: Graph) -> Dict[str, int]:
    # mem_list item: (node_name, current_alloc, delta, peak)
    peak_by_name: Dict[str, int] = {name: int(peak) for name, _, _, peak in mem_list}

    # Fill missing values to make slicing robust under control flow.
    prev_peak = 0
    for n in graph.nodes:
        if n.name in peak_by_name:
            prev_peak = peak_by_name[n.name]
        else:
            peak_by_name[n.name] = prev_peak
    return peak_by_name


def _live_param_bytes_by_node(graph: Graph, ds_id_to_size: Dict[int, int]) -> Dict[Node, int]:
    # Track when gathered param buffers become live and when they are freed by the last release.
    ds_id_to_release_total: Dict[int, int] = {}
    for n in graph.nodes:
        if n.target == torch.ops.dc.release_param.default:
            ds_id = int(n.args[2])
            n_users = int(n.args[3]) if len(n.args) >= 4 else 0
            if n_users > 0:
                ds_id_to_release_total.setdefault(ds_id, n_users)

    ds_id_release_seen: Dict[int, int] = {}
    live_ds_ids: Set[int] = set()
    live_bytes = 0
    out: Dict[Node, int] = {}

    for n in graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default:
            ds_id = int(n.args[2])
            if ds_id in ds_id_to_size and ds_id not in live_ds_ids:
                live_ds_ids.add(ds_id)
                live_bytes += int(ds_id_to_size[ds_id])

        out[n] = live_bytes

        if n.target == torch.ops.dc.release_param.default:
            ds_id = int(n.args[2])
            total = ds_id_to_release_total.get(ds_id, 0)
            if total <= 0:
                continue
            ds_id_release_seen[ds_id] = ds_id_release_seen.get(ds_id, 0) + 1
            if ds_id in live_ds_ids and ds_id_release_seen[ds_id] >= total:
                live_ds_ids.remove(ds_id)
                live_bytes -= int(ds_id_to_size.get(ds_id, 0))

    return out


def _extract_blocks_for_graph(graph: Graph, mem_list, L: int, ds_id_to_layer: Dict[int, int],
                              ds_id_to_size: Dict[int, int], total_mem_bytes: int, allocator_margin_bytes: int,
                              bwd: bool) -> List[_BlockProfile]:
    anchors = _find_layer_anchors(graph, ds_id_to_layer, L)
    if len(anchors) == 0:
        return []

    nodes = list(graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    peak_by_name = _build_mem_peak_by_node_name(mem_list, graph)
    live_by_node = _live_param_bytes_by_node(graph, ds_id_to_size)

    safe_total = int(total_mem_bytes)
    allocator_margin_bytes = max(0, int(allocator_margin_bytes))
    blocks: List[_BlockProfile] = []

    layer_order = list(range(L)) if not bwd else list(reversed(range(L)))
    for idx_in_phase, layer in enumerate(layer_order):
        if layer not in anchors:
            continue
        # Include any prefix nodes before the first layer anchor in the first block to avoid
        # dropping "pre-layer" compute (e.g., embeddings / loss prologue) from the timeline.
        start = 0 if not blocks else pos[anchors[layer]]
        if idx_in_phase + 1 < len(layer_order):
            # Find next anchor in execution order (skip missing layers).
            next_start = None
            for nxt in layer_order[idx_in_phase + 1:]:
                if nxt in anchors:
                    next_start = pos[anchors[nxt]]
                    break
            end = next_start if next_start is not None else len(nodes)
        else:
            end = len(nodes)

        compute_ms = 0.0
        fixed_comm_start_offset_ms = 0.0
        saw_fixed_comm_marker = False
        base_mem_excl_params = 0
        explicit_temp_reserve_bytes = 0
        for n in nodes[start:end]:
            if not _is_param_comm_node(n) and not _is_fixed_comm_marker_node(n) and n.target != torch.ops.dc.end_backward.default:
                compute_ms += float(n.meta.get("device_time", 0.0))
            if not saw_fixed_comm_marker and _is_fixed_comm_marker_node(n):
                fixed_comm_start_offset_ms = float(compute_ms)
                saw_fixed_comm_marker = True
            peak = int(peak_by_name.get(n.name, 0))
            live = int(live_by_node.get(n, 0))
            base_mem_excl_params = max(base_mem_excl_params, max(0, peak - live))
            explicit_temp_reserve_bytes = max(explicit_temp_reserve_bytes, _node_explicit_temp_reserve_bytes(n))

        non_param_floor_bytes = max(int(base_mem_excl_params), int(explicit_temp_reserve_bytes))
        cap_bytes = max(0, safe_total - non_param_floor_bytes - allocator_margin_bytes)
        unified_k = layer if not bwd else L + idx_in_phase
        blocks.append(
            _BlockProfile(
                k=int(unified_k),
                phase="BWD" if bwd else "FWD",
                layer_idx=int(layer),
                compute_ms=float(compute_ms),
                cap_bytes=int(cap_bytes),
                explicit_temp_reserve_bytes=int(explicit_temp_reserve_bytes),
                non_param_floor_bytes=int(non_param_floor_bytes),
                fixed_comm_start_offset_ms=float(fixed_comm_start_offset_ms),
                fixed_comm_ms=0.0,
            ))

    blocks.sort(key=lambda b: b.k)
    return blocks


def _fit_comm_model(graph_id: int, profiling_results, ds_id_to_size: Dict[int, int]) -> _CommModel:
    prof = profiling_results[graph_id]
    xs: List[float] = []
    ys: List[float] = []
    for g in (getattr(prof, "fwd_graph", None), getattr(prof, "bwd_graph", None)):
        if g is None:
            continue
        for n in g.nodes:
            # allgather_param is a better comm-cost signal than wait_allgather because waits can be fully hidden.
            if n.target != torch.ops.dc.allgather_param.default:
                continue
            ds_id = int(n.args[2])
            if ds_id not in ds_id_to_size:
                continue
            t_ms = float(n.meta.get("device_time", 0.0))
            if t_ms <= 0.0:
                continue
            xs.append(float(ds_id_to_size[ds_id]))
            ys.append(t_ms)

    if len(xs) < 2:
        # Fallback: assume 0.0 alpha and 200 GB/s effective bandwidth.
        bw_bytes_per_ms = (200.0 * (1024**3)) / 1000.0
        return _CommModel(alpha_ms=0.0, beta_ms_per_byte=1.0 / bw_bytes_per_ms)

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    var_x = sum((x - x_mean)**2 for x in xs)
    if var_x <= 0.0:
        return _CommModel(alpha_ms=max(0.0, y_mean), beta_ms_per_byte=0.0)
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    return _CommModel(alpha_ms=max(0.0, alpha), beta_ms_per_byte=max(0.0, beta))


def _build_comm_groups(tasks: Sequence[_Task],
                       comm_model: _CommModel,
                       ds_id_to_size: Dict[int, int],
                       cfg: _SchedulerConfig,
                       *,
                       fuse_deadline_window_blocks: Optional[int] = None) -> Tuple[Dict[int, List[List[int]]], Dict[int, List[_CommGroup]]]:
    # launches[start_k] -> list[list[ds_id]] (one list per fused prefetch call)
    tasks_by_start: Dict[int, List[_Task]] = {}
    for t in tasks:
        if t.start_k is None:
            continue
        tasks_by_start.setdefault(int(t.start_k), []).append(t)

    launches: Dict[int, List[List[int]]] = {}
    groups_by_start: Dict[int, List[_CommGroup]] = {}

    # dc.prefetch_params_fused can respect per-ds_id readiness (each ds_id becomes ready after its own
    # allgather completes, in ds_id list order). This enables deadline-aware fusion within a boundary
    # as long as we order ds_ids by increasing deadline / use position.
    fuse_window = int(cfg.fuse_deadline_window_blocks if fuse_deadline_window_blocks is None else fuse_deadline_window_blocks)
    fuse_window = max(0, int(fuse_window))

    for start_k, ts in tasks_by_start.items():
        ts = sorted(ts, key=lambda t: (t.deadline_k, 0 if t.kind == "FWD" else 1, t.layer_idx))

        cur_tasks: List[_Task] = []
        cur_bytes = 0
        cur_deadline: Optional[int] = None

        def flush():
            nonlocal cur_tasks, cur_bytes, cur_deadline
            if not cur_tasks:
                return
            # Keep ds_ids in a deterministic, deadline-aware order to minimize tail stalls:
            # comm ops launched earlier within a fused call tend to complete earlier on the comm stream.
            seen: Set[int] = set()
            ds_ids_ordered: List[int] = []
            task_ds_ids: List[Tuple[Tuple[int, str], List[int]]] = []
            for t in cur_tasks:
                key = (int(t.layer_idx), str(t.kind))
                ds_ids_for_task: List[int] = []
                for ds_id in t.ds_ids:
                    ds_id = int(ds_id)
                    if ds_id in seen:
                        continue
                    seen.add(ds_id)
                    ds_ids_ordered.append(ds_id)
                    ds_ids_for_task.append(ds_id)
                task_ds_ids.append((key, ds_ids_for_task))

            ds_ids_tuple = tuple(ds_ids_ordered)
            deadline_k = min(int(t.deadline_k) for t in cur_tasks)
            total_bytes = int(sum(int(ds_id_to_size.get(int(ds_id), 0)) for ds_id in ds_ids_tuple))

            n_ops = len(ds_ids_tuple)
            fuse_alpha_factor = 1.0
            if n_ops > 1:
                fuse_alpha_factor = max(0.0, min(1.0, float(cfg.fuse_factor)))
            alpha_per_op = float(comm_model.alpha_ms) * float(fuse_alpha_factor)
            beta = float(comm_model.beta_ms_per_byte)

            # Model per-task readiness within a fused prefetch: a task becomes ready when the last of its ds_ids
            # completes (ds_ids earlier in the fused list complete earlier).
            ready_offsets: List[Tuple[Tuple[int, str], float]] = []
            offset_ms = 0.0
            for key, ds_ids_for_task in task_ds_ids:
                for ds_id in ds_ids_for_task:
                    offset_ms += float(alpha_per_op) + float(beta) * float(ds_id_to_size.get(int(ds_id), 0))
                ready_offsets.append((key, float(offset_ms)))
            comm_ms = float(offset_ms)

            groups_by_start.setdefault(int(start_k), []).append(
                _CommGroup(
                    start_k=int(start_k),
                    deadline_k=int(deadline_k),
                    ds_ids=ds_ids_tuple,
                    task_ready_offsets_ms=tuple(ready_offsets),
                    total_bytes=int(total_bytes),
                    comm_ms=float(comm_ms),
                ))
            cur_tasks = []
            cur_bytes = 0
            cur_deadline = None

        for t in ts:
            if not cur_tasks:
                cur_tasks = [t]
                cur_bytes = int(t.size_bytes)
                cur_deadline = int(t.deadline_k)
                continue
            within_deadline_window = (cur_deadline is not None) and (
                int(t.deadline_k) <= int(cur_deadline) + int(fuse_window))
            within_size = cur_bytes + int(t.size_bytes) <= int(cfg.fuse_max_bytes)
            if within_deadline_window and within_size:
                cur_tasks.append(t)
                cur_bytes += int(t.size_bytes)
            else:
                flush()
                cur_tasks = [t]
                cur_bytes = int(t.size_bytes)
                cur_deadline = int(t.deadline_k)
        flush()

        groups_by_start[int(start_k)].sort(key=lambda g: (g.deadline_k, g.ds_ids))
        launches[int(start_k)] = [list(g.ds_ids) for g in groups_by_start[int(start_k)]]

    return launches, groups_by_start


def _required_task_key_for_block(block: _BlockProfile, keep_layers: Set[int]) -> Tuple[int, str]:
    layer = int(block.layer_idx)
    if block.phase == "FWD":
        return (layer, "FWD")
    return (layer, "FWD") if layer in keep_layers else (layer, "BWD")


def _simulate_schedule_in_graph_order(
        blocks: Sequence[_BlockProfile],
        groups_by_start: Dict[int, List[_CommGroup]],
        keep_layers: Set[int],
        *,
        separate_allgather_communicator: bool = False,
) -> Tuple[float, List[float], Dict[int, float], Dict[Tuple[int, str], float]]:
    K = len(blocks)
    ready_time: Dict[Tuple[int, str], float] = {}
    stall_ms_per_block = [0.0 for _ in range(K)]
    stall_bwd_per_layer: Dict[int, float] = {}

    # Model the *actual* execution order: comm groups are launched in (start_k, list-order).
    now = 0.0  # compute timeline
    ag_free = 0.0  # allgather/prefetch communicator stream availability time
    rs_free = 0.0  # reduce communicator stream availability time (grad reductions)
    separate = bool(separate_allgather_communicator)

    for k in range(K):
        # Launch groups scheduled at this block boundary.
        for g in groups_by_start.get(k, []):
            start_t = max(float(ag_free), float(now))
            comm_end = start_t + float(g.comm_ms)
            ag_free = comm_end
            for key, offset in g.task_ready_offsets_ms:
                # Earliest completion wins (should be unique per task in practice).
                end_t = start_t + float(offset)
                prev = ready_time.get(key)
                ready_time[key] = end_t if prev is None else min(float(prev), float(end_t))

        required = _required_task_key_for_block(blocks[k], keep_layers)
        t0 = float(now)
        req_ready = float(ready_time.get(required, math.inf))
        if req_ready > now:
            now = req_ready
        stall = max(0.0, now - t0)
        stall_ms_per_block[k] = stall
        if blocks[k].phase == "BWD" and stall > 0.0:
            layer = int(blocks[k].layer_idx)
            stall_bwd_per_layer[layer] = stall_bwd_per_layer.get(layer, 0.0) + stall

        # Execute compute for this block, then schedule any fixed comm that starts within the block.
        compute_ms = float(blocks[k].compute_ms)
        offset_ms = float(blocks[k].fixed_comm_start_offset_ms)
        fixed_comm_ms = float(blocks[k].fixed_comm_ms)

        pre_ms = min(compute_ms, max(0.0, offset_ms))
        now += pre_ms

        if fixed_comm_ms > 0.0:
            if separate:
                fixed_start = max(float(rs_free), float(now))
                rs_free = fixed_start + fixed_comm_ms
            else:
                fixed_start = max(float(ag_free), float(now))
                ag_free = fixed_start + fixed_comm_ms

        now += max(0.0, compute_ms - pre_ms)

    # end_backward synchronizes comm before the step finishes.
    now = max(float(now), float(rs_free if separate else ag_free))

    return float(now), stall_ms_per_block, stall_bwd_per_layer, ready_time


def _reserved_mem_by_block(tasks: Sequence[_Task], K: int) -> List[int]:
    reserved = [0 for _ in range(K)]
    for t in tasks:
        if t.start_k is None:
            continue
        s = int(t.start_k)
        e = int(t.release_end_k)
        if s > e:
            continue
        for k in range(max(0, s), min(K - 1, e) + 1):
            reserved[k] += int(t.size_bytes)
    return reserved


def _solve_milp_schedule(
    *,
    L: int,
    blocks: Sequence[_BlockProfile],
    layer_to_ds_ids: Sequence[Sequence[int]],
    ds_id_to_size: Dict[int, int],
    cfg: _SchedulerConfig,
    comm_model: _CommModel,
    forced_keep_layers: Optional[Set[int]] = None,
    keep_layer_penalty_ms: Optional[Dict[int, float]] = None,
    max_keep_layers: Optional[int] = None,
) -> Tuple[Set[int], Dict[Tuple[int, str], int], dict]:
    # MILP model (HiGHS via SciPy) chooses:
    # - keep_layers (step-level keep-unshard)
    # - per-layer task start blocks for FWD and (if not kept) BWD prefetch
    #
    # Fused group decisions are derived later from the chosen starts.
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except Exception as e:
        raise RuntimeError(
            f"[{NAME}] MILP solver requires SciPy (see requirements/requirements-deepcompile.txt). Import error: {e}"
        ) from e

    K = len(blocks)
    if K != 2 * L:
        raise ValueError(f"[{NAME}] internal error: expected K=2L blocks, got K={K} L={L}")

    # Per-layer param volume (bytes) and op count (ds_id count) used by the comm model.
    layer_sizes, layer_ops = _layer_size_and_ops(L, layer_to_ds_ids, ds_id_to_size)

    active_layers = [li for li in range(L) if layer_ops[li] > 0 and layer_sizes[li] > 0]
    keep_penalties = {int(k): max(0.0, float(v)) for k, v in (keep_layer_penalty_ms or {}).items()}

    fuse_alpha_factor = max(0.0, min(1.0, float(cfg.fuse_factor)))
    # Fused prefetch only reduces launch/latency overhead when there are multiple ops in the call.
    # For layers with a single ds_id, use the unfused alpha to avoid underestimating comm cost.
    alpha_ms = float(comm_model.alpha_ms)
    alpha_per_op_fused = float(alpha_ms) * float(fuse_alpha_factor)
    beta = float(comm_model.beta_ms_per_byte)
    beta_grad = beta

    # Upper bound for big-M on time variables (ms).
    sum_compute_ms = float(sum(float(b.compute_ms) for b in blocks))
    per_layer_comm_ms = []
    for li in range(L):
        alpha_eff = float(alpha_per_op_fused) if int(layer_ops[li]) > 1 else float(alpha_ms)
        per_layer_comm_ms.append(float(alpha_eff) * float(layer_ops[li]) + float(beta) * float(layer_sizes[li]))
    per_layer_grad_comm_ms = [float(beta_grad) * float(layer_sizes[li]) for li in range(L)]
    # Worst-case: gather twice per layer (FWD + BWD), plus a margin.
    # Also include the fixed grad-comm tail so time bounds remain safe.
    M_time = max(1.0, sum_compute_ms + 2.0 * float(sum(per_layer_comm_ms)) + float(sum(per_layer_grad_comm_ms)) + 1000.0)

    # Variable indexing.
    idx = 0
    keep_idx = [idx + li for li in range(L)]
    idx += L

    xF_idx: List[Dict[int, int]] = [{} for _ in range(L)]
    lookahead = max(0, int(cfg.lookahead_blocks))
    for li in active_layers:
        min_k = max(0, int(li) - int(lookahead))
        for k in range(min_k, li + 1):
            xF_idx[li][k] = idx
            idx += 1

    xB_idx: List[Dict[int, int]] = [{} for _ in range(L)]
    for li in active_layers:
        b_deadline = 2 * L - 1 - li
        min_k = max(int(li) + 1, int(b_deadline) - int(lookahead))
        for k in range(min_k, b_deadline + 1):
            xB_idx[li][k] = idx
            idx += 1

    readyF_idx = [idx + li for li in range(L)]
    idx += L
    readyB_idx = [idx + li for li in range(L)]
    idx += L

    now_idx = [idx + k for k in range(K + 1)]
    idx += K + 1
    stall_idx = [idx + k for k in range(K)]
    idx += K
    comm_start_idx = [idx + k for k in range(K)]
    idx += K
    comm_launch_idx = [idx + k for k in range(K)]
    idx += K
    comm_after_idx = [idx + k for k in range(K)]
    idx += K

    n_vars = idx

    # Bounds & integrality.
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, float(M_time), dtype=float)
    integrality = np.zeros(n_vars, dtype=int)

    # keep variables.
    for li in range(L):
        integrality[keep_idx[li]] = 1
        if li not in active_layers:
            ub[keep_idx[li]] = 0.0
            continue
        if forced_keep_layers is not None and li in forced_keep_layers:
            lb[keep_idx[li]] = 1.0
            ub[keep_idx[li]] = 1.0
        else:
            ub[keep_idx[li]] = 1.0

    # xF/xB binaries.
    for li in active_layers:
        for _, j in xF_idx[li].items():
            integrality[j] = 1
            ub[j] = 1.0
        for _, j in xB_idx[li].items():
            integrality[j] = 1
            if forced_keep_layers is not None and li in forced_keep_layers:
                ub[j] = 0.0
            else:
                ub[j] = 1.0

    # readyF/readyB for inactive layers are pinned to 0.
    for li in range(L):
        if li not in active_layers:
            ub[readyF_idx[li]] = 0.0
            ub[readyB_idx[li]] = 0.0

    # Objective: minimize end time now[K]. Low-communication backends add an
    # opportunity cost to long-lived keep decisions so small predicted gather
    # savings do not consume memory that could be used by cheaper mechanisms.
    c = np.zeros(n_vars, dtype=float)
    c[now_idx[K]] = 1.0
    for li, penalty in keep_penalties.items():
        if 0 <= int(li) < int(L) and float(penalty) > 0.0:
            c[keep_idx[int(li)]] += float(penalty)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    lb_rows: List[float] = []
    ub_rows: List[float] = []
    row = 0

    def add_row(coeffs: Sequence[Tuple[int, float]], lb_v: float, ub_v: float) -> None:
        nonlocal row
        for j, v in coeffs:
            if v == 0.0:
                continue
            rows.append(row)
            cols.append(int(j))
            data.append(float(v))
        lb_rows.append(float(lb_v))
        ub_rows.append(float(ub_v))
        row += 1

    INF = float(np.inf)

    # now[0] == 0.
    add_row([(now_idx[0], 1.0)], 0.0, 0.0)

    # Assignment constraints per layer.
    for li in active_layers:
        add_row([(j, 1.0) for j in xF_idx[li].values()], 1.0, 1.0)

        # keep + sum(xB) == 1  (if keep==1, skip BWD prefetch).
        coeffs = [(keep_idx[li], 1.0)] + [(j, 1.0) for j in xB_idx[li].values()]
        add_row(coeffs, 1.0, 1.0)

    # now recurrence: now[k+1] - now[k] - stall[k] == compute_ms[k].
    for k in range(K):
        add_row([(now_idx[k + 1], 1.0), (now_idx[k], -1.0), (stall_idx[k], -1.0)], float(blocks[k].compute_ms),
                float(blocks[k].compute_ms))

    # Comm timeline constraints:
    # - comm_start[k] is when the comm stream can start launches at boundary k.
    # - comm_launch[k] is the comm stream time after finishing the (scheduled) param-prefetch launch at boundary k.
    # - comm_after[k] is the comm stream time after also accounting for "fixed" comm that happens inside block k
    #   (e.g., grad reduction), which delays launches at later boundaries.
    comm_coeff: List[float] = []
    for li in range(L):
        alpha_eff = float(alpha_per_op_fused) if int(layer_ops[li]) > 1 else float(alpha_ms)
        comm_coeff.append(float(alpha_eff) * float(layer_ops[li]) + float(beta) * float(layer_sizes[li]))

    # Approximate grad-comm cost per layer (reduce-scatter / allreduce) as bandwidth-dominated.
    grad_comm_ms_per_layer = [float(beta_grad) * float(layer_sizes[li]) for li in range(L)]

    for k in range(K):
        # comm_start[k] >= comm_after[k-1]
        if k > 0:
            add_row([(comm_start_idx[k], 1.0), (comm_after_idx[k - 1], -1.0)], 0.0, INF)

        # comm_start[k] >= now[k]
        add_row([(comm_start_idx[k], 1.0), (now_idx[k], -1.0)], 0.0, INF)

        # comm_launch[k] == comm_start[k] + comm_ms_launch[k]
        coeffs_launch = [(comm_launch_idx[k], 1.0), (comm_start_idx[k], -1.0)]
        for li in active_layers:
            if k in xF_idx[li]:
                coeffs_launch.append((xF_idx[li][k], -comm_coeff[li]))
            if k in xB_idx[li]:
                coeffs_launch.append((xB_idx[li][k], -comm_coeff[li]))
        add_row(coeffs_launch, 0.0, 0.0)

        # With a dedicated allgather communicator, prefetch/allgather launches no longer serialize with
        # backward grad reductions on the reduce communicator/stream.
        fixed_ms = 0.0
        fixed_off = 0.0
        if (not bool(cfg.separate_allgather_communicator)) and blocks[k].phase == "BWD":
            fixed_ms = float(grad_comm_ms_per_layer[int(blocks[k].layer_idx)])
            fixed_off = float(blocks[k].fixed_comm_start_offset_ms) if fixed_ms > 0.0 else 0.0

        # comm_after[k] >= comm_launch[k] + fixed_ms
        add_row([(comm_after_idx[k], 1.0), (comm_launch_idx[k], -1.0)], float(fixed_ms), INF)
        # comm_after[k] >= now[k] + fixed_off + fixed_ms
        add_row([(comm_after_idx[k], 1.0), (now_idx[k], -1.0)], float(fixed_off + fixed_ms), INF)

    # Link ready times to selected comm launch boundary with big-M.
    # Within a single boundary k, comm tasks are executed in ascending deadline order, so the ready time
    # for task (li,kind) launched at k is comm_start[k] + sum(comm_cost of tasks with <= its deadline launched at k).
    M = float(M_time)
    for li in active_layers:
        for k, j in xF_idx[li].items():
            coeffs = [(readyF_idx[li], 1.0), (comm_start_idx[int(k)], -1.0)]
            for lj in active_layers:
                if int(lj) > int(li):
                    continue
                jj = xF_idx[lj].get(int(k))
                if jj is not None:
                    coeffs.append((jj, -comm_coeff[int(lj)]))

            # readyF - (comm_start + prefix) <= M*(1-xF)
            add_row(coeffs + [(j, float(M))], -INF, float(M))
            # readyF - (comm_start + prefix) >= -M*(1-xF)
            add_row(coeffs + [(j, -float(M))], -float(M), INF)

        for k, j in xB_idx[li].items():
            kk = int(k)
            coeffs = [(readyB_idx[li], 1.0), (comm_start_idx[kk], -1.0)]
            # Forward tasks are always ordered before backward tasks at the same boundary.
            for lj in active_layers:
                jj = xF_idx[lj].get(kk)
                if jj is not None:
                    coeffs.append((jj, -comm_coeff[int(lj)]))
            # Backward tasks are ordered by increasing b_deadline, i.e., decreasing layer index.
            for lj in active_layers:
                if int(lj) < int(li):
                    continue
                jj = xB_idx[lj].get(kk)
                if jj is not None:
                    coeffs.append((jj, -comm_coeff[int(lj)]))

            add_row(coeffs + [(j, float(M))], -INF, float(M))
            add_row(coeffs + [(j, -float(M))], -float(M), INF)

    # Model end-of-backward synchronization: the step can't finish before comm completes.
    add_row([(now_idx[K], 1.0), (comm_after_idx[K - 1], -1.0)], 0.0, INF)

    # Stall constraints (compute waits for the required task).
    for k in range(K):
        block = blocks[k]
        layer = int(block.layer_idx)
        if block.phase == "FWD":
            add_row([(stall_idx[k], 1.0), (now_idx[k], 1.0), (readyF_idx[layer], -1.0)], 0.0, INF)
        else:
            # If keep[layer]==1, required is FWD; else required is BWD.
            # stall >= readyF - now - M*(1-keep)
            add_row([(stall_idx[k], 1.0), (now_idx[k], 1.0), (readyF_idx[layer], -1.0), (keep_idx[layer], -float(M))],
                    -float(M), INF)
            # stall >= readyB - now - M*keep
            add_row([(stall_idx[k], 1.0), (now_idx[k], 1.0), (readyB_idx[layer], -1.0), (keep_idx[layer], float(M))],
                    0.0, INF)

    if max_keep_layers is not None:
        max_keep = max(0, int(max_keep_layers))
        add_row([(keep_idx[li], 1.0) for li in active_layers], -INF, float(max_keep))

    # Per-block launch and fuse constraints.
    for k in range(K):
        # max_tasks_per_block: count of layer-tasks launched at this boundary.
        coeffs_cnt: List[Tuple[int, float]] = []
        for li in active_layers:
            if k in xF_idx[li]:
                coeffs_cnt.append((xF_idx[li][k], 1.0))
            if k in xB_idx[li]:
                coeffs_cnt.append((xB_idx[li][k], 1.0))

        if coeffs_cnt:
            add_row(coeffs_cnt, -INF, float(cfg.max_tasks_per_block))
        # NOTE: We intentionally do not constrain total bytes launched at boundary k.
        # _build_comm_groups can split launches into multiple fused prefetch calls per boundary,
        # each capped by fuse_max_bytes. A hard "sum(bytes)<=fuse_max_bytes" constraint would be
        # overly strict and can make the MILP infeasible for large single-layer parameters
        # (e.g., embeddings) even though the runtime can still allgather them.

    # Per-block memory cap constraints.
    for k in range(K):
        rhs = float(blocks[k].cap_bytes)
        coeffs: List[Tuple[int, float]] = []
        for li in active_layers:
            size = float(layer_sizes[li])
            if size <= 0.0:
                continue
            f_deadline = li
            b_deadline = 2 * L - 1 - li

            delta_f = 1.0 if k > f_deadline else 0.0
            delta_b = 1.0 if k > b_deadline else 0.0

            # Forward task live: sum_{s <= min(k, f_deadline)} xF[li,s] - delta_f*(1-keep[li])
            # We move the constant term (-delta_f) to RHS.
            if delta_f > 0.0:
                rhs += size * delta_f
                coeffs.append((keep_idx[li], size * delta_f))

            max_s_f = min(k, f_deadline)
            for s in range(0, max_s_f + 1):
                j = xF_idx[li].get(s)
                if j is not None:
                    coeffs.append((j, size))

            # Backward task live: sum_{s <= min(k, b_deadline)} xB[li,s] - delta_b*(1-keep[li])
            # with s in [f_deadline+1, b_deadline]
            if delta_b > 0.0:
                rhs += size * delta_b
                coeffs.append((keep_idx[li], size * delta_b))

            if k <= b_deadline:
                max_s_b = min(k, b_deadline)
                for s in range(f_deadline + 1, max_s_b + 1):
                    j = xB_idx[li].get(s)
                    if j is not None:
                        coeffs.append((j, size))

        if coeffs:
            add_row(coeffs, -INF, rhs)

    A = coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                   shape=(row, n_vars)).tocsr()
    constraint = LinearConstraint(A, np.array(lb_rows, dtype=float), np.array(ub_rows, dtype=float))

    options = {
        "disp": False,
        "time_limit": float(cfg.milp_time_limit_s),
        "presolve": bool(cfg.milp_presolve),
        "mip_rel_gap": float(cfg.milp_rel_gap),
    }
    if int(cfg.milp_node_limit) > 0:
        options["node_limit"] = int(cfg.milp_node_limit)

    milp_units = int(len(active_layers))
    milp_blocks = int(K)
    milp_binary_vars = int(np.count_nonzero(integrality == 1))
    milp_constraints = int(row)
    solve_start_s = time.perf_counter()
    res = milp(c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraint, options=options)
    milp_solve_time_s = float(time.perf_counter() - solve_start_s)
    if res.status not in (0, 1) or res.x is None:
        raise RuntimeError(f"[{NAME}] MILP solver failed: status={res.status} message={res.message}")

    x = res.x

    keep_layers: Set[int] = {li for li in active_layers if float(x[keep_idx[li]]) >= 0.5}
    per_task_start: Dict[Tuple[int, str], int] = {}

    for li in active_layers:
        # FWD start
        best_k = None
        best_v = -1.0
        for k, j in xF_idx[li].items():
            v = float(x[j])
            if v > best_v:
                best_v = v
                best_k = k
        if best_k is None:
            raise RuntimeError(f"[{NAME}] MILP produced no FWD start for layer={li}")
        per_task_start[(li, "FWD")] = int(best_k)

        if li not in keep_layers:
            best_k = None
            best_v = -1.0
            for k, j in xB_idx[li].items():
                v = float(x[j])
                if v > best_v:
                    best_v = v
                    best_k = k
            if best_k is None:
                raise RuntimeError(f"[{NAME}] MILP produced no BWD start for layer={li}")
            per_task_start[(li, "BWD")] = int(best_k)

    final_gap_raw = getattr(res, "mip_gap", float("nan"))
    final_gap = float(final_gap_raw) if final_gap_raw is not None else float("nan")
    meta = {
        "status": int(res.status),
        "message": str(res.message),
        "objective": float(res.fun) if res.fun is not None else None,
        "mip_gap": final_gap,
        "final_gap": final_gap,
        "units": int(milp_units),
        "blocks": int(milp_blocks),
        "binary_vars": int(milp_binary_vars),
        "constraints": int(milp_constraints),
        "solve_time_s": float(milp_solve_time_s),
        "time_limit_s": float(cfg.milp_time_limit_s),
        "rel_gap_target": float(cfg.milp_rel_gap),
        "keep_penalty_layers": int(len(keep_penalties)),
        "keep_penalty_total_ms": float(sum(float(v) for v in keep_penalties.values())),
        "max_keep_layers": int(max_keep_layers) if max_keep_layers is not None else None,
    }

    return keep_layers, per_task_start, meta


def _build_tasks_from_plan(*,
                           L: int,
                           layer_to_ds_ids: Sequence[Sequence[int]],
                           ds_id_to_size: Dict[int, int],
                           keep_layers: Set[int],
                           per_task_start: Dict[Tuple[int, str], int],
                           persistent_set: Set[int],
                           ds_id_to_wait_pos_fwd: Optional[Dict[int, int]] = None,
                           ds_id_to_wait_pos_bwd: Optional[Dict[int, int]] = None) -> List[_Task]:
    tasks: List[_Task] = []

    def order_ds_ids(ds_ids: Sequence[int], *, kind: str) -> Tuple[int, ...]:
        if kind == "FWD":
            pos = ds_id_to_wait_pos_fwd
        else:
            pos = ds_id_to_wait_pos_bwd
        if not pos:
            return tuple(int(x) for x in ds_ids)
        # Earlier wait position => earlier allgather launch to maximize intra-layer overlap.
        # Tie-break on size to prioritize small early-use params (reduces first-wait stalls).
        return tuple(
            sorted(
                (int(x) for x in ds_ids),
                key=lambda d: (
                    int(pos.get(int(d), 1 << 60)),
                    int(ds_id_to_size.get(int(d), 0)),
                    int(d),
                ),
            ))

    for layer_idx in range(L):
        raw_ds_ids = [int(ds_id) for ds_id in layer_to_ds_ids[layer_idx] if int(ds_id) in ds_id_to_size]
        if not raw_ds_ids:
            continue

        # Dedup while preserving relative order before applying a use-position sort.
        seen: Set[int] = set()
        uniq_ds_ids: List[int] = []
        for ds_id in raw_ds_ids:
            if ds_id in seen:
                continue
            seen.add(ds_id)
            uniq_ds_ids.append(int(ds_id))

        size_bytes = int(sum(int(ds_id_to_size[int(ds_id)]) for ds_id in uniq_ds_ids))
        if size_bytes <= 0:
            continue

        f_deadline = int(layer_idx)
        b_deadline = int(2 * L - 1 - layer_idx)
        keep = layer_idx in keep_layers

        start_f = per_task_start.get((layer_idx, "FWD"))
        if start_f is None:
            raise RuntimeError(f"[{NAME}] schedule missing FWD start for layer={layer_idx}")

        fwd_ds_ids = order_ds_ids(uniq_ds_ids, kind="FWD")
        tasks.append(
            _Task(
                kind="FWD",
                layer_idx=int(layer_idx),
                ds_ids=tuple(int(x) for x in fwd_ds_ids),
                size_bytes=int(size_bytes),
                min_start_k=0,
                deadline_k=f_deadline,
                release_end_k=(b_deadline if keep else f_deadline),
                start_k=int(start_f),
            ))

        if keep:
            continue

        bwd_ds_ids = [int(ds_id) for ds_id in uniq_ds_ids if int(ds_id) not in persistent_set]
        bwd_ds_ids = list(order_ds_ids(bwd_ds_ids, kind="BWD"))
        bwd_size_bytes = int(sum(int(ds_id_to_size.get(int(ds_id), 0)) for ds_id in bwd_ds_ids))
        if not bwd_ds_ids or bwd_size_bytes <= 0:
            continue

        start_b = per_task_start.get((layer_idx, "BWD"))
        if start_b is None:
            raise RuntimeError(f"[{NAME}] schedule missing BWD start for layer={layer_idx}")

        tasks.append(
            _Task(
                kind="BWD",
                layer_idx=int(layer_idx),
                ds_ids=tuple(int(x) for x in bwd_ds_ids),
                size_bytes=int(bwd_size_bytes),
                min_start_k=f_deadline + 1,
                deadline_k=b_deadline,
                release_end_k=b_deadline,
                start_k=int(start_b),
            ))

    return tasks


def _is_mem_feasible(tasks: Sequence[_Task], blocks: Sequence[_BlockProfile]) -> bool:
    K = len(blocks)
    reserved = _reserved_mem_by_block(tasks, K)
    for k in range(K):
        if int(reserved[k]) > int(blocks[k].cap_bytes):
            return False
    return True


def _evaluate_plan_ms(*,
                      blocks: Sequence[_BlockProfile],
                      tasks: Sequence[_Task],
                      keep_layers: Set[int],
                      comm_model: _CommModel,
                      ds_id_to_size: Dict[int, int],
                      cfg: _SchedulerConfig) -> Tuple[float, Dict[int, List[List[int]]]]:
    launches, groups_by_start = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg)
    step_ms, _, _, _ = _simulate_schedule_in_graph_order(blocks,
                                                         groups_by_start,
                                                         keep_layers,
                                                         separate_allgather_communicator=bool(
                                                             cfg.separate_allgather_communicator))
    return float(step_ms), launches


def _refine_plan_local_search(*,
                              L: int,
                              blocks: Sequence[_BlockProfile],
                              layer_to_ds_ids: Sequence[Sequence[int]],
                              ds_id_to_size: Dict[int, int],
                              cfg: _SchedulerConfig,
                              comm_model: _CommModel,
                              keep_layers: Set[int],
                              per_task_start: Dict[Tuple[int, str], int],
                              persistent_set: Set[int],
                              ds_id_to_wait_pos_fwd: Optional[Dict[int, int]] = None,
                              ds_id_to_wait_pos_bwd: Optional[Dict[int, int]] = None) -> Tuple[Dict[Tuple[int, str], int], dict]:
    max_iters = int(cfg.inner_refine_max_iters)
    if max_iters <= 0:
        return per_task_start, {"iters": 0}

    lookahead = max(0, int(cfg.lookahead_blocks))
    min_gain = float(cfg.inner_refine_min_gain_ms)

    # Refine only tasks that exist (skip layers with no ds_ids, or BWD tasks fully persistent).
    base_tasks = _build_tasks_from_plan(L=L,
                                        layer_to_ds_ids=layer_to_ds_ids,
                                        ds_id_to_size=ds_id_to_size,
                                        keep_layers=keep_layers,
                                        per_task_start=per_task_start,
                                        persistent_set=persistent_set,
                                        ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                        ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)
    refined_keys = sorted({(int(t.layer_idx), str(t.kind)) for t in base_tasks})
    if not refined_keys:
        return per_task_start, {"iters": 0}

    if not _is_mem_feasible(base_tasks, blocks):
        # Shouldn't happen (MILP enforces), but keep the schedule unchanged if it does.
        return per_task_start, {"iters": 0, "skipped": "infeasible_base_mem"}

    best_ms, _ = _evaluate_plan_ms(blocks=blocks,
                                   tasks=base_tasks,
                                   keep_layers=keep_layers,
                                   comm_model=comm_model,
                                   ds_id_to_size=ds_id_to_size,
                                   cfg=cfg)
    start_ms = float(best_ms)

    it = 0
    while it < max_iters:
        best_move = None  # (key, new_k, cand_ms)

        for layer, kind in refined_keys:
            key = (int(layer), str(kind))
            cur_k = int(per_task_start.get(key, 0))

            if kind == "FWD":
                lo = max(0, int(layer) - int(lookahead))
                hi = int(layer)
            else:
                if layer in keep_layers:
                    continue
                b_deadline = int(2 * L - 1 - int(layer))
                lo = max(int(layer) + 1, int(b_deadline) - int(lookahead))
                hi = int(b_deadline)

            # Explore a small set of candidate start positions instead of only +/-1.
            # This helps unlock beneficial fusion (co-locating tasks at the same boundary)
            # and can escape local minima quickly.
            candidate_ks: Set[int] = {int(lo), int(hi)}
            candidate_ks.update(int(v) for v in per_task_start.values() if int(lo) <= int(v) <= int(hi))
            for delta in (-2, -1, 1, 2):
                candidate_ks.add(int(cur_k) + int(delta))

            for cand_k in sorted(candidate_ks):
                cand_k = int(cand_k)
                if cand_k < lo or cand_k > hi or cand_k == cur_k:
                    continue

                per_task_start[key] = int(cand_k)
                try:
                    cand_tasks = _build_tasks_from_plan(L=L,
                                                        layer_to_ds_ids=layer_to_ds_ids,
                                                        ds_id_to_size=ds_id_to_size,
                                                        keep_layers=keep_layers,
                                                        per_task_start=per_task_start,
                                                        persistent_set=persistent_set,
                                                        ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                                        ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)
                finally:
                    per_task_start[key] = int(cur_k)

                if not _is_mem_feasible(cand_tasks, blocks):
                    continue

                cand_ms, _ = _evaluate_plan_ms(blocks=blocks,
                                               tasks=cand_tasks,
                                               keep_layers=keep_layers,
                                               comm_model=comm_model,
                                               ds_id_to_size=ds_id_to_size,
                                               cfg=cfg)
                improved = float(cand_ms) < float(best_ms) - float(min_gain)
                if improved:
                    if best_move is None:
                        best_move = (key, int(cand_k), float(cand_ms))
                        continue
                    _, _, cur_best_ms = best_move
                    if float(cand_ms) < float(cur_best_ms) - 1e-6:
                        best_move = (key, int(cand_k), float(cand_ms))

        if best_move is None:
            break

        key, new_k, cand_ms = best_move
        per_task_start[key] = int(new_k)
        best_ms = float(cand_ms)
        it += 1

    return per_task_start, {
        "iters": int(it),
        "start_predicted_step_ms": float(start_ms),
        "final_predicted_step_ms": float(best_ms),
    }


def _build_schedule(graph_id: int, profiling_results, param_manager: Dict[int, DSGraphParamManager]) -> Optional[dict]:
    if _CFG is None or _LAYER_MAPPING is None:
        return None

    cfg = _CFG
    mapping = _LAYER_MAPPING

    L = mapping.L
    if L <= 0:
        return None

    ds_id_to_size = _build_ds_id_to_size_bytes(graph_id, profiling_results, param_manager)
    total_mem = _min_total_mem_bytes_across_ranks()
    peak_mem = _peak_mem_bytes(profiling_results)
    mem_budget_bytes = max(0, int(total_mem) - int(peak_mem))
    allocator_margin_bytes = _allocator_margin_bytes(total_mem)

    prof = profiling_results.get(graph_id)
    if prof is None or getattr(prof, "fwd_graph", None) is None or getattr(prof, "bwd_graph", None) is None:
        return None

    ds_id_to_wait_pos_fwd = _build_ds_id_to_wait_pos(getattr(prof, "fwd_graph", None))
    ds_id_to_wait_pos_bwd = _build_ds_id_to_wait_pos(getattr(prof, "bwd_graph", None))

    blocks_fwd = _extract_blocks_for_graph(prof.fwd_graph,
                                          getattr(prof, "fwd_mem", []),
                                          L,
                                          mapping.ds_id_to_layer,
                                          ds_id_to_size,
                                          total_mem_bytes=total_mem,
                                          allocator_margin_bytes=allocator_margin_bytes,
                                          bwd=False)
    blocks_bwd = _extract_blocks_for_graph(prof.bwd_graph,
                                          getattr(prof, "bwd_mem", []),
                                          L,
                                          mapping.ds_id_to_layer,
                                          ds_id_to_size,
                                          total_mem_bytes=total_mem,
                                          allocator_margin_bytes=allocator_margin_bytes,
                                          bwd=True)

    K = 2 * L
    block_by_k: Dict[int, _BlockProfile] = {b.k: b for b in (blocks_fwd + blocks_bwd)}
    if len(block_by_k) != K:
        missing = sorted(set(range(K)) - set(block_by_k.keys()))
        log_rank0(f"[{NAME}] WARNING: missing block profiles for k={missing}; skip scheduling.", enable=True)
        return None

    blocks = [block_by_k[k] for k in range(K)]

    # Layer groups used by the scheduler. Start from regex-based mapping, then optionally
    # fold in "unmapped" ds_ids (e.g., embeddings / head) by their first wait position.
    layer_to_ds_ids: List[List[int]] = [list(map(int, ids)) for ids in mapping.layer_to_ds_ids]
    extra_ds_id_to_layer: Dict[int, int] = {}
    unmapped = sorted({int(ds_id) for ds_id in ds_id_to_size.keys() if int(ds_id) not in mapping.ds_id_to_layer})
    if unmapped:
        anchors_fwd = _find_layer_anchors(prof.fwd_graph, mapping.ds_id_to_layer, L)
        if anchors_fwd:
            nodes = list(prof.fwd_graph.nodes)
            pos = {n: i for i, n in enumerate(nodes)}
            layer_anchor_pos = sorted([(int(layer), int(pos[n])) for layer, n in anchors_fwd.items()],
                                      key=lambda x: x[1])
            if layer_anchor_pos:
                wait_pos_by_ds_id: Dict[int, int] = {}
                for n in prof.fwd_graph.nodes:
                    if n.target == torch.ops.dc.wait_allgather.default:
                        wait_pos_by_ds_id[_get_ds_id_from_wait(n)] = int(pos[n])
                for ds_id in unmapped:
                    wp = wait_pos_by_ds_id.get(int(ds_id))
                    if wp is None:
                        continue
                    assigned_layer = int(layer_anchor_pos[0][0])
                    for layer, ap in layer_anchor_pos:
                        if int(ap) <= int(wp):
                            assigned_layer = int(layer)
                        else:
                            break
                    extra_ds_id_to_layer[int(ds_id)] = int(assigned_layer)
                    layer_to_ds_ids[int(assigned_layer)].append(int(ds_id))

        # Normalize (dedup + sort) for determinism.
        for i in range(L):
            layer_to_ds_ids[i] = sorted(set(int(x) for x in layer_to_ds_ids[i]))

    comm_model = _fit_comm_model(graph_id, profiling_results, ds_id_to_size)
    ds_id_to_allgather_ms = _build_ds_id_to_allgather_ms(profiling_results)

    # Approximate fixed (non-prefetch) communication that runs on the comm stream, e.g., grad reduction.
    # We model it as bandwidth-dominated and proportional to per-layer parameter volume.
    beta_grad = float(comm_model.beta_ms_per_byte)
    layer_sizes, layer_ops = _layer_size_and_ops(L, layer_to_ds_ids, ds_id_to_size)
    backend_policy = _resolve_comm_backend_policy(cfg, blocks, layer_sizes, layer_ops, comm_model)
    keep_layer_penalty_ms = _build_keep_opportunity_penalties(L=L,
                                                              layer_to_ds_ids=layer_to_ds_ids,
                                                              layer_sizes=layer_sizes,
                                                              ds_id_to_size=ds_id_to_size,
                                                              ds_id_to_allgather_ms=ds_id_to_allgather_ms,
                                                              cfg=cfg,
                                                              backend_policy=backend_policy)

    grad_comm_ms_per_layer = [float(beta_grad) * float(layer_sizes[li]) for li in range(L)]
    blocks = [
        _BlockProfile(
            k=int(b.k),
            phase=str(b.phase),
            layer_idx=int(b.layer_idx),
            compute_ms=float(b.compute_ms),
            cap_bytes=int(b.cap_bytes),
            explicit_temp_reserve_bytes=int(b.explicit_temp_reserve_bytes),
            non_param_floor_bytes=int(b.non_param_floor_bytes),
            fixed_comm_start_offset_ms=float(b.fixed_comm_start_offset_ms),
            fixed_comm_ms=float(grad_comm_ms_per_layer[int(b.layer_idx)]) if b.phase == "BWD" else 0.0,
        ) for b in blocks
    ]

    persistent_candidates: List[int] = []
    if int(mem_budget_bytes) > 0:
        # Rank candidates once during planning. The actual persistent set is
        # selected later (apply stage) based on observed runtime peak memory.
        persistent_candidates = _rank_persistent_candidates(ds_id_to_size, ds_id_to_allgather_ms)

    # NOTE: Persistent gathered buffers are *allocated* in the runtime (C++) and
    # can significantly increase the optimizer-step peak memory. Profiling-based
    # peak_mem does not include optimizer.step(), so selecting the persistent set
    # here can be overly optimistic and lead to OOM. We defer the selection &
    # allocation to apply(), where we can use the observed warmup-step peak.
    persistent_ds_ids: List[int] = []
    persistent_mem_bytes = 0
    persistent_set: Set[int] = set()
    forced_keep_layers: Set[int] = set()

    lookahead_blocks = _lookahead_blocks_for_num_layers(L)
    max_tasks_per_block = _max_tasks_per_block_for_num_layers(L)
    cfg_used = replace(cfg, lookahead_blocks=int(lookahead_blocks), max_tasks_per_block=int(max_tasks_per_block))

    keep_layers, per_task_start, milp_meta = _solve_milp_schedule(L=L,
                                                                  blocks=blocks,
                                                                  layer_to_ds_ids=layer_to_ds_ids,
                                                                  ds_id_to_size=ds_id_to_size,
                                                                  cfg=cfg_used,
                                                                  comm_model=comm_model,
                                                                  forced_keep_layers=forced_keep_layers,
                                                                  keep_layer_penalty_ms=keep_layer_penalty_ms)

    refine_meta: dict = {"iters": 0}
    if int(cfg_used.inner_refine_max_iters) > 0:
        per_task_start, refine_meta = _refine_plan_local_search(L=L,
                                                                blocks=blocks,
                                                                layer_to_ds_ids=layer_to_ds_ids,
                                                                ds_id_to_size=ds_id_to_size,
                                                                cfg=cfg_used,
                                                                comm_model=comm_model,
                                                                keep_layers=keep_layers,
                                                                per_task_start=per_task_start,
                                                                persistent_set=persistent_set,
                                                                ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                                                ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)

    tasks = _build_tasks_from_plan(L=L,
                                   layer_to_ds_ids=layer_to_ds_ids,
                                   ds_id_to_size=ds_id_to_size,
                                   keep_layers=keep_layers,
                                   per_task_start=per_task_start,
                                   persistent_set=persistent_set,
                                   ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                   ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)

    if not _is_mem_feasible(tasks, blocks):
        return None

    launches, groups_by_start = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg_used)
    step_ms, stall_ms_per_block, stall_bwd_per_layer, _ = _simulate_schedule_in_graph_order(
        blocks,
        groups_by_start,
        keep_layers,
        separate_allgather_communicator=bool(cfg_used.separate_allgather_communicator),
    )
    reserved_mem = _reserved_mem_by_block(tasks, K)
    total_prefetch_calls = int(sum(len(v) for v in launches.values()))

    trials: List[dict] = [{
        "lookahead_blocks": int(lookahead_blocks),
        "status": "ok",
        "predicted_step_ms": float(step_ms),
        "prefetch_calls": int(total_prefetch_calls),
        "milp": milp_meta,
        "inner_refine": refine_meta,
    }]

    used_fuse_window = max(0, int(cfg_used.fuse_deadline_window_blocks))
    best_milp_meta = milp_meta
    best_refine_meta = refine_meta

    keep_ds_ids: List[int] = []
    for li in sorted(keep_layers):
        keep_ds_ids.extend([int(ds_id) for ds_id in layer_to_ds_ids[li] if int(ds_id) in ds_id_to_size])
    keep_ds_ids = sorted(set(keep_ds_ids))

    schedule = {
        "version": 9,
        "mapping_hash": mapping.mapping_hash,
        "L": L,
        "keep_layers": sorted(keep_layers),
        "keep_ds_ids": keep_ds_ids,
        "persistent_ds_ids": persistent_ds_ids,
        "launches": {int(k): v for k, v in sorted(launches.items())},
        "per_task_start_k": {f"layer.{li}.{kind}": int(k) for (li, kind), k in sorted(per_task_start.items())},
        "predicted_reserved_mem": [int(x) for x in reserved_mem],
        "predicted_step_ms": float(step_ms),
        "predicted_stall_ms": [float(x) for x in stall_ms_per_block],
        "predicted_stall_bwd_per_layer": {int(li): float(v) for li, v in sorted(stall_bwd_per_layer.items())},
        "comm_model": {
            "alpha_ms": float(comm_model.alpha_ms),
            "beta_ms_per_byte": float(comm_model.beta_ms_per_byte),
        },
        "block_profiles": {
            "compute_ms": [float(b.compute_ms) for b in blocks],
            "cap_bytes": [int(b.cap_bytes) for b in blocks],
            "explicit_temp_reserve_bytes": [int(b.explicit_temp_reserve_bytes) for b in blocks],
            "non_param_floor_bytes": [int(b.non_param_floor_bytes) for b in blocks],
            "fixed_comm_start_offset_ms": [float(b.fixed_comm_start_offset_ms) for b in blocks],
            "fixed_comm_ms": [float(b.fixed_comm_ms) for b in blocks],
        },
        "meta": {
            "lookahead_blocks": int(cfg_used.lookahead_blocks),
            "lookahead_trials": trials,
            "fuse_max_bytes": int(cfg_used.fuse_max_bytes),
            "fuse_deadline_window_blocks": int(used_fuse_window),
            "requested_fuse_deadline_window_blocks": int(cfg.fuse_deadline_window_blocks),
            "fuse_factor": float(cfg_used.fuse_factor),
            "inner_refine_max_iters": int(cfg_used.inner_refine_max_iters),
            "inner_refine_min_gain_ms": float(cfg_used.inner_refine_min_gain_ms),
            "inner_refine": best_refine_meta,
            "total_mem_bytes": int(total_mem),
            "peak_mem_bytes": int(peak_mem),
            "mem_budget_bytes": int(mem_budget_bytes),
            "allocator_margin_bytes": int(allocator_margin_bytes),
            "explicit_temp_reserve_bytes": int(max((int(b.explicit_temp_reserve_bytes) for b in blocks), default=0)),
            "persistent_mem_bytes": int(persistent_mem_bytes),
            "persistent_deferred": True,
            "persistent_candidates": [int(x) for x in persistent_candidates],
            "persistent_candidate_sizes": {str(int(ds_id)): int(ds_id_to_size.get(int(ds_id), 0)) for ds_id in persistent_candidates},
            "persistent_candidate_allgather_ms": {str(int(ds_id)): float(ds_id_to_allgather_ms.get(int(ds_id), 0.0)) for ds_id in persistent_candidates},
            "ds_id_to_layer": {
                str(int(ds_id)): int(layer) for ds_id, layer in sorted({
                    **{int(k): int(v) for k, v in mapping.ds_id_to_layer.items()},
                    **{int(k): int(v) for k, v in extra_ds_id_to_layer.items()},
                }.items())
            },
            "extra_unmapped_ds_ids": len(extra_ds_id_to_layer),
            "prefetch_calls": int(total_prefetch_calls),
            "milp": best_milp_meta,
            "milp_time_limit_s": float(cfg_used.milp_time_limit_s),
            "milp_node_limit": int(cfg_used.milp_node_limit),
            "milp_rel_gap": float(cfg_used.milp_rel_gap),
            "milp_presolve": bool(cfg_used.milp_presolve),
            "separate_allgather_communicator": bool(cfg_used.separate_allgather_communicator),
            "comm_backend_policy": backend_policy,
            "keep_opportunity_penalty_layers": int(len(keep_layer_penalty_ms)),
            "keep_opportunity_penalty_total_ms": float(sum(float(v) for v in keep_layer_penalty_ms.values())),
        },
    }
    predicted_graph_peak_mem_bytes = _predicted_peak_mem_bytes_from_schedule(schedule)
    schedule["meta"]["predicted_graph_peak_mem_bytes"] = int(predicted_graph_peak_mem_bytes)
    schedule["meta"]["predicted_graph_peak_mem_gb"] = float(predicted_graph_peak_mem_bytes) / float(1024**3)
    schedule["meta"]["predicted_peak_mem_bytes"] = int(predicted_graph_peak_mem_bytes)
    schedule["meta"]["predicted_peak_mem_gb"] = float(predicted_graph_peak_mem_bytes) / float(1024**3)
    schedule["meta"]["predicted_peak_memory_scope"] = "graph_only_until_apply"
    schedule["meta"]["predicted_peak_includes_persistent"] = False
    schedule["schedule_hash"] = _schedule_hash(schedule)
    return schedule


def _broadcast_and_store_schedule(schedule: dict) -> dict:
    global _LATEST_SCHEDULE

    if dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [schedule] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        schedule = obj_list[0]

    expected = schedule.get("schedule_hash")
    if expected is None:
        raise RuntimeError(f"[{NAME}] schedule missing schedule_hash")
    actual = _schedule_hash(schedule)
    if actual != expected:
        raise RuntimeError(f"[{NAME}] schedule hash mismatch: expected={expected} actual={actual}")

    _LATEST_SCHEDULE = schedule
    return schedule


def _dump_schedule_if_enabled(schedule: dict) -> None:
    if _CFG is None:
        return
    if dist.get_rank() != 0:
        return
    if not _CFG.dump_schedule and os.environ.get("DS_GLOBAL_LAYER_SCHEDULER_DUMP", "") == "":
        return

    dump_dir = _CFG.dump_dir or os.environ.get("DS_GLOBAL_LAYER_SCHEDULER_DUMP", "").strip() or "."
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"global_layer_schedule_{schedule.get('schedule_hash','unknown')}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_canonical_json(schedule))
    log_rank0(f"[{NAME}] Wrote schedule JSON: {path}", enable=True)


def _blocks_from_schedule(sched: dict) -> List[_BlockProfile]:
    L = int(sched.get("L", 0))
    K = 2 * L
    block_profiles = sched.get("block_profiles", {}) or {}

    def _arr(name: str, default):
        vals = block_profiles.get(name, [])
        if len(vals) < K:
            vals = list(vals) + [default for _ in range(K - len(vals))]
        return vals

    compute_ms = _arr("compute_ms", 0.0)
    cap_bytes = _arr("cap_bytes", 0)
    explicit_temp = _arr("explicit_temp_reserve_bytes", 0)
    non_param_floor = _arr("non_param_floor_bytes", 0)
    fixed_offsets = _arr("fixed_comm_start_offset_ms", 0.0)
    fixed_ms = _arr("fixed_comm_ms", 0.0)

    blocks: List[_BlockProfile] = []
    for k in range(K):
        if k < L:
            phase = "FWD"
            layer_idx = int(k)
        else:
            phase = "BWD"
            layer_idx = int(K - 1 - k)
        blocks.append(
            _BlockProfile(
                k=int(k),
                phase=phase,
                layer_idx=int(layer_idx),
                compute_ms=float(compute_ms[k]),
                cap_bytes=int(cap_bytes[k]),
                explicit_temp_reserve_bytes=int(explicit_temp[k]),
                non_param_floor_bytes=int(non_param_floor[k]),
                fixed_comm_start_offset_ms=float(fixed_offsets[k]),
                fixed_comm_ms=float(fixed_ms[k]),
            ))
    return blocks


def _parse_per_task_start(per_task_start_k: dict) -> Dict[Tuple[int, str], int]:
    out: Dict[Tuple[int, str], int] = {}
    for raw_key, raw_v in (per_task_start_k or {}).items():
        parts = str(raw_key).split(".")
        if len(parts) != 3 or parts[0] != "layer":
            continue
        kind = str(parts[2])
        if kind not in {"FWD", "BWD"}:
            continue
        try:
            out[(int(parts[1]), kind)] = int(raw_v)
        except Exception:
            continue
    return out


def _layer_to_ds_ids_from_mapping(L: int, ds_id_to_layer: Dict[int, int],
                                  ds_id_to_size: Dict[int, int]) -> List[List[int]]:
    layer_to_ds_ids: List[List[int]] = [[] for _ in range(int(L))]
    for ds_id, layer in ds_id_to_layer.items():
        layer = int(layer)
        if 0 <= layer < int(L) and int(ds_id) in ds_id_to_size:
            layer_to_ds_ids[layer].append(int(ds_id))
    for li in range(int(L)):
        layer_to_ds_ids[li] = sorted(set(layer_to_ds_ids[li]))
    return layer_to_ds_ids


def _schedule_with_keep_layers(sched: dict,
                               keep_layers: Set[int],
                               cfg: _SchedulerConfig,
                               comm_model: _CommModel,
                               ds_id_to_size: Dict[int, int],
                               ds_id_to_layer: Dict[int, int],
                               profiling_results,
                               graph_id: int,
                               persistent_set: Optional[Set[int]] = None) -> Optional[dict]:
    L = int(sched.get("L", 0))
    if L <= 0:
        return None

    blocks = _blocks_from_schedule(sched)
    layer_to_ds_ids = _layer_to_ds_ids_from_mapping(L, ds_id_to_layer, ds_id_to_size)
    if not any(layer_to_ds_ids):
        return None

    meta = sched.get("meta", {}) or {}
    lookahead = int(meta.get("lookahead_blocks", cfg.lookahead_blocks))
    cfg_used = replace(cfg,
                       lookahead_blocks=int(lookahead),
                       max_tasks_per_block=int(meta.get("max_tasks_per_block", cfg.max_tasks_per_block)))

    per_task_start = _parse_per_task_start(sched.get("per_task_start_k", {}))
    for li in range(L):
        if not layer_to_ds_ids[li]:
            continue
        if (li, "FWD") not in per_task_start:
            per_task_start[(li, "FWD")] = max(0, int(li) - int(cfg_used.lookahead_blocks))
        if li not in keep_layers and (li, "BWD") not in per_task_start:
            b_deadline = int(2 * L - 1 - li)
            per_task_start[(li, "BWD")] = max(int(li) + 1, int(b_deadline) - int(cfg_used.lookahead_blocks))

    prof = profiling_results.get(graph_id) if profiling_results is not None else None
    ds_id_to_wait_pos_fwd = _build_ds_id_to_wait_pos(getattr(prof, "fwd_graph", None)) if prof is not None else {}
    ds_id_to_wait_pos_bwd = _build_ds_id_to_wait_pos(getattr(prof, "bwd_graph", None)) if prof is not None else {}

    persistent_set_used = set(int(x) for x in (persistent_set or set()))

    refine_meta: dict = {"iters": 0, "skipped": "disabled"}
    if int(cfg_used.inner_refine_max_iters) > 0:
        per_task_start, refine_meta = _refine_plan_local_search(L=L,
                                                                blocks=blocks,
                                                                layer_to_ds_ids=layer_to_ds_ids,
                                                                ds_id_to_size=ds_id_to_size,
                                                                cfg=cfg_used,
                                                                comm_model=comm_model,
                                                                keep_layers=set(int(x) for x in keep_layers),
                                                                per_task_start=per_task_start,
                                                                persistent_set=persistent_set_used,
                                                                ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                                                ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)

    tasks = _build_tasks_from_plan(L=L,
                                   layer_to_ds_ids=layer_to_ds_ids,
                                   ds_id_to_size=ds_id_to_size,
                                   keep_layers=set(int(x) for x in keep_layers),
                                   per_task_start=per_task_start,
                                   persistent_set=persistent_set_used,
                                   ds_id_to_wait_pos_fwd=ds_id_to_wait_pos_fwd,
                                   ds_id_to_wait_pos_bwd=ds_id_to_wait_pos_bwd)
    if not _is_mem_feasible(tasks, blocks):
        return None

    launches, groups_by_start = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg_used)
    step_ms, stall_ms_per_block, stall_bwd_per_layer, _ = _simulate_schedule_in_graph_order(
        blocks,
        groups_by_start,
        set(int(x) for x in keep_layers),
        separate_allgather_communicator=bool(cfg_used.separate_allgather_communicator),
    )
    reserved_mem = _reserved_mem_by_block(tasks, 2 * L)

    keep_ds_ids: List[int] = []
    for li in sorted(set(int(x) for x in keep_layers)):
        keep_ds_ids.extend([int(ds_id) for ds_id in layer_to_ds_ids[int(li)] if int(ds_id) in ds_id_to_size])
    keep_ds_ids = sorted(set(keep_ds_ids))

    new_sched = dict(sched)
    new_meta = dict(meta)
    new_sched["keep_layers"] = sorted(set(int(x) for x in keep_layers))
    new_sched["keep_ds_ids"] = keep_ds_ids
    new_sched["persistent_ds_ids"] = sorted(persistent_set_used)
    new_sched["launches"] = {int(k): v for k, v in sorted(launches.items())}
    new_sched["per_task_start_k"] = {f"layer.{li}.{kind}": int(k) for (li, kind), k in sorted(per_task_start.items())}
    new_sched["predicted_reserved_mem"] = [int(x) for x in reserved_mem]
    new_sched["predicted_step_ms"] = float(step_ms)
    new_sched["predicted_stall_ms"] = [float(x) for x in stall_ms_per_block]
    new_sched["predicted_stall_bwd_per_layer"] = {int(li): float(v) for li, v in sorted(stall_bwd_per_layer.items())}
    new_meta["prefetch_calls"] = int(sum(len(v) for v in launches.values()))
    new_meta["inner_refine_after_rebalance"] = refine_meta
    new_meta["predicted_peak_includes_persistent"] = False
    new_meta["max_tasks_per_block"] = int(cfg_used.max_tasks_per_block)
    new_sched["meta"] = new_meta

    predicted_graph_peak_mem_bytes = _predicted_peak_mem_bytes_from_schedule(new_sched)
    new_meta["predicted_graph_peak_mem_bytes"] = int(predicted_graph_peak_mem_bytes)
    new_meta["predicted_graph_peak_mem_gb"] = float(predicted_graph_peak_mem_bytes) / float(1024**3)
    new_meta["predicted_peak_mem_bytes"] = int(predicted_graph_peak_mem_bytes)
    new_meta["predicted_peak_mem_gb"] = float(predicted_graph_peak_mem_bytes) / float(1024**3)
    new_meta["predicted_peak_memory_scope"] = "graph_only_until_apply"
    new_sched["meta"] = new_meta
    new_sched["schedule_hash"] = _schedule_hash(new_sched)
    return new_sched


def _choose_low_comm_keep_subset(current_keep_layers: Set[int],
                                 L: int,
                                 ds_id_to_layer: Dict[int, int],
                                 ds_id_to_size: Dict[int, int],
                                 ds_id_to_value: Dict[int, float],
                                 max_keep_layers: int) -> Set[int]:
    if max_keep_layers <= 0:
        return set()
    layer_value: Dict[int, float] = {int(li): 0.0 for li in current_keep_layers}
    layer_size: Dict[int, int] = {int(li): 0 for li in current_keep_layers}
    for ds_id, layer in ds_id_to_layer.items():
        layer = int(layer)
        if layer not in current_keep_layers:
            continue
        layer_value[layer] = layer_value.get(layer, 0.0) + max(0.0, float(ds_id_to_value.get(int(ds_id), 0.0)))
        layer_size[layer] = layer_size.get(layer, 0) + max(0, int(ds_id_to_size.get(int(ds_id), 0)))

    ranked = sorted(
        (int(li) for li in current_keep_layers if 0 <= int(li) < int(L)),
        key=lambda li: (
            float(layer_value.get(int(li), 0.0)),
            float(layer_value.get(int(li), 0.0)) / float(max(1, int(layer_size.get(int(li), 0)))),
            -int(li),
        ),
        reverse=True,
    )
    return set(int(x) for x in ranked[:max(0, int(max_keep_layers))])


def _maybe_rebalance_low_comm_schedule(sched: dict,
                                       cfg: _SchedulerConfig,
                                       persistent_budget_bytes: int,
                                       candidates: Sequence[int],
                                       ds_id_to_size: Dict[int, int],
                                       ds_id_to_value: Dict[int, float],
                                       ds_id_to_layer: Dict[int, int],
                                       profiling_results,
                                       graph_id: int) -> Tuple[dict, dict]:
    meta = sched.get("meta", {}) or {}
    backend_policy = meta.get("comm_backend_policy", {}) or {}
    if str(backend_policy.get("mode", "generic")) != "low_comm":
        return sched, {"status": "skipped", "reason": "backend_not_low_comm"}

    current_keep = set(map(int, sched.get("keep_layers", [])))
    if not current_keep:
        return sched, {"status": "skipped", "reason": "no_keep_layers"}

    candidate_total = int(sum(max(0, int(ds_id_to_size.get(int(ds_id), 0))) for ds_id in candidates))
    if candidate_total <= 0:
        return sched, {"status": "skipped", "reason": "no_persistent_candidates"}

    coverage = float(max(0, int(persistent_budget_bytes))) / float(candidate_total)
    starvation_fraction = max(0.0, min(1.0, float(cfg.low_comm_persistent_starvation_fraction)))
    if coverage >= starvation_fraction:
        return sched, {
            "status": "skipped",
            "reason": "persistent_not_starved",
            "persistent_budget_coverage": float(coverage),
            "candidate_total_bytes": int(candidate_total),
        }

    cap_fraction = max(0.0, min(1.0, float(cfg.low_comm_keep_cap_fraction)))
    max_keep_layers = int(math.floor(float(int(sched.get("L", 0))) * cap_fraction))
    if cap_fraction > 0.0 and max_keep_layers <= 0:
        max_keep_layers = 1
    if len(current_keep) <= max_keep_layers:
        return sched, {
            "status": "skipped",
            "reason": "already_within_keep_cap",
            "persistent_budget_coverage": float(coverage),
            "max_keep_layers": int(max_keep_layers),
        }

    L = int(sched.get("L", 0))
    keep_subset = _choose_low_comm_keep_subset(current_keep,
                                              L,
                                              ds_id_to_layer,
                                              ds_id_to_size,
                                              ds_id_to_value,
                                              max_keep_layers)
    comm_meta = sched.get("comm_model", {}) or {}
    comm_model = _CommModel(alpha_ms=float(comm_meta.get("alpha_ms", 0.0)),
                            beta_ms_per_byte=float(comm_meta.get("beta_ms_per_byte", 0.0)))
    candidate_sched = _schedule_with_keep_layers(sched,
                                                 keep_subset,
                                                 cfg,
                                                 comm_model,
                                                 ds_id_to_size,
                                                 ds_id_to_layer,
                                                 profiling_results,
                                                 graph_id)
    if candidate_sched is None:
        return sched, {
            "status": "skipped",
            "reason": "candidate_infeasible",
            "persistent_budget_coverage": float(coverage),
            "max_keep_layers": int(max_keep_layers),
        }

    old_step = float(sched.get("predicted_step_ms", 0.0))
    new_step = float(candidate_sched.get("predicted_step_ms", 0.0))
    if old_step > 0.0 and new_step > old_step * 1.20:
        return sched, {
            "status": "skipped",
            "reason": "predicted_regression_too_large",
            "old_predicted_step_ms": float(old_step),
            "new_predicted_step_ms": float(new_step),
            "persistent_budget_coverage": float(coverage),
            "max_keep_layers": int(max_keep_layers),
        }

    rebalance_meta = {
        "status": "applied",
        "reason": "low_comm_persistent_starved",
        "persistent_budget_coverage": float(coverage),
        "candidate_total_bytes": int(candidate_total),
        "persistent_budget_bytes": int(persistent_budget_bytes),
        "old_keep_layers": int(len(current_keep)),
        "new_keep_layers": int(len(keep_subset)),
        "max_keep_layers": int(max_keep_layers),
        "old_predicted_step_ms": float(old_step),
        "new_predicted_step_ms": float(new_step),
    }
    new_meta = dict(candidate_sched.get("meta", {}) or {})
    new_meta["low_comm_rebalance"] = rebalance_meta
    candidate_sched["meta"] = new_meta
    candidate_sched["schedule_hash"] = _schedule_hash(candidate_sched)
    return candidate_sched, rebalance_meta



def _maybe_plan_recompute_relief(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                                 create_inputs_fn, mem_budget: float,
                                 param_manager: Dict[int, DSGraphParamManager], bwd: bool) -> None:
    if _CFG is None or not bool(_CFG.low_comm_recompute_relief):
        return
    try:
        from . import selective_activation_recompute as recompute
        recompute.plan(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd)
    except Exception as exc:
        log_rank0(f"[{NAME}] Recompute relief planning skipped: {exc}", enable=True)


def _has_global_layer_scheduler_apply_pass(scheduled_passes: Optional[Sequence[object]]) -> bool:
    if not scheduled_passes:
        return False
    for p in scheduled_passes:
        if getattr(p, "__name__", "") != "apply":
            continue
        if getattr(p, "__module__", "").endswith(".global_layer_scheduler"):
            return True
    return False


def maybe_apply_before_recompile(scheduled_passes: Optional[Sequence[object]]) -> None:
    if _CFG is None or not bool(_CFG.low_comm_recompute_relief):
        return
    if not _has_global_layer_scheduler_apply_pass(scheduled_passes):
        return
    try:
        from . import selective_activation_recompute as recompute
        plan = getattr(recompute, "_LATEST_PLAN", None)
        if plan is None or bool(getattr(recompute, "_APPLY_LOGGED", False)):
            return
        recompute._refine_plan_with_runtime_peak(plan)
        recompute._APPLY_LOGGED = True
        target_layers = {int(x) for x in plan.get("checkpointed_layers", plan.get("selected_layers", []))}
        recompute._set_layer_checkpoint_selection(target_layers)
        log_rank0(
            f"[{NAME}] Applied Chorus recompute relief before recompile: "
            f"checkpointed_layers={plan.get('checkpointed_layers', plan.get('selected_layers', []))} "
            f"disabled_layers={plan.get('disabled_layers', [])} "
            f"runtime_peak_alloc_bytes={plan.get('runtime_peak_alloc_bytes', 0)} "
            f"effective_budget_bytes={plan.get('effective_budget_bytes', plan.get('budget_bytes', 0))} "
            f"reserve_bytes={plan.get('pressure_reserve_bytes', 0)}",
            enable=True,
        )
    except Exception as exc:
        log_rank0(f"[{NAME}] Recompute relief apply skipped: {exc}", enable=True)

def _maybe_set_persistent(persistent_ds_ids: Sequence[int]) -> None:
    ds_ids = [int(x) for x in (persistent_ds_ids or [])]
    if not ds_ids:
        return
    nz3 = get_deepcompile_handle()
    # DeepCompile's `set_persistent()` eagerly allocates gathered buffers by
    # calling into the Z3 custom op executor. When torch.compile is using
    # FakeTensorMode, executing this op can fail because NCCL/allgather needs
    # real data pointers. Temporarily disable fake tensor mode here.
    set_chorus_persistent = getattr(nz3, "set_chorus_persistent", nz3.set_persistent)
    if unset_fake_temporarily is None:
        for ds_id in ds_ids:
            set_chorus_persistent(int(ds_id))
    else:
        with unset_fake_temporarily():
            for ds_id in ds_ids:
                set_chorus_persistent(int(ds_id))
    log_rank0(f"[{NAME}] Set Chorus persistent buffers: {len(ds_ids)} ds_ids", enable=True)


def plan(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
         mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool):
    if not bwd:
        return None
    if _LAYER_MAPPING is None:
        return None

    last_bwd = _last_backward_graph_id(graph_order)
    if last_bwd is None or graph_id != last_bwd:
        return None

    schedule = _build_schedule(graph_id, profiling_results, param_manager)
    if schedule is None:
        return None

    if dist.get_rank() == 0:
        meta = schedule.get("meta", {})
        log_rank0(
            f"[{NAME}] Planned schedule: L={schedule['L']} keep_layers={len(schedule['keep_layers'])} keep_ds_ids={len(schedule['keep_ds_ids'])} launches={len(schedule['launches'])} prefetch_calls={meta.get('prefetch_calls', None)} lookahead={meta.get('lookahead_blocks', None)} predicted_graph_peak_mem_gb={meta.get('predicted_graph_peak_mem_gb', None)} mem_budget_bytes={meta.get('mem_budget_bytes', None)}",
            enable=True,
        )

    schedule = _broadcast_and_store_schedule(schedule)
    _maybe_plan_recompute_relief(gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd)
    global _PERSISTENT_SET_DONE, _LATEST_SCHEDULE
    _PERSISTENT_SET_DONE = False
    if dist.get_rank() == 0:
        log_rank0(f"[{NAME}] Persistent buffers will be selected/allocated in apply() using observed warmup-step peak memory.", enable=True)
    _dump_schedule_if_enabled(schedule)
    return None


def _find_layer_anchors(graph: Graph, ds_id_to_layer: Dict[int, int], L: int) -> Dict[int, Node]:
    anchors: Dict[int, Node] = {}
    for n in graph.nodes:
        if n.target == torch.ops.dc.wait_allgather.default:
            ds_id = _get_ds_id_from_wait(n)
            layer = ds_id_to_layer.get(ds_id)
            if layer is None or layer < 0 or layer >= L:
                continue
            anchors.setdefault(layer, n)
    if len(anchors) == L:
        return anchors

    # Fallback: use allgather nodes if a layer has no wait node (rare).
    for n in graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default:
            ds_id = int(n.args[2])
            layer = ds_id_to_layer.get(ds_id)
            if layer is None or layer < 0 or layer >= L:
                continue
            anchors.setdefault(layer, n)
            if len(anchors) == L:
                break
    return anchors


def _ds_id_to_param_node(graph: Graph, pm: DSGraphParamManager) -> Dict[int, Node]:
    param_name_to_ds_id = pm.ds_ids
    out: Dict[int, Node] = {}
    for n in graph.nodes:
        if n.op != "placeholder":
            continue
        if n.name not in param_name_to_ds_id:
            continue
        out[int(param_name_to_ds_id[n.name])] = n
    return out


def _ds_id_to_target_dtype(graph: Graph, pm: DSGraphParamManager) -> Dict[int, torch.dtype]:
    # Default to param dtype; override with allgather_param(dtype=...) when present.
    out: Dict[int, torch.dtype] = {}

    for param_name, ds_param in pm.params.items():
        ds_id = int(pm.ds_ids[param_name])
        out[ds_id] = ds_param.dtype

    for n in graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default and "dtype" in n.kwargs:
            ds_id = int(n.args[2])
            out[ds_id] = n.kwargs["dtype"]

    return out


def _rewrite_graph_with_layer_comm(graph: Graph, graph_id: int, pm: DSGraphParamManager, mapping: _LayerMapping,
                                  sched: dict, bwd: bool) -> Optional[Graph]:
    L = int(sched["L"])
    K = 2 * L
    launches: Dict[int, List[List[int]]] = {int(k): v for k, v in sched.get("launches", {}).items()}

    keep_ds_ids: Set[int] = set(map(int, sched.get("keep_ds_ids", [])))
    persistent_ds_ids: Set[int] = set(map(int, sched.get("persistent_ds_ids", [])))

    anchors = _find_layer_anchors(graph, mapping.ds_id_to_layer, L)
    if not anchors:
        return None
    anchor_node_to_layer = {n: layer for layer, n in anchors.items()}

    ds_id_to_ag: Dict[int, Node] = {}
    ds_id_to_wait: Dict[int, Node] = {}
    for n in graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default:
            ds_id_to_ag[int(n.args[2])] = n
        elif n.target == torch.ops.dc.wait_allgather.default:
            ds_id_to_wait[int(n.args[2])] = n

    ds_id_to_param = _ds_id_to_param_node(graph, pm)
    ds_id_to_dtype = _ds_id_to_target_dtype(graph, pm)

    new_graph = Graph()
    env: Dict[Node, Node] = {}

    def copy_node(old: Node) -> Node:
        new = new_graph.node_copy(old, lambda n: env[n])
        env[old] = new
        return new

    head_k = 0 if not bwd else int(L)

    def insert_prefetch_for_unified_k(unified_k: int) -> bool:
        inserted = False
        groups = launches.get(int(unified_k), [])
        for gi, ds_ids in enumerate(groups):
            params_new: List[Node] = []
            ds_ids_new: List[int] = []
            dtypes_new: List[torch.dtype] = []
            for ds_id in ds_ids:
                if int(ds_id) in persistent_ds_ids:
                    continue
                pn_old = ds_id_to_param.get(int(ds_id))
                if pn_old is None:
                    continue
                params_new.append(env[pn_old])
                ds_ids_new.append(int(ds_id))
                dtypes_new.append(ds_id_to_dtype.get(int(ds_id), torch.float16))
            if not ds_ids_new:
                continue
            name = f"gls_prefetch_{'bwd' if bwd else 'fwd'}_k{unified_k}_g{gi}"
            new_graph.create_node("call_function",
                                  torch.ops.dc.prefetch_params_fused.default,
                                  args=(graph_id, params_new, ds_ids_new, dtypes_new, True),
                                  name=name)
            inserted = True
        return inserted

    def insert_prefetch_for_layer(layer: int) -> None:
        unified_k = layer if not bwd else (K - 1 - layer)
        # For the first boundary in each graph, we insert prefetch at graph head to overlap
        # with any prefix compute (e.g., embeddings) that happens before the first layer anchor.
        if int(unified_k) == int(head_k):
            return
        insert_prefetch_for_unified_k(int(unified_k))

    def insert_layer_comm(layer: int) -> None:
        ds_ids = [
            int(ds_id) for ds_id in mapping.layer_to_ds_ids[layer]
            if int(ds_id) in ds_id_to_ag and int(ds_id) in ds_id_to_wait
        ]
        for ds_id in sorted(ds_ids):
            ag_old = ds_id_to_ag[ds_id]
            if ag_old not in env:
                copy_node(ag_old)

    moved_ag = set()
    for layer in range(L):
        for ds_id in mapping.layer_to_ds_ids[layer]:
            ds_id = int(ds_id)
            if ds_id in ds_id_to_ag and ds_id in ds_id_to_wait:
                moved_ag.add(ds_id_to_ag[ds_id])

    changed = False
    inserted_head = False
    for n in graph.nodes:
        if n.op == "placeholder":
            copy_node(n)
            continue

        if not inserted_head:
            if insert_prefetch_for_unified_k(int(head_k)):
                changed = True
            inserted_head = True

        if n in anchor_node_to_layer:
            layer = int(anchor_node_to_layer[n])
            insert_prefetch_for_layer(layer)
            insert_layer_comm(layer)
            changed = True

        if n in moved_ag:
            # These comm nodes are re-inserted at the layer anchor.
            continue

        if n.target == torch.ops.dc.release_param.default:
            ds_id = _get_ds_id_from_release(n)
            if ((not bwd) and ds_id in (keep_ds_ids | persistent_ds_ids)) or (bwd and ds_id in persistent_ds_ids):
                env[n] = env[n.args[0]]
                changed = True
                continue

        if n.op == "output":
            copy_node(n)
            continue

        copy_node(n)

    if not changed:
        return None
    new_graph.lint()
    return new_graph


def _collapse_persistent_state_nodes(gm: GraphModule, persistent_ds_ids: Sequence[int], *, elide_releases: bool,
                                     elide_waits: bool) -> Tuple[GraphModule, int, int]:
    persistent_set: Set[int] = set(int(x) for x in (persistent_ds_ids or ()))
    if not persistent_set or (not elide_releases and not elide_waits):
        return gm, 0, 0

    new_graph = Graph()
    env: Dict[Node, Node] = {}
    elided_releases = 0
    elided_waits = 0

    for n in gm.graph.nodes:
        if elide_waits and n.target == torch.ops.dc.wait_allgather.default:
            ds_id = _get_ds_id_from_wait(n)
            if int(ds_id) in persistent_set:
                src = n.args[0]
                if isinstance(src, Node) and src in env:
                    env[n] = env[src]
                    elided_waits += 1
                    continue
        if elide_releases and n.target == torch.ops.dc.release_param.default:
            ds_id = _get_ds_id_from_release(n)
            if int(ds_id) in persistent_set:
                src = n.args[0]
                if isinstance(src, Node) and src in env:
                    env[n] = env[src]
                    elided_releases += 1
                    continue
        new = new_graph.node_copy(n, lambda old: env[old])
        env[n] = new

    if elided_releases <= 0 and elided_waits <= 0:
        return gm, 0, 0
    new_graph.lint()
    gm.graph = new_graph
    return gm, int(elided_releases), int(elided_waits)


def _prune_persistent_prefetch_nodes(gm: GraphModule, persistent_ds_ids: Sequence[int]) -> Tuple[GraphModule, int, int, int]:
    persistent_set: Set[int] = set(int(x) for x in (persistent_ds_ids or ()))
    if not persistent_set:
        return gm, 0, 0, 0

    new_graph = Graph()
    env: Dict[Node, Node] = {}
    dropped_ops = 0
    filtered_ops = 0
    dropped_ds_ids = 0

    for n in gm.graph.nodes:
        if n.target == torch.ops.dc.prefetch_params_fused.default and len(n.args) >= 3:
            params = list(n.args[1])
            ds_ids = [int(x) for x in list(n.args[2])]
            if len(params) == len(ds_ids):
                keep_indices = [i for i, ds_id in enumerate(ds_ids) if int(ds_id) not in persistent_set]
                removed = len(ds_ids) - len(keep_indices)
                if removed > 0 and not n.users:
                    dropped_ds_ids += int(removed)
                    if not keep_indices:
                        dropped_ops += 1
                        continue

                    new_params = [env[p] if isinstance(p, Node) else p for i, p in enumerate(params) if i in keep_indices]
                    new_ds_ids = [int(ds_ids[i]) for i in keep_indices]
                    grouped_arg = bool(n.args[4]) if len(n.args) >= 5 else False
                    if len(n.args) >= 4 and n.args[3] is not None:
                        dtypes = list(n.args[3])
                        if len(dtypes) == len(ds_ids):
                            new_dtypes = [dtypes[i] for i in keep_indices]
                            new_args = (n.args[0], new_params, new_ds_ids, new_dtypes, grouped_arg)
                        else:
                            new_args = (n.args[0], new_params, new_ds_ids, None, grouped_arg)
                    elif len(n.args) >= 5:
                        new_args = (n.args[0], new_params, new_ds_ids, None, grouped_arg)
                    else:
                        new_args = (n.args[0], new_params, new_ds_ids)
                    new = new_graph.create_node(n.op, n.target, args=new_args, kwargs=dict(n.kwargs), name=n.name)
                    env[n] = new
                    filtered_ops += 1
                    continue

        new = new_graph.node_copy(n, lambda old: env[old])
        env[n] = new

    if dropped_ops <= 0 and filtered_ops <= 0:
        return gm, 0, 0, 0
    new_graph.lint()
    gm.graph = new_graph
    return gm, int(dropped_ops), int(filtered_ops), int(dropped_ds_ids)



_CHORUS_PREFETCH_MARGIN = 0.1


def _get_ds_id_from_allgather(node: Node) -> int:
    assert node.target == torch.ops.dc.allgather_param.default
    return int(node.args[2])


def _schedule_chorus_local_prefetch(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                                    create_inputs_fn, mem_budget: float, param_manager: Dict[int, DSGraphParamManager],
                                    bwd: bool, *, fuse_slack: float, max_fuse_size: float,
                                    max_buffered_size: float, grouped_prefetch: bool) -> GraphModule:
    del graph_order, create_inputs_fn, mem_budget, param_manager

    max_mem = get_accelerator().total_memory() * (1 - _CHORUS_PREFETCH_MARGIN)
    vals_to_bcast = torch.tensor([max_mem], device=torch.device(get_accelerator().current_device()))
    dist.all_reduce(vals_to_bcast, dist.ReduceOp.MIN)
    max_mem = vals_to_bcast[0].item()

    mem = profiling_results[graph_id].bwd_mem if bwd else profiling_results[graph_id].fwd_mem
    op_time = profiling_results[graph_id].bwd_time if bwd else profiling_results[graph_id].fwd_time
    tensor_sizes = profiling_results[graph_id].bwd_tensor_sizes if bwd else profiling_results[graph_id].fwd_tensor_sizes

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, _delta, peak in mem}
    _time_dict = {name: (device_time, wall_time) for name, device_time, wall_time in op_time}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

    def tensor_size_for_node(node: Node) -> int:
        if node.name in tensor_size_dict:
            return int(tensor_size_dict[node.name])
        meta = getattr(node, "meta", {}) or {}
        if "tensor_size" in meta:
            return int(meta["tensor_size"])
        if "alloc_mem" in meta:
            return int(meta["alloc_mem"])
        return 0

    graph = gm.graph
    total_param_size = sum(tensor_size_for_node(n) for n in graph.nodes
                           if n.target == torch.ops.dc.allgather_param.default)
    log_rank0(
        f"[{NAME}] Chorus local_prefetch graph_id={graph_id} bwd={bwd} max_mem={max_mem} "
        f"available_memory={get_accelerator().available_memory()} "
        f"memory_allocated={get_accelerator().memory_allocated()} "
        f"max_allocated={get_accelerator().max_memory_allocated()} total_param_size={total_param_size} "
        f"margin={_CHORUS_PREFETCH_MARGIN}",
        enable=True,
    )

    prev_mem = 0
    prev_peak = 0
    for node in graph.nodes:
        if node.name in mem_dict:
            prev_mem = mem_dict[node.name][0]
            prev_peak = mem_dict[node.name][1]
        else:
            mem_dict[node.name] = (prev_mem, prev_peak)

    comm_predictor = create_predictor()
    order_rev = list(reversed(graph.nodes))
    new_order_rev = []
    prefetch_ags = []
    prefetch_ag_groups = []
    ag_tensor_size_sum = 0

    for i, node in enumerate(order_rev):
        if node.op != "placeholder":
            assert i < len(order_rev) - 1
            assert node.name in mem_dict
            next_node = order_rev[i + 1]
            _next_alloc_mem, next_peak = mem_dict[next_node.name]

            while next_peak + ag_tensor_size_sum > max_mem or ag_tensor_size_sum > max_buffered_size:
                if len(prefetch_ag_groups) > 0:
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum(tensor_size_for_node(ag_node) for ag_node in fused_ag_nodes)
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)
                    assert len(fused_ag_nodes) > 0
                elif len(prefetch_ags) > 0:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                else:
                    break

            if node.target == torch.ops.dc.allgather_param.default:
                current_ag_size = sum(tensor_size_for_node(ag_node) for ag_node in prefetch_ags)
                node_size = tensor_size_for_node(node)
                pred_time_current = comm_predictor(current_ag_size)
                pred_time_next = comm_predictor(node_size)
                pred_time_fused = comm_predictor(current_ag_size + node_size)

                do_fuse = max(pred_time_current, pred_time_next) * float(fuse_slack) > pred_time_fused and (
                    current_ag_size + node_size) < float(max_fuse_size)

                if len(prefetch_ags) > 0 and not do_fuse:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                prefetch_ags.append(node)
                ag_tensor_size_sum += node_size

        new_order_rev.append(node)

        if (node.op != "placeholder"
                and node.target != torch.ops.dc.reload_parameter) and order_rev[i + 1].op == "placeholder":
            for ag_group in prefetch_ag_groups:
                assert len(ag_group) > 0
                new_order_rev.append(ag_group)
                total_ag_tensor_size = sum(tensor_size_for_node(ag_node) for ag_node in ag_group)
                ag_tensor_size_sum -= total_ag_tensor_size
            if len(prefetch_ags) > 0:
                new_order_rev.append(prefetch_ags)
                ag_tensor_size_sum -= sum(tensor_size_for_node(ag_node) for ag_node in prefetch_ags)
            assert ag_tensor_size_sum == 0

        assert ag_tensor_size_sum >= 0

    new_graph = Graph()
    env: Dict[str, Node] = {}
    prefetch_groups_inserted = 0
    prefetch_ds_ids_inserted = 0
    for node in reversed(new_order_rev):
        if isinstance(node, Node):
            new_node = new_graph.node_copy(node, lambda n: env[n.name])
            env[node.name] = new_node
        else:
            param_nodes = [ag_node.args[0] for ag_node in node]
            param_nodes_copy = [env[param_node.name] for param_node in param_nodes]
            ds_ids = [_get_ds_id_from_allgather(ag_node) for ag_node in node]
            prefetch_args = (graph_id, param_nodes_copy, ds_ids)
            if bool(grouped_prefetch):
                prefetch_args = (graph_id, param_nodes_copy, ds_ids, None, True)
            new_graph.call_function(torch.ops.dc.prefetch_params_fused.default, args=prefetch_args)
            prefetch_groups_inserted += 1
            prefetch_ds_ids_inserted += len(ds_ids)

    log_rank0(
        f"[{NAME}] Chorus local_prefetch summary graph_id={graph_id} bwd={bwd} "
        f"prefetch_calls={prefetch_groups_inserted} prefetch_ds_ids={prefetch_ds_ids_inserted} "
        f"fuse_slack={float(fuse_slack)} max_fuse_size={float(max_fuse_size)} "
        f"max_buffered_size={float(max_buffered_size)} grouped_prefetch={bool(grouped_prefetch)}",
        enable=True,
    )
    new_graph.lint()
    gm.graph = new_graph
    return gm

def _apply_low_comm_local_prefetch(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                                   create_inputs_fn, mem_budget: float, param_manager: Dict[int, DSGraphParamManager],
                                   bwd: bool, sched: dict, cfg: _SchedulerConfig) -> GraphModule:
    gm = _schedule_chorus_local_prefetch(
        gm,
        graph_id,
        graph_order,
        profiling_results,
        create_inputs_fn,
        mem_budget,
        param_manager,
        bwd,
        fuse_slack=float(cfg.low_comm_prefetch_fuse_slack),
        max_fuse_size=float(cfg.low_comm_prefetch_fuse_max_bytes),
        max_buffered_size=float(cfg.low_comm_prefetch_buffer_max_bytes),
        grouped_prefetch=True,
    )
    if bool(cfg.low_comm_elide_persistent_prefetches) and bool(bwd):
        # Forward prefetch refreshes persistent buffers after optimizer updates.
        # Backward can reuse the forward-refreshed persistent buffers, so only
        # backward persistent-prefetch calls are redundant in training.
        gm, dropped_prefetch_ops, filtered_prefetch_ops, dropped_prefetch_ds_ids = _prune_persistent_prefetch_nodes(
            gm, sched.get("persistent_ds_ids", []))
        log_rank0(
            f"[{NAME}] Low-comm persistent-prefetch pruning graph_id={graph_id} bwd={bwd} "
            f"dropped_prefetch_ops={dropped_prefetch_ops} filtered_prefetch_ops={filtered_prefetch_ops} "
            f"dropped_prefetch_ds_ids={dropped_prefetch_ds_ids}",
            enable=True,
        )
    if bool(cfg.low_comm_elide_persistent_releases) or bool(cfg.low_comm_elide_persistent_waits):
        gm, elided_releases, elided_waits = _collapse_persistent_state_nodes(
            gm,
            sched.get("persistent_ds_ids", []),
            elide_releases=bool(cfg.low_comm_elide_persistent_releases),
            elide_waits=bool(cfg.low_comm_elide_persistent_waits),
        )
        log_rank0(
            f"[{NAME}] Low-comm persistent-state collapse graph_id={graph_id} bwd={bwd} "
            f"elided_release_ops={elided_releases} elided_wait_ops={elided_waits}",
            enable=True,
        )
    return gm

def apply(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
          mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool) -> GraphModule:
    global _PERSISTENT_SET_DONE, _LATEST_SCHEDULE

    if _LATEST_SCHEDULE is None or _LAYER_MAPPING is None or _CFG is None:
        return None

    sched = _LATEST_SCHEDULE
    mapping = _LAYER_MAPPING

    if sched.get("mapping_hash") != mapping.mapping_hash:
        raise RuntimeError(f"[{NAME}] schedule mapping_hash mismatch: schedule={sched.get('mapping_hash')} local={mapping.mapping_hash}")

    L = int(sched["L"])
    if L <= 0:
        return None

    pm = param_manager.get(graph_id)
    if pm is None:
        return None

    # If enabled, select + allocate persistent buffers using observed runtime
    # peak memory (warmup steps), not compile-time peak memory.
    if not _PERSISTENT_SET_DONE:
        # This codepath runs inside the torch.compile backend and can be reached
        # by multiple graphs (and potentially multiple threads). Ensure the
        # persistent selection + collectives run exactly once per process.
        with _PERSISTENT_SET_LOCK:
            if _PERSISTENT_SET_DONE:
                pass
            else:
                meta = sched.get("meta", {})

                device = get_accelerator().device(get_accelerator().current_device())

                # Use the maximum observed per-step CUDA peak across ranks.
                observed_peak_alloc = int(get_tracked_step_peak_memory_bytes())
                observed_peak_reserved = int(get_tracked_step_peak_reserved_memory_bytes())
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if unset_fake_temporarily is None:
                        vals = torch.tensor([observed_peak_alloc, observed_peak_reserved], device=device, dtype=torch.int64)
                        dist.all_reduce(vals, dist.ReduceOp.MAX)
                        observed_peak_alloc = int(vals[0].item())
                        observed_peak_reserved = int(vals[1].item())
                    else:
                        with unset_fake_temporarily():
                            vals = torch.tensor([observed_peak_alloc, observed_peak_reserved], device=device, dtype=torch.int64)
                            dist.all_reduce(vals, dist.ReduceOp.MAX)
                            observed_peak_alloc = int(vals[0].item())
                            observed_peak_reserved = int(vals[1].item())
                observed_peak = max(int(observed_peak_alloc), int(observed_peak_reserved))
                profile_peak = int(meta.get("peak_mem_bytes", 0))
                total_mem = int(meta.get("total_mem_bytes", _min_total_mem_bytes_across_ranks()))
                safe_total = int(total_mem)
                allocator_margin_bytes = int(meta.get("allocator_margin_bytes", _allocator_margin_bytes(total_mem)))
                # run_opt_passes() calls empty_cache() before invoking this
                # pass, so the current reserved-vs-alloc gap is a much better
                # approximation of the allocator floor than the warmup-step
                # peak reserved memory.
                accelerator = get_accelerator()
                device_index = int(accelerator.current_device())
                current_alloc = int(accelerator.memory_allocated(device_index) or 0)
                current_reserved = int(accelerator.memory_reserved(device_index) or 0)
                current_allocator_floor_slack = max(0, int(current_reserved) - int(current_alloc))
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if unset_fake_temporarily is None:
                        vals = torch.tensor([current_alloc, current_reserved, current_allocator_floor_slack],
                                            device=device,
                                            dtype=torch.int64)
                        dist.all_reduce(vals, dist.ReduceOp.MAX)
                        current_alloc = int(vals[0].item())
                        current_reserved = int(vals[1].item())
                        current_allocator_floor_slack = int(vals[2].item())
                    else:
                        with unset_fake_temporarily():
                            vals = torch.tensor([current_alloc, current_reserved, current_allocator_floor_slack],
                                                device=device,
                                                dtype=torch.int64)
                            dist.all_reduce(vals, dist.ReduceOp.MAX)
                            current_alloc = int(vals[0].item())
                            current_reserved = int(vals[1].item())
                            current_allocator_floor_slack = int(vals[2].item())

                # Budget persistent buffers against the observed live tensor
                # peak, plus the allocator floor still present after the
                # backend-level empty_cache() before apply().
                baseline_live_peak = max(int(observed_peak_alloc), int(profile_peak))
                baseline_peak = int(baseline_live_peak) + int(current_allocator_floor_slack)

                # Non-torch memory (CUDA context, NCCL, other processes, allocator bypass) reduces
                # the effective budget available to the caching allocator.
                non_torch_bytes = int(_estimate_non_torch_mem_bytes())
                if dist.is_initialized() and dist.get_world_size() > 1:
                    if unset_fake_temporarily is None:
                        vals = torch.tensor([non_torch_bytes], device=device, dtype=torch.int64)
                        dist.all_reduce(vals, dist.ReduceOp.MAX)
                        non_torch_bytes = int(vals[0].item())
                    else:
                        with unset_fake_temporarily():
                            vals = torch.tensor([non_torch_bytes], device=device, dtype=torch.int64)
                            dist.all_reduce(vals, dist.ReduceOp.MAX)
                            non_torch_bytes = int(vals[0].item())

                safe_total_for_torch = max(0, int(safe_total) - int(non_torch_bytes))
                conservative_persistent_budget_bytes = max(
                    0, int(safe_total_for_torch) - int(baseline_peak) - int(allocator_margin_bytes))

                backend_policy = meta.get("comm_backend_policy", {}) or {}
                persistent_budget_mode = str(getattr(_CFG, "low_comm_persistent_budget_mode", "selective")).strip().lower()
                selective_persistent_budget_bytes = 0
                if str(backend_policy.get("mode", "generic")) == "low_comm" and persistent_budget_mode != "conservative":
                    usable_fraction = max(0.0, min(1.0, float(getattr(_CFG, "low_comm_persistent_usable_fraction", 0.90))))
                    selective_persistent_budget_bytes = max(
                        0, int(float(total_mem) * float(usable_fraction)) - int(baseline_live_peak))
                    if persistent_budget_mode == "max":
                        max_persistent_budget_bytes = max(0, int(safe_total) - int(baseline_live_peak))
                        persistent_budget_bytes = max(int(conservative_persistent_budget_bytes),
                                                      int(selective_persistent_budget_bytes),
                                                      int(max_persistent_budget_bytes))
                    else:
                        persistent_budget_bytes = max(int(conservative_persistent_budget_bytes),
                                                      int(selective_persistent_budget_bytes))
                else:
                    persistent_budget_bytes = int(conservative_persistent_budget_bytes)

                # Load candidate ranking and sizes from the planned schedule when available.
                cand_list = meta.get("persistent_candidates", [])
                cand_sizes = meta.get("persistent_candidate_sizes", {})
                cand_values = meta.get("persistent_candidate_allgather_ms", {})
                ds_id_to_size = {int(k): int(v) for k, v in cand_sizes.items()}
                ds_id_to_value = {int(k): max(0.0, float(v)) for k, v in cand_values.items()}
                candidates = [int(x) for x in cand_list if int(x) in ds_id_to_size]

                if candidates and not ds_id_to_value:
                    ds_id_to_allgather_ms = _build_ds_id_to_allgather_ms(profiling_results)
                    ds_id_to_value = {
                        int(ds_id): max(0.0, float(ds_id_to_allgather_ms.get(int(ds_id), 0.0))) for ds_id in candidates
                    }

                if not candidates:
                    # Fallback: derive candidates from whatever profiling results are available
                    # in the current compilation step.
                    ds_id_to_size = _build_ds_id_to_size_bytes(graph_id, profiling_results, param_manager)
                    ds_id_to_allgather_ms = _build_ds_id_to_allgather_ms(profiling_results)
                    ds_id_to_value = {int(k): max(0.0, float(v)) for k, v in ds_id_to_allgather_ms.items()}
                    candidates = _rank_persistent_candidates(ds_id_to_size, ds_id_to_allgather_ms)

                state_utility_meta = {"status": "skipped", "reason": "backend_not_low_comm"}
                if str(backend_policy.get("mode", "generic")) == "low_comm":
                    state_op_ms = max(0.0, float(getattr(_CFG, "low_comm_state_op_ms", 0.0)))
                    value_mode = str(getattr(_CFG, "low_comm_persistent_value_mode", "comm_state")).strip().lower()
                    comm_value_weight = max(0.0, float(getattr(_CFG, "low_comm_comm_value_weight", 1.0)))
                    if state_op_ms > 0.0:
                        state_counts = _build_ds_id_state_op_counts(profiling_results)
                        state_value_total = 0.0
                        comm_value_total_before = float(sum(float(ds_id_to_value.get(int(ds_id), 0.0)) for ds_id in candidates))
                        touched = 0
                        for ds_id in candidates:
                            ds_id = int(ds_id)
                            added = float(state_op_ms) * float(state_counts.get(ds_id, 0))
                            old_value = max(0.0, float(ds_id_to_value.get(ds_id, 0.0)))
                            if value_mode in {"event_density", "state_density", "events"}:
                                ds_id_to_value[ds_id] = float(comm_value_weight) * old_value + max(0.0, added)
                            elif added > 0.0:
                                ds_id_to_value[ds_id] = old_value + added
                            if added > 0.0 or value_mode in {"event_density", "state_density", "events"}:
                                state_value_total += float(added)
                                touched += 1
                        comm_value_total_after = float(sum(float(ds_id_to_value.get(int(ds_id), 0.0)) for ds_id in candidates))
                        state_utility_meta = {
                            "status": "applied",
                            "value_mode": str(value_mode),
                            "comm_value_weight": float(comm_value_weight),
                            "state_op_ms": float(state_op_ms),
                            "candidate_ds_ids": int(touched),
                            "added_value_ms": float(state_value_total),
                            "comm_value_before_ms": float(comm_value_total_before),
                            "objective_value_after_ms": float(comm_value_total_after),
                        }
                    else:
                        state_utility_meta = {"status": "skipped", "reason": "zero_state_op_ms"}

                ds_id_to_layer = {int(k): int(v) for k, v in meta.get("ds_id_to_layer", {}).items()}
                rebalance_meta = {"status": "skipped", "reason": "not_attempted"}
                if _CFG is not None:
                    sched, rebalance_meta = _maybe_rebalance_low_comm_schedule(sched,
                                                                               _CFG,
                                                                               persistent_budget_bytes,
                                                                               candidates,
                                                                               ds_id_to_size,
                                                                               ds_id_to_value,
                                                                               ds_id_to_layer,
                                                                               profiling_results,
                                                                               graph_id)
                    _LATEST_SCHEDULE = sched
                    meta = sched.get("meta", {})
                    ds_id_to_layer = {int(k): int(v) for k, v in meta.get("ds_id_to_layer", {}).items()}

                persistent_ds_ids, persistent_mem_bytes, persistent_value_ms, persistent_select_meta = _select_persistent_ds_ids_block_safe(
                    candidates, ds_id_to_size, persistent_budget_bytes, sched, ds_id_to_layer, ds_id_to_value)

                post_persistent_schedule_meta = {"status": "skipped", "reason": "no_persistent_ids"}
                if persistent_ds_ids and _CFG is not None:
                    comm_meta = sched.get("comm_model", {}) or {}
                    comm_model = _CommModel(alpha_ms=float(comm_meta.get("alpha_ms", 0.0)),
                                            beta_ms_per_byte=float(comm_meta.get("beta_ms_per_byte", 0.0)))
                    old_prefetch_calls = int((sched.get("meta", {}) or {}).get("prefetch_calls", 0))
                    post_sched = _schedule_with_keep_layers(sched,
                                                            set(map(int, sched.get("keep_layers", []))),
                                                            _CFG,
                                                            comm_model,
                                                            ds_id_to_size,
                                                            ds_id_to_layer,
                                                            profiling_results,
                                                            graph_id,
                                                            persistent_set=set(map(int, persistent_ds_ids)))
                    if post_sched is not None:
                        sched = post_sched
                        _LATEST_SCHEDULE = sched
                        meta = sched.get("meta", {})
                        new_prefetch_calls = int(meta.get("prefetch_calls", 0))
                        post_persistent_schedule_meta = {
                            "status": "applied",
                            "old_prefetch_calls": int(old_prefetch_calls),
                            "new_prefetch_calls": int(new_prefetch_calls),
                        }
                    else:
                        post_persistent_schedule_meta = {"status": "skipped", "reason": "candidate_infeasible"}
                if dist.get_rank() == 0:
                    log_rank0(
                        f"[{NAME}] Deferred persistent selection: observed_peak_alloc_bytes={int(observed_peak_alloc)} "
                        f"observed_peak_reserved_bytes={int(observed_peak_reserved)} "
                        f"current_alloc_bytes={int(current_alloc)} "
                        f"current_reserved_bytes={int(current_reserved)} "
                        f"current_allocator_floor_slack_bytes={int(current_allocator_floor_slack)} "
                        f"non_torch_bytes={int(non_torch_bytes)} "
                        f"profile_peak_bytes={int(profile_peak)} baseline_live_peak_bytes={int(baseline_live_peak)} "
                        f"baseline_peak_bytes={int(baseline_peak)} "
                        f"allocator_margin_bytes={int(allocator_margin_bytes)} "
                        f"persistent_budget_mode={persistent_budget_mode} "
                        f"conservative_persistent_budget_bytes={int(conservative_persistent_budget_bytes)} "
                        f"selective_persistent_budget_bytes={int(selective_persistent_budget_bytes)} "
                        f"persistent_budget_bytes={int(persistent_budget_bytes)} "
                        f"rebalance_status={rebalance_meta.get('status', 'unknown')} "
                        f"rebalance_reason={rebalance_meta.get('reason', '')} "
                        f"post_persistent_schedule_status={post_persistent_schedule_meta.get('status', 'unknown')} "
                        f"state_utility_status={state_utility_meta.get('status', 'unknown')} "
                        f"state_value_mode={state_utility_meta.get('value_mode', 'none')} "
                        f"state_utility_added_ms={float(state_utility_meta.get('added_value_ms', 0.0)):.4f} "
                        f"state_objective_after_ms={float(state_utility_meta.get('objective_value_after_ms', 0.0)):.4f} "
                        f"selected_ds_ids={len(persistent_ds_ids)} "
                        f"persistent_mem_bytes={int(persistent_mem_bytes)} selected_value_ms={float(persistent_value_ms):.4f} "
                        f"selection_method={persistent_select_meta.get('method', 'unknown')} "
                        f"selection_status={persistent_select_meta.get('status', 'unknown')}",
                        enable=True,
                    )

                # Keep meta consistent across ranks (selection is deterministic).
                persistent_mem_bytes = int(sum(int(ds_id_to_size.get(int(ds_id), 0)) for ds_id in persistent_ds_ids))

                sched["persistent_ds_ids"] = [int(x) for x in persistent_ds_ids]
                meta["observed_peak_mem_alloc_bytes"] = int(observed_peak_alloc)
                meta["observed_peak_mem_reserved_bytes"] = int(observed_peak_reserved)
                meta["observed_peak_mem_bytes"] = int(observed_peak)
                meta["current_alloc_bytes"] = int(current_alloc)
                meta["current_reserved_bytes"] = int(current_reserved)
                meta["current_allocator_floor_slack_bytes"] = int(current_allocator_floor_slack)
                meta["non_torch_mem_bytes"] = int(non_torch_bytes)
                meta["allocator_margin_bytes"] = int(allocator_margin_bytes)
                meta["baseline_live_peak_bytes"] = int(baseline_live_peak)
                meta["baseline_peak_bytes"] = int(baseline_peak)
                meta["persistent_budget_mode"] = str(persistent_budget_mode)
                meta["conservative_persistent_budget_bytes"] = int(conservative_persistent_budget_bytes)
                meta["selective_persistent_budget_bytes"] = int(selective_persistent_budget_bytes)
                meta["persistent_budget_bytes"] = int(persistent_budget_bytes)
                meta["low_comm_rebalance"] = rebalance_meta
                meta["low_comm_state_utility"] = state_utility_meta
                meta["post_persistent_schedule"] = post_persistent_schedule_meta
                meta["persistent_mem_bytes"] = int(persistent_mem_bytes)
                meta["persistent_value_ms"] = float(persistent_value_ms)
                meta["persistent_selection"] = {
                    str(k): (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in persistent_select_meta.items()
                }
                meta["persistent_deferred"] = False
                sched["meta"] = meta
                predicted_graph_peak_mem_bytes = _predicted_peak_mem_bytes_from_schedule(sched, ds_id_to_size,
                                                                                         ds_id_to_layer)
                predicted_peak_baseline_offset_bytes = max(0, int(observed_peak_alloc) - int(profile_peak))
                predicted_total_peak_mem_bytes = int(predicted_graph_peak_mem_bytes) + int(
                    predicted_peak_baseline_offset_bytes)
                meta["predicted_graph_peak_mem_bytes"] = int(predicted_graph_peak_mem_bytes)
                meta["predicted_graph_peak_mem_gb"] = float(predicted_graph_peak_mem_bytes) / float(1024**3)
                meta["predicted_peak_baseline_offset_bytes"] = int(predicted_peak_baseline_offset_bytes)
                meta["predicted_peak_baseline_offset_gb"] = float(predicted_peak_baseline_offset_bytes) / float(1024**3)
                meta["predicted_total_peak_mem_bytes"] = int(predicted_total_peak_mem_bytes)
                meta["predicted_total_peak_mem_gb"] = float(predicted_total_peak_mem_bytes) / float(1024**3)
                meta["predicted_peak_mem_bytes"] = int(predicted_total_peak_mem_bytes)
                meta["predicted_peak_mem_gb"] = float(predicted_total_peak_mem_bytes) / float(1024**3)
                meta["predicted_peak_memory_scope"] = "total_cuda_allocated"
                meta["predicted_peak_includes_persistent"] = True
                sched["meta"] = meta
                sched["schedule_hash"] = _schedule_hash(sched)

                _maybe_set_persistent(persistent_ds_ids)
                _PERSISTENT_SET_DONE = True

    meta = sched.get("meta", {}) or {}
    backend_policy = meta.get("comm_backend_policy", {}) or {}
    use_local_prefetch = (
        str(backend_policy.get("mode", "generic")) == "low_comm"
        and str(_CFG.low_comm_graph_rewrite_mode) in {"local_prefetch", "prefetch"}
        and not sched.get("keep_layers")
        and not sched.get("keep_ds_ids")
    )
    if use_local_prefetch:
        log_rank0(
            f"[{NAME}] Using low-comm graph rewrite mode: {str(_CFG.low_comm_graph_rewrite_mode)} graph_id={graph_id} bwd={bwd}",
            enable=True)
        return _apply_low_comm_local_prefetch(gm, graph_id, graph_order, profiling_results, create_inputs_fn,
                                             mem_budget, param_manager, bwd, sched, _CFG)

    new_graph = _rewrite_graph_with_layer_comm(gm.graph, graph_id, pm, mapping, sched, bwd=bwd)
    if new_graph is None:
        return None
    gm.graph = new_graph
    if bool(_CFG.low_comm_elide_persistent_releases) or bool(_CFG.low_comm_elide_persistent_waits):
        gm, elided_releases, elided_waits = _collapse_persistent_state_nodes(
            gm,
            sched.get("persistent_ds_ids", []),
            elide_releases=bool(_CFG.low_comm_elide_persistent_releases),
            elide_waits=bool(_CFG.low_comm_elide_persistent_waits),
        )
        log_rank0(
            f"[{NAME}] Persistent-state collapse graph_id={graph_id} bwd={bwd} "
            f"elided_release_ops={elided_releases} elided_wait_ops={elided_waits}",
            enable=True,
        )
    return gm
