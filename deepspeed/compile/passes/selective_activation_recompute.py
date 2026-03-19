# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..graph_param import DSGraphParamManager
from ..util import (get_tracked_step_peak_memory_bytes, get_tracked_step_peak_reserved_memory_bytes, log_rank0)

NAME = "selective_activation_recompute"
_DEFAULT_PRESSURE_RATIO: float = 0.9
_PRESSURE_RATIO: float = _DEFAULT_PRESSURE_RATIO

try:
    from torch._subclasses.fake_tensor import unset_fake_temporarily
except Exception:
    unset_fake_temporarily = None


@dataclass(frozen=True)
class _RecomputeConfig:
    layer_regexes: Tuple[str, ...]
    module_regexes: Tuple[str, ...]
    dump_plan: bool
    dump_dir: Optional[str]


@dataclass(frozen=True)
class _LayerMapping:
    mapping_hash: str
    layer_indices: Tuple[int, ...]
    layer_to_name: Dict[int, str]
    ds_id_to_layer: Dict[int, int]


@dataclass(frozen=True)
class _LayerCandidate:
    layer_idx: int
    saved_bytes: int
    recompute_cost_ms: float


@dataclass(frozen=True)
class _BlockProfile:
    k: int
    phase: str
    layer_idx: int
    peak_bytes: int


_CFG: Optional[_RecomputeConfig] = None
_LAYER_MAPPING: Optional[_LayerMapping] = None
_LAYER_MODULES: Dict[int, torch.nn.Module] = {}
_NON_CANDIDATE_GC_MODULES: List[torch.nn.Module] = []
_SELECTED_LAYERS: Set[int] = set()
_LATEST_PLAN: Optional[dict] = None
_APPLY_LOGGED: bool = False


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plan_hash(plan_wo_hash: dict) -> str:
    plan = dict(plan_wo_hash)
    plan.pop("plan_hash", None)
    return _sha256_hex(_canonical_json(plan).encode("utf-8"))


def _infer_layer_idx(name: str, regexes: Sequence[str]) -> Optional[int]:
    for pattern in regexes:
        m = re.search(pattern, name)
        if m is None:
            continue
        try:
            return int(m.group(1))
        except Exception:
            continue
    return None


def _contains_pass(schedule, suffix: str) -> bool:
    if schedule is None:
        return False
    for _, passes in schedule:
        for p in passes:
            mod = getattr(p, "__module__", "")
            if mod.endswith(suffix):
                return True
    return False


def _make_checkpoint_func():
    # Always use non-reentrant checkpointing.
    return functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)


def _max_int64_across_ranks(v: int) -> int:
    v = int(v)
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return v

    device = get_accelerator().device(get_accelerator().current_device())
    if unset_fake_temporarily is None:
        vals = torch.tensor([v], device=device, dtype=torch.int64)
        dist.all_reduce(vals, dist.ReduceOp.MAX)
        return int(vals[0].item())

    with unset_fake_temporarily():
        vals = torch.tensor([v], device=device, dtype=torch.int64)
        dist.all_reduce(vals, dist.ReduceOp.MAX)
        return int(vals[0].item())


def _runtime_tracked_peak_alloc_bytes() -> int:
    # Max across ranks for safety (plan must be feasible for all ranks).
    return _max_int64_across_ranks(get_tracked_step_peak_memory_bytes())


def _runtime_tracked_peak_reserved_bytes() -> int:
    return _max_int64_across_ranks(get_tracked_step_peak_reserved_memory_bytes())


def _effective_budget_bytes(budget_bytes: int, baseline_peak_bytes: int) -> Tuple[int, int]:
    """Return (effective_budget_bytes, reserve_bytes) for planning.

    We intentionally keep a safety reserve to avoid entering allocator high-pressure
    regimes where disabling checkpointing can *hurt* throughput and/or trigger OOM
    during compile-time profiling.
    """
    budget_bytes = int(budget_bytes)
    baseline_peak_bytes = int(baseline_peak_bytes)

    ratio = float(_PRESSURE_RATIO)
    if not (0.0 < ratio <= 1.0):
        ratio = _DEFAULT_PRESSURE_RATIO
    reserve_bytes = max(0, int(budget_bytes - int(budget_bytes * ratio)))

    # Ensure baseline fits with a bit of headroom; otherwise reduce reserve.
    min_headroom = 256 * 1024 * 1024
    max_reserve = max(0, int(budget_bytes) - int(baseline_peak_bytes) - int(min_headroom))
    reserve_bytes = min(int(reserve_bytes), int(max_reserve))

    effective = max(0, int(budget_bytes) - int(reserve_bytes))
    return int(effective), int(reserve_bytes)


def _has_apply_pass(scheduled_passes: Optional[Sequence[object]]) -> bool:
    if not scheduled_passes:
        return False
    for p in scheduled_passes:
        if getattr(p, "__name__", "") != "apply":
            continue
        if getattr(p, "__module__", "").endswith(".selective_activation_recompute"):
            return True
    return False


def _set_layer_checkpoint_selection(selected_layers: Iterable[int]) -> None:
    global _SELECTED_LAYERS

    selected = {int(x) for x in selected_layers}
    _SELECTED_LAYERS = selected
    if _CFG is None:
        return

    checkpoint_fn = _make_checkpoint_func()

    for layer_idx, module in _LAYER_MODULES.items():
        module.gradient_checkpointing = int(layer_idx) in selected
        module._gradient_checkpointing_func = checkpoint_fn

    non_candidate_enabled = bool(selected)
    for module in _NON_CANDIDATE_GC_MODULES:
        module.gradient_checkpointing = non_candidate_enabled
        module._gradient_checkpointing_func = checkpoint_fn


def _refine_plan_with_runtime_peak(plan: dict) -> dict:
    """Refine an initial plan using runtime-observed peak memory from warmup steps.

    The step-0 plan is computed from compile-time graphs and can be overly optimistic
    about peak memory (it misses runtime overhead and allocator behavior). Before
    switching to the selective plan we re-solve using the tracked runtime peak so
    the selected plan is actually feasible and avoids high memory pressure.
    """
    if plan.get("runtime_refined", False):
        return plan

    budget_bytes = int(plan.get("budget_bytes", 0))
    runtime_peak_alloc = int(_runtime_tracked_peak_alloc_bytes())
    runtime_peak_reserved = int(_runtime_tracked_peak_reserved_bytes())

    # If we have no runtime data yet, keep the original plan.
    if runtime_peak_alloc <= 0:
        plan["runtime_refined"] = False
        return plan

    candidates_meta = plan.get("candidates", [])
    block_meta = plan.get("block_profiles", [])
    if not candidates_meta or not block_meta:
        plan["runtime_refined"] = False
        return plan

    candidates = [
        _LayerCandidate(
            layer_idx=int(c.get("layer_idx", 0)),
            saved_bytes=int(c.get("saved_bytes", 0)),
            recompute_cost_ms=float(c.get("recompute_cost_ms", 0.0)),
        ) for c in candidates_meta
    ]

    # Calibrate block peaks upward so constraints match the runtime scale.
    compile_peak_max = max(int(b.get("peak_bytes", 0)) for b in block_meta)
    overhead = max(0, int(runtime_peak_alloc) - int(compile_peak_max))
    blocks = [
        _BlockProfile(
            k=int(b.get("k", 0)),
            phase=str(b.get("phase", "")),
            layer_idx=int(b.get("layer_idx", 0)),
            peak_bytes=int(b.get("peak_bytes", 0)) + int(overhead),
        ) for b in block_meta
    ]

    effective_budget, reserve_bytes = _effective_budget_bytes(budget_bytes, runtime_peak_alloc)
    disabled_layers, feasible, solver = _solve_selection(candidates, blocks, effective_budget)

    active_layers = [int(c.layer_idx) for c in candidates]
    disabled_set = set(int(x) for x in disabled_layers)
    checkpointed_layers = [int(layer) for layer in active_layers if int(layer) not in disabled_set]

    if not feasible:
        # Fall back to full checkpointing (safe baseline).
        disabled_layers = []
        checkpointed_layers = sorted(active_layers)
        solver = f"{solver}_fallback_full"
        feasible = True

    plan["checkpointed_layers"] = [int(x) for x in checkpointed_layers]
    plan["disabled_layers"] = [int(x) for x in disabled_layers]
    plan["selected_layers"] = [int(x) for x in checkpointed_layers]
    plan["solver"] = str(solver)
    plan["feasible"] = bool(feasible)
    plan["runtime_refined"] = True
    plan["runtime_peak_alloc_bytes"] = int(runtime_peak_alloc)
    plan["runtime_peak_reserved_bytes"] = int(runtime_peak_reserved)
    plan["compile_peak_max_bytes"] = int(compile_peak_max)
    plan["peak_overhead_bytes"] = int(overhead)
    plan["effective_budget_bytes"] = int(effective_budget)
    plan["pressure_reserve_bytes"] = int(reserve_bytes)
    plan["plan_hash"] = _plan_hash(plan)
    return plan


def maybe_apply_before_recompile(scheduled_passes: Optional[Sequence[object]]) -> None:
    global _APPLY_LOGGED

    if _LATEST_PLAN is None or _APPLY_LOGGED or not _has_apply_pass(scheduled_passes):
        return

    # Refine using warmup runtime peak memory before applying.
    _refine_plan_with_runtime_peak(_LATEST_PLAN)

    _APPLY_LOGGED = True
    target_layers = {int(x) for x in _LATEST_PLAN.get("checkpointed_layers", _LATEST_PLAN.get("selected_layers", []))}
    _set_layer_checkpoint_selection(target_layers)
    log_rank0(
        f"[{NAME}] Applied selective recompute plan before recompile: checkpointed_layers={_LATEST_PLAN.get('checkpointed_layers', _LATEST_PLAN.get('selected_layers', []))} disabled_layers={_LATEST_PLAN.get('disabled_layers', [])} runtime_peak_alloc_bytes={_LATEST_PLAN.get('runtime_peak_alloc_bytes', 0)} effective_budget_bytes={_LATEST_PLAN.get('effective_budget_bytes', _LATEST_PLAN.get('budget_bytes', 0))} reserve_bytes={_LATEST_PLAN.get('pressure_reserve_bytes', 0)}",
        enable=True,
    )


def _last_backward_graph_id(graph_order: List[Tuple[int, bool]]) -> Optional[int]:
    last = None
    for g_id, needs_bwd in graph_order:
        if needs_bwd:
            last = g_id
            break
    return last


def _min_total_mem_bytes_across_ranks() -> int:
    total_mem = int(get_accelerator().total_memory())
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return total_mem

    device = get_accelerator().device(get_accelerator().current_device())
    if unset_fake_temporarily is None:
        vals = torch.tensor([total_mem], device=device, dtype=torch.int64)
        dist.all_reduce(vals, dist.ReduceOp.MIN)
        return int(vals[0].item())

    with unset_fake_temporarily():
        vals = torch.tensor([total_mem], device=device, dtype=torch.int64)
        dist.all_reduce(vals, dist.ReduceOp.MIN)
        return int(vals[0].item())


def _budget_bytes() -> int:
    # Use total device memory as the budget; additional safety is handled by
    # _PRESSURE_RATIO via _effective_budget_bytes().
    return max(0, int(_min_total_mem_bytes_across_ranks()))


def _peak_mem_bytes(profiling_results) -> int:
    peak = 0
    for prof in profiling_results.values():
        if getattr(prof, "fwd_mem", None):
            peak = max(peak, max(int(m[3]) for m in prof.fwd_mem))
        if getattr(prof, "bwd_mem", None):
            peak = max(peak, max(int(m[3]) for m in prof.bwd_mem))
    return int(peak)


def _build_ds_id_to_size_bytes(graph_id: int, profiling_results, param_manager: Dict[int, DSGraphParamManager]) -> Dict[int, int]:
    ds_id_to_size: Dict[int, int] = {}

    for _, pm in param_manager.items():
        for param_name, ds_param in pm.params.items():
            ds_id = int(pm.ds_ids[param_name])
            ds_id_to_size[ds_id] = int(ds_param.param.numel() * ds_param.param.element_size())

    prof = profiling_results[graph_id]
    for g in (getattr(prof, "fwd_graph", None), getattr(prof, "bwd_graph", None)):
        if g is None:
            continue
        for n in g.nodes:
            if n.target != torch.ops.dc.allgather_param.default:
                continue
            ds_id = int(n.args[2])
            alloc_mem = int(n.meta.get("alloc_mem", 0))
            if alloc_mem > 0:
                ds_id_to_size[ds_id] = max(int(ds_id_to_size.get(ds_id, 0)), alloc_mem)
                continue
            if "tensor_size" in n.meta:
                ds_id_to_size[ds_id] = max(int(ds_id_to_size.get(ds_id, 0)), int(n.meta["tensor_size"]))

    return ds_id_to_size


def _build_mem_peak_by_node_name(mem_list, graph: Graph) -> Dict[str, int]:
    peak_by_name: Dict[str, int] = {name: int(peak) for name, _, _, peak in mem_list}
    prev_peak = 0
    for n in graph.nodes:
        if n.name in peak_by_name:
            prev_peak = peak_by_name[n.name]
        else:
            peak_by_name[n.name] = prev_peak
    return peak_by_name


def _build_mem_alloc_by_node_name(mem_list, graph: Graph) -> Dict[str, int]:
    alloc_by_name: Dict[str, int] = {name: int(current_alloc) for name, current_alloc, _, _ in mem_list}
    prev_alloc = 0
    for n in graph.nodes:
        if n.name in alloc_by_name:
            prev_alloc = alloc_by_name[n.name]
        else:
            alloc_by_name[n.name] = prev_alloc
    return alloc_by_name


def _live_param_bytes_by_node(graph: Graph, ds_id_to_size: Dict[int, int]) -> Dict[Node, int]:
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


def _find_layer_anchors(graph: Graph, ds_id_to_layer: Dict[int, int], layer_set: Set[int]) -> Dict[int, Node]:
    anchors: Dict[int, Node] = {}
    for n in graph.nodes:
        if n.target == torch.ops.dc.wait_allgather.default:
            ds_id = int(n.args[2])
            layer = ds_id_to_layer.get(ds_id)
            if layer not in layer_set:
                continue
            anchors.setdefault(int(layer), n)
    if len(anchors) == len(layer_set):
        return anchors

    for n in graph.nodes:
        if n.target == torch.ops.dc.allgather_param.default:
            ds_id = int(n.args[2])
            layer = ds_id_to_layer.get(ds_id)
            if layer not in layer_set:
                continue
            anchors.setdefault(int(layer), n)
    return anchors


def _is_param_comm_node(node: Node) -> bool:
    return node.target in {
        torch.ops.dc.allgather_param.default,
        torch.ops.dc.wait_allgather.default,
        torch.ops.dc.release_param.default,
        torch.ops.dc.prefetch_params_fused.default,
    }


def _is_comm_node(node: Node) -> bool:
    return bool(getattr(node, "meta", {}).get("comm", False))


def _is_fixed_comm_marker_node(node: Node) -> bool:
    return _is_comm_node(node) and not _is_param_comm_node(node)


def _extract_forward_candidates(graph: Graph, mem_list, ordered_layers: Sequence[int], ds_id_to_layer: Dict[int, int],
                                ds_id_to_size: Dict[int, int]) -> List[_LayerCandidate]:
    anchors = _find_layer_anchors(graph, ds_id_to_layer, set(int(x) for x in ordered_layers))
    if not anchors:
        return []

    nodes = list(graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    alloc_by_name = _build_mem_alloc_by_node_name(mem_list, graph)
    live_by_node = _live_param_bytes_by_node(graph, ds_id_to_size)

    out: List[_LayerCandidate] = []
    for idx, layer in enumerate(ordered_layers):
        anchor = anchors.get(int(layer))
        if anchor is None:
            continue
        start = pos[anchor]
        next_start = len(nodes)
        for next_layer in ordered_layers[idx + 1:]:
            next_anchor = anchors.get(int(next_layer))
            if next_anchor is not None:
                next_start = pos[next_anchor]
                break

        entry_node = nodes[start - 1] if start > 0 else None
        entry_non_param_alloc = 0
        if entry_node is not None:
            entry_non_param_alloc = max(
                0,
                int(alloc_by_name.get(entry_node.name, 0)) - int(live_by_node.get(entry_node, 0)),
            )

        max_non_param_alloc = entry_non_param_alloc
        compute_ms = 0.0
        for n in nodes[start:next_start]:
            if not _is_param_comm_node(n) and not _is_fixed_comm_marker_node(n) and n.target != torch.ops.dc.end_backward.default:
                compute_ms += float(n.meta.get("device_time", 0.0))
            current_non_param_alloc = max(
                0,
                int(alloc_by_name.get(n.name, 0)) - int(live_by_node.get(n, 0)),
            )
            max_non_param_alloc = max(max_non_param_alloc, current_non_param_alloc)

        saved_bytes = max(0, int(max_non_param_alloc) - int(entry_non_param_alloc))
        out.append(
            _LayerCandidate(
                layer_idx=int(layer),
                saved_bytes=int(saved_bytes),
                recompute_cost_ms=float(compute_ms),
            ))

    return out


def _extract_block_profiles(graph: Graph, mem_list, ordered_layers: Sequence[int], ds_id_to_layer: Dict[int, int], *,
                            bwd: bool, k_offset: int) -> List[_BlockProfile]:
    anchors = _find_layer_anchors(graph, ds_id_to_layer, set(int(x) for x in ordered_layers))
    if not anchors:
        return []

    nodes = list(graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    alloc_by_name = _build_mem_alloc_by_node_name(mem_list, graph)
    layer_order = list(reversed([int(x) for x in ordered_layers])) if bwd else list(int(x) for x in ordered_layers)

    out: List[_BlockProfile] = []
    for idx, layer in enumerate(layer_order):
        anchor = anchors.get(int(layer))
        if anchor is None:
            continue
        start = pos[anchor]
        next_start = len(nodes)
        for next_layer in layer_order[idx + 1:]:
            next_anchor = anchors.get(int(next_layer))
            if next_anchor is not None:
                next_start = pos[next_anchor]
                break

        block_peak = 0
        for n in nodes[start:next_start]:
            block_peak = max(block_peak, int(alloc_by_name.get(n.name, 0)))

        out.append(
            _BlockProfile(
                k=int(k_offset + idx),
                phase="BWD" if bwd else "FWD",
                layer_idx=int(layer),
                peak_bytes=int(block_peak),
            )
        )

    return out


def _build_solver_matrix(candidates: Sequence[_LayerCandidate], blocks: Sequence[_BlockProfile]):
    ordered_layers = [int(c.layer_idx) for c in candidates]
    layer_to_pos = {int(layer): i for i, layer in enumerate(ordered_layers)}
    n = len(candidates)
    K = len(blocks)
    saved = [int(c.saved_bytes) for c in candidates]

    A = [[0 for _ in range(n)] for _ in range(K)]
    for i, layer in enumerate(ordered_layers):
        p = layer_to_pos[int(layer)]
        q = 2 * n - 1 - p
        for k in range(p + 1, q + 1):
            if 0 <= k < K:
                A[k][i] = int(saved[i])
    return A


def _greedy_disable_recompute(candidates: Sequence[_LayerCandidate], blocks: Sequence[_BlockProfile],
                              budget_bytes: int) -> Tuple[List[int], bool]:
    A = _build_solver_matrix(candidates, blocks)
    slack = [max(0, int(budget_bytes) - int(block.peak_bytes)) for block in blocks]
    selected: Set[int] = set()
    remaining = set(range(len(candidates)))

    while remaining:
        best_idx = None
        best_score = None
        best_value = 0.0
        best_added = 0
        for idx in remaining:
            added = [int(A[k][idx]) for k in range(len(blocks))]
            if any(added[k] > slack[k] for k in range(len(blocks))):
                continue
            value = max(float(candidates[idx].recompute_cost_ms), 1e-6)
            total_added = sum(added)
            score = value / float(max(total_added, 1))
            if (best_score is None or score > best_score or
                    (score == best_score and (value > best_value or
                                              (value == best_value and total_added < best_added)))):
                best_idx = idx
                best_score = score
                best_value = value
                best_added = total_added

        if best_idx is None:
            break

        selected.add(int(best_idx))
        remaining.remove(int(best_idx))
        for k in range(len(blocks)):
            slack[k] = max(0, int(slack[k]) - int(A[k][best_idx]))

    layers = [int(candidates[i].layer_idx) for i in sorted(selected)]
    return layers, True


def _solve_selection(candidates: Sequence[_LayerCandidate], blocks: Sequence[_BlockProfile],
                     budget_bytes: int) -> Tuple[List[int], bool, str]:
    if not candidates:
        return [], True, "empty"

    over_budget = [int(block.peak_bytes) > int(budget_bytes) for block in blocks]
    if any(over_budget):
        return [], False, "bootstrap_over_budget"

    slack = [max(0, int(budget_bytes) - int(block.peak_bytes)) for block in blocks]
    if max(slack, default=0) <= 0:
        return [], True, "full_checkpoint"

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        A = np.asarray(_build_solver_matrix(candidates, blocks), dtype=float)
        c = -np.asarray([max(float(cand.recompute_cost_ms), 1e-6) for cand in candidates], dtype=float)
        integrality = np.ones(len(candidates), dtype=int)
        bounds = Bounds(lb=np.zeros(len(candidates), dtype=float), ub=np.ones(len(candidates), dtype=float))
        constraints = [LinearConstraint(A, lb=np.full(len(blocks), -np.inf), ub=np.asarray(slack, dtype=float))]
        result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if result.success and result.x is not None:
            layers = [
                int(candidates[i].layer_idx) for i, x in enumerate(result.x.tolist()) if float(x) >= 0.5
            ]
            return sorted(layers), True, "milp"
    except Exception:
        pass

    layers, feasible = _greedy_disable_recompute(candidates, blocks, budget_bytes)
    return sorted(layers), bool(feasible), "greedy"


def _broadcast_and_store_plan(plan: dict) -> dict:
    global _LATEST_PLAN

    if dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [plan] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        plan = obj_list[0]

    expected = plan.get("plan_hash")
    if expected is None:
        raise RuntimeError(f"[{NAME}] plan missing plan_hash")
    actual = _plan_hash(plan)
    if actual != expected:
        raise RuntimeError(f"[{NAME}] plan hash mismatch: expected={expected} actual={actual}")

    _LATEST_PLAN = plan
    return plan


def _dump_plan_if_enabled(plan: dict) -> None:
    if _CFG is None or dist.get_rank() != 0:
        return
    if not _CFG.dump_plan and os.environ.get("DS_SELECTIVE_ACTIVATION_RECOMPUTE_DUMP", "") == "":
        return

    dump_dir = _CFG.dump_dir or os.environ.get("DS_SELECTIVE_ACTIVATION_RECOMPUTE_DUMP", "").strip() or "."
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"selective_activation_recompute_{plan.get('plan_hash', 'unknown')}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_canonical_json(plan))
    log_rank0(f"[{NAME}] Wrote plan JSON: {path}", enable=True)


def maybe_init_layer_mapping(model: torch.nn.Module, compile_config, schedule) -> None:
    global _CFG, _LAYER_MAPPING, _LAYER_MODULES, _NON_CANDIDATE_GC_MODULES, _LATEST_PLAN, _APPLY_LOGGED, _PRESSURE_RATIO

    should_enable = bool(getattr(compile_config, "selective_activation_recompute", False))
    if not should_enable:
        should_enable = _contains_pass(schedule, ".selective_activation_recompute")
    if not should_enable:
        return

    if bool(getattr(compile_config, "global_layer_scheduler", False)) or _contains_pass(schedule, ".global_layer_scheduler"):
        raise RuntimeError(f"[{NAME}] combining with global_layer_scheduler is not implemented yet")

    ratio_cfg = getattr(compile_config, "selective_activation_recompute_pressure_ratio", _DEFAULT_PRESSURE_RATIO)
    try:
        ratio = float(ratio_cfg)
    except Exception:
        ratio = _DEFAULT_PRESSURE_RATIO
    if not (0.0 < ratio <= 1.0):
        ratio = _DEFAULT_PRESSURE_RATIO
    _PRESSURE_RATIO = float(ratio)

    layer_regexes = tuple(
        getattr(
            compile_config,
            "selective_activation_recompute_layer_regexes",
            (
                r"(?:^|\.)layers\.(\d+)(?:\.|$)",
                r"(?:^|\.)h\.(\d+)(?:\.|$)",
            ),
        )
    )
    module_regexes = tuple(
        getattr(
            compile_config,
            "selective_activation_recompute_module_regexes",
            (
                r"(?:^|\.)layers\.(\d+)$",
                r"(?:^|\.)h\.(\d+)$",
            ),
        )
    )

    _CFG = _RecomputeConfig(
        layer_regexes=layer_regexes,
        module_regexes=module_regexes,
        dump_plan=bool(getattr(compile_config, "selective_activation_recompute_dump_plan", False)),
        dump_dir=(str(getattr(compile_config, "selective_activation_recompute_dump_dir", "")).strip() or None),
    )

    layer_to_module: Dict[int, torch.nn.Module] = {}
    layer_to_name: Dict[int, str] = {}
    gc_attr_modules: List[torch.nn.Module] = []
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            gc_attr_modules.append(module)
        layer_idx = _infer_layer_idx(name, module_regexes)
        if layer_idx is None:
            continue
        if not hasattr(module, "gradient_checkpointing"):
            continue
        prev_name = layer_to_name.get(int(layer_idx))
        if prev_name is not None and len(prev_name) <= len(name):
            continue
        layer_to_module[int(layer_idx)] = module
        layer_to_name[int(layer_idx)] = name

    if not layer_to_module:
        log_rank0(f"[{NAME}] No checkpoint-capable transformer layers detected; pass disabled.", enable=True)
        _LAYER_MAPPING = None
        _LAYER_MODULES = {}
        _NON_CANDIDATE_GC_MODULES = []
        return

    ds_id_to_layer: Dict[int, int] = {}
    name_and_ds = []
    for name, p in model.named_parameters():
        if not hasattr(p, "ds_id"):
            continue
        layer_idx = _infer_layer_idx(name, layer_regexes)
        if layer_idx is None or int(layer_idx) not in layer_to_module:
            continue
        ds_id = int(p.ds_id)
        ds_id_to_layer[ds_id] = int(layer_idx)
        name_and_ds.append((name, ds_id, int(layer_idx)))

    mapping_hash = _sha256_hex(
        _canonical_json({
            "layers": sorted((int(k), v) for k, v in layer_to_name.items()),
            "params": sorted(name_and_ds),
        }).encode("utf-8"))

    if dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [mapping_hash] if dist.get_rank() == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        if obj_list[0] != mapping_hash:
            raise RuntimeError(
                f"[{NAME}] layer mapping hash mismatch across ranks: local={mapping_hash} rank0={obj_list[0]}")

    _LAYER_MAPPING = _LayerMapping(
        mapping_hash=mapping_hash,
        layer_indices=tuple(sorted(layer_to_module.keys())),
        layer_to_name=dict(sorted(layer_to_name.items())),
        ds_id_to_layer=ds_id_to_layer,
    )
    _LAYER_MODULES = dict(sorted(layer_to_module.items()))
    candidate_modules = {id(m) for m in _LAYER_MODULES.values()}
    _NON_CANDIDATE_GC_MODULES = [m for m in gc_attr_modules if id(m) not in candidate_modules]
    _LATEST_PLAN = None
    _APPLY_LOGGED = False
    _set_layer_checkpoint_selection(_LAYER_MODULES.keys())

    log_rank0(
        f"[{NAME}] Initialized layer mapping: layers={len(_LAYER_MODULES)} mapped_ds_ids={len(ds_id_to_layer)} budget_bytes={_budget_bytes()} bootstrap=full_checkpoint",
        enable=True,
    )


def _build_plan(graph_id: int, profiling_results, param_manager: Dict[int, DSGraphParamManager]) -> Optional[dict]:
    if _CFG is None or _LAYER_MAPPING is None:
        return None

    prof = profiling_results.get(graph_id)
    if prof is None or prof.fwd_graph is None or prof.bwd_graph is None:
        return None

    ds_id_to_size = _build_ds_id_to_size_bytes(graph_id, profiling_results, param_manager)
    layer_set = set(int(x) for x in _LAYER_MAPPING.layer_indices)
    anchor_fwd = _find_layer_anchors(prof.fwd_graph, _LAYER_MAPPING.ds_id_to_layer, layer_set)
    anchor_bwd = _find_layer_anchors(prof.bwd_graph, _LAYER_MAPPING.ds_id_to_layer, layer_set)

    ordered_layers = [
        int(layer) for layer in _LAYER_MAPPING.layer_indices if int(layer) in anchor_fwd and int(layer) in anchor_bwd
    ]
    if not ordered_layers:
        log_rank0(f"[{NAME}] No layers with both forward/backward anchors; planning skipped.", enable=True)
        return None

    candidates = _extract_forward_candidates(
        prof.fwd_graph,
        prof.fwd_mem,
        ordered_layers,
        _LAYER_MAPPING.ds_id_to_layer,
        ds_id_to_size,
    )
    if not candidates:
        return None

    active_layers = [int(c.layer_idx) for c in candidates]
    blocks_fwd = _extract_block_profiles(
        prof.fwd_graph,
        prof.fwd_mem,
        active_layers,
        _LAYER_MAPPING.ds_id_to_layer,
        bwd=False,
        k_offset=0,
    )
    blocks_bwd = _extract_block_profiles(
        prof.bwd_graph,
        prof.bwd_mem,
        active_layers,
        _LAYER_MAPPING.ds_id_to_layer,
        bwd=True,
        k_offset=len(active_layers),
    )
    blocks = blocks_fwd + blocks_bwd
    if len(blocks) != 2 * len(active_layers):
        log_rank0(
            f"[{NAME}] Incomplete block profiles: expected={2 * len(active_layers)} actual={len(blocks)}; planning skipped.",
            enable=True,
        )
        return None

    budget_bytes = _budget_bytes()
    disabled_layers, feasible, solver = _solve_selection(candidates, blocks, budget_bytes)
    disabled_set = set(int(x) for x in disabled_layers)
    checkpointed_layers = [int(layer) for layer in active_layers if int(layer) not in disabled_set]
    if not feasible:
        disabled_layers = []
        checkpointed_layers = sorted(active_layers)

    bootstrap_peak = _peak_mem_bytes(profiling_results)
    candidates_meta = [
        {
            "layer_idx": int(c.layer_idx),
            "saved_bytes": int(c.saved_bytes),
            "recompute_cost_ms": float(c.recompute_cost_ms),
        } for c in candidates
    ]
    plan = {
        "name": NAME,
        "mapping_hash": _LAYER_MAPPING.mapping_hash,
        "budget_bytes": int(budget_bytes),
        "bootstrap_mode": "full_checkpoint",
        "bootstrap_peak_bytes": int(bootstrap_peak),
        "ordered_layers": [int(x) for x in active_layers],
        "selected_layers": [int(x) for x in checkpointed_layers],
        "checkpointed_layers": [int(x) for x in checkpointed_layers],
        "disabled_layers": [int(x) for x in disabled_layers],
        "solver": str(solver),
        "feasible": bool(feasible),
        "candidates": candidates_meta,
        "block_profiles": [
            {
                "k": int(block.k),
                "phase": str(block.phase),
                "layer_idx": int(block.layer_idx),
                "peak_bytes": int(block.peak_bytes),
            } for block in blocks
        ],
    }
    plan["plan_hash"] = _plan_hash(plan)
    return plan


def plan(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
         mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool):
    del create_inputs_fn, mem_budget, gm

    if not bwd or _LAYER_MAPPING is None:
        return None

    last_bwd = _last_backward_graph_id(graph_order)
    if last_bwd is None or graph_id != last_bwd:
        return None

    plan_dict = _build_plan(graph_id, profiling_results, param_manager)
    if plan_dict is None:
        return None

    plan_dict = _broadcast_and_store_plan(plan_dict)
    _dump_plan_if_enabled(plan_dict)

    log_rank0(
        f"[{NAME}] Planned selective recompute: checkpointed_layers={len(plan_dict['checkpointed_layers'])}/{len(plan_dict['ordered_layers'])} disabled_layers={len(plan_dict['disabled_layers'])} solver={plan_dict['solver']} feasible={plan_dict['feasible']} budget_bytes={plan_dict['budget_bytes']} bootstrap_peak_bytes={plan_dict['bootstrap_peak_bytes']}",
        enable=True,
    )
    return None


def apply(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
          mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool):
    del gm, graph_id, graph_order, profiling_results, create_inputs_fn, mem_budget, param_manager, bwd

    global _APPLY_LOGGED
    if _LATEST_PLAN is None or _APPLY_LOGGED:
        return None

    # If we didn't get a chance to pre-apply at step boundaries, refine and apply here.
    _refine_plan_with_runtime_peak(_LATEST_PLAN)
    _APPLY_LOGGED = True
    target_layers = {int(x) for x in _LATEST_PLAN.get("checkpointed_layers", _LATEST_PLAN.get("selected_layers", []))}
    _set_layer_checkpoint_selection(target_layers)
    log_rank0(
        f"[{NAME}] Applied selective recompute plan: checkpointed_layers={_LATEST_PLAN.get('checkpointed_layers', _LATEST_PLAN.get('selected_layers', []))} disabled_layers={_LATEST_PLAN.get('disabled_layers', [])} runtime_peak_alloc_bytes={_LATEST_PLAN.get('runtime_peak_alloc_bytes', 0)} effective_budget_bytes={_LATEST_PLAN.get('effective_budget_bytes', _LATEST_PLAN.get('budget_bytes', 0))} reserve_bytes={_LATEST_PLAN.get('pressure_reserve_bytes', 0)}",
        enable=True,
    )
    return None
