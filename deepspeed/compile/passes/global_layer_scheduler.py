# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.fx import Graph, GraphModule, Node

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

from ..graph_param import DSGraphParamManager
from ..util import get_deepcompile_handle, log_rank0

NAME = "global_layer_scheduler"


@dataclass(frozen=True)
class _SchedulerConfig:
    layer_regexes: Tuple[str, ...]
    lookahead_blocks: int
    fuse_max_bytes: int
    fuse_deadline_window_blocks: int
    fuse_factor: float
    inner_refine_max_iters: int
    inner_refine_min_gain_ms: float
    mem_margin: float
    safety_margin_bytes: int
    outer_keep_max_iters: int
    outer_keep_min_gain_ms: float
    outer_keep_top_n: int
    max_tasks_per_block: int
    rewrite_comm_ops: bool
    include_unmapped_params: bool
    set_persistent: bool
    dump_schedule: bool
    dump_dir: Optional[str]


@dataclass(frozen=True)
class _LayerMapping:
    mapping_hash: str
    L: int
    layer_to_ds_ids: Tuple[Tuple[int, ...], ...]
    ds_id_to_layer: Dict[int, int]


_CFG: Optional[_SchedulerConfig] = None
_LAYER_MAPPING: Optional[_LayerMapping] = None
_LATEST_SCHEDULE: Optional[dict] = None


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
        lookahead_blocks=int(getattr(compile_config, "global_layer_scheduler_lookahead_blocks", 4)),
        fuse_max_bytes=int(getattr(compile_config, "global_layer_scheduler_fuse_max_bytes", 256 * 1024 * 1024)),
        fuse_deadline_window_blocks=int(
            getattr(compile_config, "global_layer_scheduler_fuse_deadline_window_blocks", 1)),
        fuse_factor=float(getattr(compile_config, "global_layer_scheduler_fuse_factor", 0.8)),
        inner_refine_max_iters=int(getattr(compile_config, "global_layer_scheduler_inner_refine_max_iters", 128)),
        inner_refine_min_gain_ms=float(getattr(compile_config, "global_layer_scheduler_inner_refine_min_gain_ms", 0.05)),
        mem_margin=float(getattr(compile_config, "global_layer_scheduler_mem_margin", 0.1)),
        safety_margin_bytes=int(getattr(compile_config, "global_layer_scheduler_safety_margin_bytes", 512 * 1024 * 1024)),
        outer_keep_max_iters=int(getattr(compile_config, "global_layer_scheduler_outer_keep_max_iters", 32)),
        outer_keep_min_gain_ms=float(getattr(compile_config, "global_layer_scheduler_outer_keep_min_gain_ms", 0.5)),
        outer_keep_top_n=int(getattr(compile_config, "global_layer_scheduler_outer_keep_top_n", 3)),
        max_tasks_per_block=int(getattr(compile_config, "global_layer_scheduler_max_tasks_per_block", 16)),
        rewrite_comm_ops=bool(getattr(compile_config, "global_layer_scheduler_rewrite_comm_ops", True)),
        include_unmapped_params=bool(getattr(compile_config, "global_layer_scheduler_include_unmapped_params", True)),
        set_persistent=bool(getattr(compile_config, "global_layer_scheduler_set_persistent", False)),
        dump_schedule=bool(getattr(compile_config, "global_layer_scheduler_dump_schedule", False)),
        dump_dir=(str(getattr(compile_config, "global_layer_scheduler_dump_dir", "")).strip() or None),
    )
    _CFG = cfg

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
        vals = torch.tensor([total_mem], device=torch.device(get_accelerator().current_device()))
        dist.all_reduce(vals, dist.ReduceOp.MIN)
        total_mem = int(vals[0].item())
    return total_mem


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


def _build_ds_id_to_bwd_wait_ms(graph_id: int, profiling_results) -> Dict[int, float]:
    prof = profiling_results[graph_id]
    g = getattr(prof, "bwd_graph", None)
    if g is None:
        return {}
    out: Dict[int, float] = {}
    for n in g.nodes:
        # Use allgather_param device_time as a proxy for backward comm cost (more stable than wait timing).
        if n.target == torch.ops.dc.allgather_param.default and "device_time" in n.meta:
            ds_id = int(n.args[2])
            out[ds_id] = out.get(ds_id, 0.0) + float(n.meta["device_time"])
    return out


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


def _choose_keep_layers(L: int, layer_to_ds_ids: Sequence[Sequence[int]], ds_id_to_size: Dict[int, int],
                        ds_id_to_bwd_wait_ms: Dict[int, float], mem_budget_bytes: int) -> List[int]:
    if L <= 0 or mem_budget_bytes <= 0:
        return []

    K = 2 * L
    reserved = [0] * K

    candidates = []
    for i in range(L):
        ds_ids = [ds_id for ds_id in layer_to_ds_ids[i] if ds_id in ds_id_to_size]
        if not ds_ids:
            continue
        size_bytes = sum(ds_id_to_size[ds_id] for ds_id in ds_ids)
        if size_bytes <= 0:
            continue
        benefit_ms = sum(ds_id_to_bwd_wait_ms.get(ds_id, 0.0) for ds_id in ds_ids)
        span = (2 * L - 1) - 2 * i
        score = benefit_ms / max(size_bytes * span, 1)
        candidates.append((score, benefit_ms, size_bytes, i))

    candidates.sort(reverse=True)

    keep_layers: List[int] = []
    for _, _, size_bytes, i in candidates:
        f_k = i
        b_k = 2 * L - 1 - i
        feasible = True
        for k in range(f_k, b_k + 1):
            if reserved[k] + size_bytes > mem_budget_bytes:
                feasible = False
                break
        if not feasible:
            continue
        for k in range(f_k, b_k + 1):
            reserved[k] += size_bytes
        keep_layers.append(i)

    keep_layers.sort()
    return keep_layers


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


def _is_param_comm_node(n: Node) -> bool:
    return n.target in {
        torch.ops.dc.allgather_param.default,
        torch.ops.dc.wait_allgather.default,
        torch.ops.dc.release_param.default,
        torch.ops.dc.prefetch_params_fused.default,
    }


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
                              ds_id_to_size: Dict[int, int], total_mem_bytes: int, mem_margin: float,
                              safety_margin_bytes: int, bwd: bool) -> List[_BlockProfile]:
    anchors = _find_layer_anchors(graph, ds_id_to_layer, L)
    if len(anchors) == 0:
        return []

    nodes = list(graph.nodes)
    pos = {n: i for i, n in enumerate(nodes)}
    peak_by_name = _build_mem_peak_by_node_name(mem_list, graph)
    live_by_node = _live_param_bytes_by_node(graph, ds_id_to_size)

    safe_total = int(total_mem_bytes * (1.0 - float(mem_margin)))
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
        base_mem_excl_params = 0
        for n in nodes[start:end]:
            if not _is_param_comm_node(n):
                compute_ms += float(n.meta.get("device_time", 0.0))
            peak = int(peak_by_name.get(n.name, 0))
            live = int(live_by_node.get(n, 0))
            base_mem_excl_params = max(base_mem_excl_params, max(0, peak - live))

        cap_bytes = max(0, safe_total - base_mem_excl_params - int(safety_margin_bytes))
        unified_k = layer if not bwd else L + idx_in_phase
        blocks.append(
            _BlockProfile(
                k=int(unified_k),
                phase="BWD" if bwd else "FWD",
                layer_idx=int(layer),
                compute_ms=float(compute_ms),
                cap_bytes=int(cap_bytes),
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


def _compute_prefix_ms(blocks: Sequence[_BlockProfile]) -> List[float]:
    prefix = [0.0]
    for b in blocks:
        prefix.append(prefix[-1] + float(b.compute_ms))
    return prefix


def _sum_compute_ms(prefix: Sequence[float], start_k: int, end_k_exclusive: int) -> float:
    start_k = max(0, int(start_k))
    end_k_exclusive = max(start_k, int(end_k_exclusive))
    if start_k >= len(prefix) - 1:
        return 0.0
    end_k_exclusive = min(end_k_exclusive, len(prefix) - 1)
    return float(prefix[end_k_exclusive] - prefix[start_k])


def _build_tasks(L: int, layer_to_ds_ids: Sequence[Sequence[int]], ds_id_to_size: Dict[int, int],
                 keep_layers: Set[int]) -> List[_Task]:
    tasks: List[_Task] = []
    for layer_idx in range(L):
        ds_ids = tuple(int(ds_id) for ds_id in layer_to_ds_ids[layer_idx] if int(ds_id) in ds_id_to_size)
        if not ds_ids:
            continue
        size_bytes = int(sum(ds_id_to_size[ds_id] for ds_id in ds_ids))
        if size_bytes <= 0:
            continue

        f_deadline = layer_idx
        b_deadline = 2 * L - 1 - layer_idx
        keep = layer_idx in keep_layers

        tasks.append(
            _Task(
                kind="FWD",
                layer_idx=layer_idx,
                ds_ids=ds_ids,
                size_bytes=size_bytes,
                min_start_k=0,
                deadline_k=f_deadline,
                release_end_k=(b_deadline if keep else f_deadline),
            ))
        if not keep:
            tasks.append(
                _Task(
                    kind="BWD",
                    layer_idx=layer_idx,
                    ds_ids=ds_ids,
                    size_bytes=size_bytes,
                    min_start_k=f_deadline + 1,
                    deadline_k=b_deadline,
                    release_end_k=b_deadline,
                ))
    return tasks


def _reserve_range_feasible(reserved: List[int], blocks: Sequence[_BlockProfile], start_k: int, end_k: int,
                            delta_bytes: int) -> bool:
    start_k = int(start_k)
    end_k = int(end_k)
    if delta_bytes <= 0:
        return True
    for k in range(start_k, end_k + 1):
        if k < 0 or k >= len(blocks):
            return False
        if reserved[k] + delta_bytes > int(blocks[k].cap_bytes):
            return False
    return True


def _reserve_range_apply(reserved: List[int], start_k: int, end_k: int, delta_bytes: int) -> None:
    start_k = int(start_k)
    end_k = int(end_k)
    if delta_bytes <= 0:
        return
    for k in range(start_k, end_k + 1):
        reserved[k] += int(delta_bytes)


def _assign_task_starts(tasks: List[_Task], blocks: Sequence[_BlockProfile], comm_model: _CommModel,
                        cfg: _SchedulerConfig) -> Tuple[List[int], Dict[Tuple[int, str], int]]:
    # Greedy EDF assignment with memory constraints. Every task is started at some block <= deadline.
    tasks.sort(key=lambda t: (t.deadline_k, 0 if t.kind == "FWD" else 1, -t.size_bytes, t.layer_idx))

    reserved = [0 for _ in blocks]
    compute_prefix_ms = _compute_prefix_ms(blocks)
    starts_count: Dict[int, int] = {}
    per_task_start: Dict[Tuple[int, str], int] = {}

    for t in tasks:
        n_ops = len(t.ds_ids)
        # prefetch_params_fused issues *per-ds_id* allgathers; model per-op overhead via alpha*n_ops.
        # Apply fusion discount to the fixed launch overhead only (alpha), not the bandwidth term (beta*bytes).
        fuse_alpha_factor = 1.0
        if n_ops > 1:
            fuse_alpha_factor = max(0.0, min(1.0, float(cfg.fuse_factor)))
        comm_ms = (float(comm_model.alpha_ms) * float(fuse_alpha_factor) * float(n_ops) +
                   float(comm_model.beta_ms_per_byte) * float(t.size_bytes))

        hard_earliest = int(t.min_start_k)
        soft_earliest = max(int(t.min_start_k), int(t.deadline_k) - int(cfg.lookahead_blocks))

        desired = hard_earliest
        found = False
        for s in range(int(t.deadline_k), int(soft_earliest) - 1, -1):
            if _sum_compute_ms(compute_prefix_ms, s, int(t.deadline_k)) >= comm_ms:
                desired = int(s)
                found = True
                break
        if not found and int(soft_earliest) > int(hard_earliest):
            for s in range(int(soft_earliest) - 1, int(hard_earliest) - 1, -1):
                if _sum_compute_ms(compute_prefix_ms, s, int(t.deadline_k)) >= comm_ms:
                    desired = int(s)
                    break

        chosen: Optional[int] = None
        for s in range(desired, t.deadline_k + 1):
            if starts_count.get(s, 0) >= cfg.max_tasks_per_block:
                continue
            if _reserve_range_feasible(reserved, blocks, s, t.release_end_k, t.size_bytes):
                chosen = s
                break

        if chosen is None:
            # Fallback: delay to the latest feasible start (often deadline) to satisfy memory.
            for s in range(t.deadline_k, hard_earliest - 1, -1):
                if starts_count.get(s, 0) >= cfg.max_tasks_per_block:
                    continue
                if _reserve_range_feasible(reserved, blocks, s, t.release_end_k, t.size_bytes):
                    chosen = s
                    break

        if chosen is None:
            # As a last resort, force at deadline without reserving extra (should be rare if baseline fits).
            chosen = int(t.deadline_k)
            if not _reserve_range_feasible(reserved, blocks, chosen, t.release_end_k, t.size_bytes):
                log_rank0(
                    f"[{NAME}] WARNING: cannot reserve memory for task layer={t.layer_idx} kind={t.kind} size={t.size_bytes}B start={chosen}..{t.release_end_k}",
                    enable=True,
                )
                # Still record start; leave reserved unchanged.
                t.start_k = chosen
                per_task_start[(t.layer_idx, t.kind)] = chosen
                starts_count[chosen] = starts_count.get(chosen, 0) + 1
                continue

        _reserve_range_apply(reserved, chosen, t.release_end_k, t.size_bytes)
        t.start_k = int(chosen)
        per_task_start[(t.layer_idx, t.kind)] = int(chosen)
        starts_count[int(chosen)] = starts_count.get(int(chosen), 0) + 1

    return reserved, per_task_start


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

    fuse_window = int(cfg.fuse_deadline_window_blocks if fuse_deadline_window_blocks is None else fuse_deadline_window_blocks)

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
) -> Tuple[float, List[float], Dict[int, float], Dict[Tuple[int, str], float]]:
    K = len(blocks)
    ready_time: Dict[Tuple[int, str], float] = {}
    stall_ms_per_block = [0.0 for _ in range(K)]
    stall_bwd_per_layer: Dict[int, float] = {}

    # Model the *actual* execution order: comm groups are launched in (start_k, list-order).
    now = 0.0  # compute timeline
    comm_free = 0.0  # comm stream availability time

    for k in range(K):
        # Launch groups scheduled at this block boundary.
        for g in groups_by_start.get(k, []):
            start_t = max(float(comm_free), float(now))
            comm_end = start_t + float(g.comm_ms)
            comm_free = comm_end
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

        now += float(blocks[k].compute_ms)

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


def _starts_count_by_block(tasks: Sequence[_Task]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for t in tasks:
        if t.start_k is None:
            continue
        k = int(t.start_k)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _try_refine_task_starts(
    tasks: List[_Task],
    blocks: Sequence[_BlockProfile],
    comm_model: _CommModel,
    ds_id_to_size: Dict[int, int],
    cfg: _SchedulerConfig,
    keep_layers: Set[int],
) -> Tuple[List[_Task], List[int], Dict[Tuple[int, str], int], Dict[int, List[List[int]]], float, List[float], Dict[int, float], int]:
    # Local search: iteratively move the most-stalling task earlier (within lookahead + memory cap).
    tasks = list(tasks)

    # Index for quick lookup.
    key_to_task: Dict[Tuple[int, str], _Task] = {(t.layer_idx, t.kind): t for t in tasks if t.start_k is not None}

    best_reserved = _reserved_mem_by_block(tasks, len(blocks))
    best_per_task_start = {(t.layer_idx, t.kind): int(t.start_k) for t in tasks if t.start_k is not None}
    best_launches, best_groups_by_start = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg)
    best_step, best_stall, best_stall_bwd, _ = _simulate_schedule_in_graph_order(blocks, best_groups_by_start, keep_layers)

    # Try both: configured fusion window and strict window=0 (often reduces readiness coupling).
    def eval_with_fuse_window(window: int) -> Tuple[float, List[float], Dict[int, float], Dict[int, List[_CommGroup]], Dict[int, List[List[int]]]]:
        launches, groups = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg, fuse_deadline_window_blocks=window)
        step, stalls, stall_bwd, _ = _simulate_schedule_in_graph_order(blocks, groups, keep_layers)
        return step, stalls, stall_bwd, groups, launches

    fuse_candidates = sorted(set([0, int(cfg.fuse_deadline_window_blocks)]))
    chosen_fuse = int(cfg.fuse_deadline_window_blocks)
    for w in fuse_candidates:
        step, stalls, stall_bwd, groups, launches = eval_with_fuse_window(int(w))
        if step + 1e-9 < best_step:
            best_step, best_stall, best_stall_bwd = step, stalls, stall_bwd
            best_groups_by_start, best_launches = groups, launches
            chosen_fuse = int(w)

    # Main refinement loop.
    for _ in range(int(cfg.inner_refine_max_iters)):
        # Re-evaluate current state with the chosen fusion rule.
        cur_step, cur_stall, _, _, _ = eval_with_fuse_window(chosen_fuse)

        max_stall = max(cur_stall) if cur_stall else 0.0
        if max_stall <= 1e-9:
            # No predicted stalls left.
            if cur_step + 1e-9 < best_step:
                best_step, best_stall = cur_step, cur_stall
                best_reserved = _reserved_mem_by_block(tasks, len(blocks))
                best_per_task_start = {(t.layer_idx, t.kind): int(t.start_k) for t in tasks if t.start_k is not None}
                best_launches, best_groups_by_start = _build_comm_groups(tasks,
                                                                        comm_model,
                                                                        ds_id_to_size,
                                                                        cfg,
                                                                        fuse_deadline_window_blocks=chosen_fuse)
            break

        # Try stall blocks in descending stall order; tie-break by earliest k.
        stall_blocks = sorted([(float(s), int(k)) for k, s in enumerate(cur_stall) if s > 1e-9],
                              key=lambda x: (-x[0], x[1]))
        moved = False

        for _, k_pick in stall_blocks:
            required = _required_task_key_for_block(blocks[k_pick], keep_layers)
            t = key_to_task.get(required)
            if t is None or t.start_k is None:
                continue

            old_start = int(t.start_k)
            # Refinement is allowed to move earlier than the nominal lookahead window as long as memory caps hold.
            earliest_allowed = int(t.min_start_k)
            if earliest_allowed >= old_start:
                continue

            # Precompute constraints once.
            counts = _starts_count_by_block(tasks)
            reserved = _reserved_mem_by_block(tasks, len(blocks))
            size = int(t.size_bytes)

            best_candidate = None  # (new_step, new_start, new_stalls, new_stall_bwd, new_groups, new_launches)

            for candidate_start in range(old_start - 1, earliest_allowed - 1, -1):
                if counts.get(candidate_start, 0) + 1 > int(cfg.max_tasks_per_block):
                    continue

                feasible = True
                for kk in range(candidate_start, old_start):
                    if kk < 0 or kk >= len(blocks):
                        feasible = False
                        break
                    if reserved[kk] + size > int(blocks[kk].cap_bytes):
                        feasible = False
                        break
                if not feasible:
                    continue

                t.start_k = int(candidate_start)
                new_step, new_stalls, new_stall_bwd, new_groups_by_start, new_launches = eval_with_fuse_window(
                    chosen_fuse)
                t.start_k = int(old_start)

                if best_candidate is None or new_step + 1e-9 < best_candidate[0]:
                    best_candidate = (float(new_step), int(candidate_start), list(new_stalls), dict(new_stall_bwd),
                                      new_groups_by_start, new_launches)

            if best_candidate is None:
                continue

            cand_step, cand_start, cand_stalls, cand_stall_bwd, cand_groups_by_start, cand_launches = best_candidate
            if cand_step + 1e-6 < best_step - float(cfg.inner_refine_min_gain_ms):
                # Apply winning move.
                t.start_k = int(cand_start)
                best_step = float(cand_step)
                best_stall = list(cand_stalls)
                best_stall_bwd = dict(cand_stall_bwd)
                best_groups_by_start = cand_groups_by_start
                best_launches = cand_launches
                best_reserved = _reserved_mem_by_block(tasks, len(blocks))
                best_per_task_start = {(tt.layer_idx, tt.kind): int(tt.start_k) for tt in tasks if tt.start_k is not None}
                moved = True
                break

        if not moved:
            break

    # Rebuild tasks list with best starts.
    for t in tasks:
        if (t.layer_idx, t.kind) in best_per_task_start:
            t.start_k = int(best_per_task_start[(t.layer_idx, t.kind)])

    return (
        tasks,
        best_reserved,
        best_per_task_start,
        {int(k): v for k, v in sorted(best_launches.items())},
        float(best_step),
        list(best_stall),
        dict(best_stall_bwd),
        int(chosen_fuse),
    )


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
    safe_total = int(total_mem * (1.0 - float(cfg.mem_margin)))
    mem_budget_bytes = max(0, int(safe_total) - int(peak_mem) - int(cfg.safety_margin_bytes))

    prof = profiling_results.get(graph_id)
    if prof is None or getattr(prof, "fwd_graph", None) is None or getattr(prof, "bwd_graph", None) is None:
        return None

    blocks_fwd = _extract_blocks_for_graph(prof.fwd_graph,
                                          getattr(prof, "fwd_mem", []),
                                          L,
                                          mapping.ds_id_to_layer,
                                          ds_id_to_size,
                                          total_mem_bytes=total_mem,
                                          mem_margin=cfg.mem_margin,
                                          safety_margin_bytes=cfg.safety_margin_bytes,
                                          bwd=False)
    blocks_bwd = _extract_blocks_for_graph(prof.bwd_graph,
                                          getattr(prof, "bwd_mem", []),
                                          L,
                                          mapping.ds_id_to_layer,
                                          ds_id_to_size,
                                          total_mem_bytes=total_mem,
                                          mem_margin=cfg.mem_margin,
                                          safety_margin_bytes=cfg.safety_margin_bytes,
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
    if cfg.include_unmapped_params:
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
    ds_id_to_bwd_wait_ms = _build_ds_id_to_bwd_wait_ms(graph_id, profiling_results)
    ds_id_to_allgather_ms = _build_ds_id_to_allgather_ms(profiling_results)
    keep_layers_hint = set(
        _choose_keep_layers(
            L=L,
            layer_to_ds_ids=layer_to_ds_ids,
            ds_id_to_size=ds_id_to_size,
            ds_id_to_bwd_wait_ms=ds_id_to_bwd_wait_ms,
            mem_budget_bytes=mem_budget_bytes,
        ))

    def run_inner(keep_layers: Set[int]) -> dict:
        tasks = _build_tasks(L, layer_to_ds_ids, ds_id_to_size, keep_layers)
        reserved_mem, per_task_start = _assign_task_starts(tasks, blocks, comm_model, cfg)

        used_fuse_window = int(cfg.fuse_deadline_window_blocks)
        if int(cfg.inner_refine_max_iters) > 0:
            tasks, reserved_mem, per_task_start, launches, step_ms, stall_ms_per_block, stall_bwd_per_layer, used_fuse_window = _try_refine_task_starts(
                tasks, blocks, comm_model, ds_id_to_size, cfg, keep_layers)
        else:
            launches, groups_by_start = _build_comm_groups(tasks, comm_model, ds_id_to_size, cfg)
            step_ms, stall_ms_per_block, stall_bwd_per_layer, _ = _simulate_schedule_in_graph_order(
                blocks, groups_by_start, keep_layers)

        keep_ds_ids: List[int] = []
        for li in sorted(keep_layers):
            keep_ds_ids.extend([int(ds_id) for ds_id in layer_to_ds_ids[li] if int(ds_id) in ds_id_to_size])
        keep_ds_ids = sorted(set(keep_ds_ids))

        persistent_ds_ids: List[int] = []
        persistent_mem_bytes = 0
        if bool(cfg.set_persistent) and int(mem_budget_bytes) > 0:
            def _time_per_byte(ds_id: int) -> float:
                size = int(ds_id_to_size.get(int(ds_id), 0))
                if size <= 0:
                    return 0.0
                return float(ds_id_to_allgather_ms.get(int(ds_id), 0.0)) / float(size)

            candidates = sorted(
                keep_ds_ids,
                key=lambda d: (_time_per_byte(int(d)), float(ds_id_to_allgather_ms.get(int(d), 0.0)),
                               int(ds_id_to_size.get(int(d), 0)), int(d)),
                reverse=True,
            )
            for ds_id in candidates:
                size = int(ds_id_to_size.get(int(ds_id), 0))
                if size <= 0:
                    continue
                if persistent_mem_bytes + size > int(mem_budget_bytes):
                    continue
                persistent_ds_ids.append(int(ds_id))
                persistent_mem_bytes += int(size)

        schedule = {
            "version": 4,
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
            },
            "meta": {
                "lookahead_blocks": cfg.lookahead_blocks,
                "fuse_max_bytes": cfg.fuse_max_bytes,
                "fuse_deadline_window_blocks": int(used_fuse_window),
                "fuse_factor": float(cfg.fuse_factor),
                "inner_refine_max_iters": int(cfg.inner_refine_max_iters),
                "inner_refine_min_gain_ms": float(cfg.inner_refine_min_gain_ms),
                "mem_margin": cfg.mem_margin,
                "safety_margin_bytes": cfg.safety_margin_bytes,
                "total_mem_bytes": int(total_mem),
                "peak_mem_bytes": int(peak_mem),
                "mem_budget_bytes": int(mem_budget_bytes),
                "persistent_mem_bytes": int(persistent_mem_bytes),
                "include_unmapped_params": bool(cfg.include_unmapped_params),
                "extra_unmapped_ds_ids": len(extra_ds_id_to_layer),
                "set_persistent": bool(cfg.set_persistent),
                "keep_layers_hint": sorted(keep_layers_hint),
            },
        }
        schedule["schedule_hash"] = _schedule_hash(schedule)
        return schedule

    best = run_inner(set())
    best_step = float(best["predicted_step_ms"])
    keep_layers: Set[int] = set()

    if keep_layers_hint:
        hinted = run_inner(set(keep_layers_hint))
        hinted_step = float(hinted["predicted_step_ms"])
        if hinted_step + 1e-6 < best_step:
            keep_layers = set(keep_layers_hint)
            best = hinted
            best_step = hinted_step

    for _ in range(int(cfg.outer_keep_max_iters)):
        candidates = []
        for layer_idx in range(L):
            if layer_idx in keep_layers:
                continue
            layer_ds_ids = [int(ds_id) for ds_id in layer_to_ds_ids[layer_idx] if int(ds_id) in ds_id_to_size]
            if not layer_ds_ids:
                continue
            size_bytes = sum(ds_id_to_size[ds_id] for ds_id in layer_ds_ids)
            # Use measured backward wait as a cheap benefit proxy to seed candidates; trial schedules decide.
            benefit_ms = sum(ds_id_to_bwd_wait_ms.get(ds_id, 0.0) for ds_id in layer_ds_ids)
            span = (2 * L - 1 - layer_idx) - layer_idx
            cost = max(1, int(size_bytes) * int(span))
            candidates.append((benefit_ms / cost, benefit_ms, layer_idx))

        candidates.sort(reverse=True)
        if not candidates:
            break

        improved = False
        for _, benefit_ms, layer_idx in candidates[:int(cfg.outer_keep_top_n)]:
            if benefit_ms < float(cfg.outer_keep_min_gain_ms):
                continue
            trial_keep = set(keep_layers)
            trial_keep.add(int(layer_idx))
            trial = run_inner(trial_keep)
            trial_step = float(trial["predicted_step_ms"])
            if trial_step + 1e-6 < best_step - float(cfg.outer_keep_min_gain_ms):
                keep_layers = trial_keep
                best = trial
                best_step = trial_step
                improved = True
                break

        if not improved:
            break

    return best


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


def _maybe_set_persistent(persistent_ds_ids: Sequence[int]) -> None:
    ds_ids = [int(x) for x in (persistent_ds_ids or [])]
    if not ds_ids:
        return
    nz3 = get_deepcompile_handle()
    for ds_id in ds_ids:
        nz3.set_persistent(int(ds_id))
    log_rank0(f"[{NAME}] Set persistent buffers: {len(ds_ids)} ds_ids", enable=True)


def plan(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
         mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool):
    del create_inputs_fn, mem_budget, gm

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
        log_rank0(
            f"[{NAME}] Planned schedule: L={schedule['L']} keep_layers={len(schedule['keep_layers'])} keep_ds_ids={len(schedule['keep_ds_ids'])} launches={len(schedule['launches'])} mem_budget_bytes={schedule.get('meta',{}).get('mem_budget_bytes', None)} predicted_step_ms={schedule.get('predicted_step_ms', None)}",
            enable=True,
        )

    schedule = _broadcast_and_store_schedule(schedule)
    _maybe_set_persistent(schedule.get("persistent_ds_ids", []))
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


def _insert_prefetch(graph: Graph, graph_id: int, anchor: Node, pm: DSGraphParamManager,
                     ds_ids: Sequence[int], ds_id_to_dtype: Dict[int, torch.dtype], name: str) -> None:
    ds_id_to_param = _ds_id_to_param_node(graph, pm)
    filtered_ds_ids: List[int] = []
    filtered_params: List[Node] = []
    filtered_dtypes: List[torch.dtype] = []
    for ds_id in ds_ids:
        pn = ds_id_to_param.get(int(ds_id))
        if pn is None:
            continue
        filtered_ds_ids.append(int(ds_id))
        filtered_params.append(pn)
        filtered_dtypes.append(ds_id_to_dtype.get(int(ds_id), pm.params[pn.name].dtype if pn.name in pm.params else torch.float16))

    if not filtered_ds_ids:
        return

    with graph.inserting_before(anchor):
        graph.create_node("call_function",
                          torch.ops.dc.prefetch_params_fused.default,
                          args=(graph_id, filtered_params, filtered_ds_ids, filtered_dtypes),
                          name=name)


def _rewrite_graph_with_layer_comm(graph: Graph, graph_id: int, pm: DSGraphParamManager, mapping: _LayerMapping,
                                  sched: dict, bwd: bool, cfg: _SchedulerConfig) -> Optional[Graph]:
    L = int(sched["L"])
    K = 2 * L
    launches: Dict[int, List[List[int]]] = {int(k): v for k, v in sched.get("launches", {}).items()}

    keep_ds_ids: Set[int] = set(map(int, sched.get("keep_ds_ids", [])))

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

    def insert_prefetch_for_layer(layer: int) -> None:
        unified_k = layer if not bwd else (K - 1 - layer)
        groups = launches.get(int(unified_k), [])
        for gi, ds_ids in enumerate(groups):
            params_new: List[Node] = []
            ds_ids_new: List[int] = []
            dtypes_new: List[torch.dtype] = []
            for ds_id in ds_ids:
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
                                  args=(graph_id, params_new, ds_ids_new, dtypes_new),
                                  name=name)

    def insert_layer_comm(layer: int) -> None:
        ds_ids = [
            int(ds_id) for ds_id in mapping.layer_to_ds_ids[layer]
            if int(ds_id) in ds_id_to_ag and int(ds_id) in ds_id_to_wait
        ]
        for ds_id in sorted(ds_ids):
            ag_old = ds_id_to_ag[ds_id]
            wait_old = ds_id_to_wait[ds_id]
            if ag_old not in env:
                copy_node(ag_old)
            if wait_old not in env:
                copy_node(wait_old)

    moved_ag = set()
    moved_wait = set()
    for layer in range(L):
        for ds_id in mapping.layer_to_ds_ids[layer]:
            ds_id = int(ds_id)
            if ds_id in ds_id_to_ag and ds_id in ds_id_to_wait:
                moved_ag.add(ds_id_to_ag[ds_id])
                moved_wait.add(ds_id_to_wait[ds_id])

    changed = False
    for n in graph.nodes:
        if n.op == "placeholder":
            copy_node(n)
            continue

        if n in anchor_node_to_layer and cfg.rewrite_comm_ops:
            layer = int(anchor_node_to_layer[n])
            insert_prefetch_for_layer(layer)
            insert_layer_comm(layer)
            changed = True

        if n in moved_ag or n in moved_wait:
            # These comm nodes are re-inserted at the layer anchor.
            continue

        if (not bwd) and n.target == torch.ops.dc.release_param.default and _get_ds_id_from_release(n) in keep_ds_ids:
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


def _remove_forward_releases(graph: Graph, keep_ds_ids: Set[int]) -> bool:
    to_erase: List[Node] = []
    for n in graph.nodes:
        if n.target != torch.ops.dc.release_param.default:
            continue
        ds_id = _get_ds_id_from_release(n)
        if ds_id in keep_ds_ids:
            to_erase.append(n)

    changed = False
    for n in to_erase:
        inp = n.args[0]
        for user in list(n.users):
            user.replace_input_with(n, inp)
        graph.erase_node(n)
        changed = True
    return changed


def apply(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results, create_inputs_fn,
          mem_budget: float, param_manager: Dict[int, DSGraphParamManager], bwd: bool) -> GraphModule:
    del graph_order, profiling_results, create_inputs_fn, mem_budget

    if _LATEST_SCHEDULE is None or _LAYER_MAPPING is None or _CFG is None:
        return None

    sched = _LATEST_SCHEDULE
    mapping = _LAYER_MAPPING
    cfg = _CFG

    if sched.get("mapping_hash") != mapping.mapping_hash:
        raise RuntimeError(f"[{NAME}] schedule mapping_hash mismatch: schedule={sched.get('mapping_hash')} local={mapping.mapping_hash}")

    L = int(sched["L"])
    if L <= 0:
        return None

    pm = param_manager.get(graph_id)
    if pm is None:
        return None

    if cfg.rewrite_comm_ops:
        new_graph = _rewrite_graph_with_layer_comm(gm.graph, graph_id, pm, mapping, sched, bwd=bwd, cfg=cfg)
        if new_graph is None:
            return None
        gm.graph = new_graph
        return gm

    graph = gm.graph

    changed = False
    if not bwd:
        changed |= _remove_forward_releases(graph, set(map(int, sched.get("keep_ds_ids", []))))

    anchors = _find_layer_anchors(graph, mapping.ds_id_to_layer, L)
    ds_id_to_dtype = _ds_id_to_target_dtype(graph, pm)

    launches: Dict[int, List[List[int]]] = {int(k): v for k, v in sched.get("launches", {}).items()}

    if not bwd:
        for k in range(L):
            groups = launches.get(k)
            if not groups:
                continue
            anchor = anchors.get(k)
            if anchor is None:
                continue
            for gi, ds_ids in enumerate(groups):
                _insert_prefetch(graph,
                                 graph_id,
                                 anchor,
                                 pm,
                                 ds_ids,
                                 ds_id_to_dtype,
                                 name=f"gls_prefetch_fwd_k{k}_g{gi}")
                changed = True
    else:
        for k in range(L, 2 * L):
            groups = launches.get(k)
            if not groups:
                continue
            layer = 2 * L - 1 - k
            anchor = anchors.get(layer)
            if anchor is None:
                continue
            for gi, ds_ids in enumerate(groups):
                _insert_prefetch(graph,
                                 graph_id,
                                 anchor,
                                 pm,
                                 ds_ids,
                                 ds_id_to_dtype,
                                 name=f"gls_prefetch_bwd_k{k}_g{gi}")
                changed = True

    if not changed:
        return None

    return gm
