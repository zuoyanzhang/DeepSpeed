"""
Chunk linear pass: 对 allgather + gemm 进行按 K 维度分块并流水化。
"""
from typing import List, Tuple, Optional
from contextlib import nullcontext
from collections import deque
import math

import torch
from torch.fx import GraphModule, Node

from ..util import is_cast_op
from ..graph_param import DSGraphParamManager
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator

NAME = "chunk_gemm"

# TODO: 先手动设置chunksize，后面需要加上cost model来自动确定chunk size
DEFAULT_CHUNK_SIZE = 4096
# allgather/gemm 流水 lookahead 深度（提前发多少个 chunk 的通信）
CHUNK_LOOKAHEAD = 1
# 最大允许的分块数，防止过碎导致性能退化
MAX_CHUNKS = 2
# 每个通信包最小的 K 尺寸，过小则合并到前一块
MIN_CHUNK_K = 1024
# 单次 allgather 最多合并的 compute chunk 数（同一 K 长度才会打包）
MAX_PACKED_COMM_CHUNKS = 1
# 单次打包通信的 K 上限，避免一次 allgather 过大（单位：元素数，未乘 elems_per_k）
MAX_COMM_GROUP_K = DEFAULT_CHUNK_SIZE * MAX_PACKED_COMM_CHUNKS
# chunk 启动的最小 K，避免在小 K 上拆分
CHUNK_ENABLE_K_THRESHOLD = 8192
# 通信/计算耗时比的最低阈值（缺少 profile 则退化为大小阈值）
CHUNK_COMM_RATIO_THRESHOLD = 0.1
# allgather 张量尺寸过小则不做 chunk（字节）
CHUNK_MIN_AG_BYTES = 1 * 1024 * 1024
# 预留显存裕度，避免 chunk 额外 buffer 触发 OOM
CHUNK_MEM_MARGIN = 0.1
# 组合优化（prefetch + chunk）启用时的搜索/保守参数
COMBO_CANDIDATE_N_CHUNKS = (2, 4, 8)
COMBO_SPEEDUP_MARGIN = 0.05
COMBO_MIN_GAIN_MS = 0.2
COMBO_MM_TIME_FLOOR_MS = 0.05
COMBO_COMM_RATIO_THRESHOLD = 0.3
COMBO_PREFETCH_REMAIN_FRACTION = 0.3
COMBO_COMP_PENALTY_PER_EXTRA_CHUNK = 0.25
COMBO_LAUNCH_OVERHEAD_MS = 0.05
PREFETCH_MAX_FUSE_SIZE = int(1e9)
PREFETCH_MAX_BUFFERED_SIZE = int(4e9)


def _min_all_reduce_int(value: int) -> int:
    try:
        vals = torch.tensor([value],
                            device=torch.device(get_accelerator().current_device()),
                            dtype=torch.int64)
        dist.all_reduce(vals, dist.ReduceOp.MIN)
        return int(vals.item())
    except Exception:
        return value


def _infer_tensor_bytes_from_meta(node: Node) -> Optional[int]:
    tmeta = node.meta.get("tensor_meta", None) if hasattr(node, "meta") else None
    if tmeta is None or not hasattr(tmeta, "shape") or not hasattr(tmeta, "dtype"):
        return None
    shape = tmeta.shape
    if any((not isinstance(d, int)) or d <= 0 for d in shape):
        return None
    numel = 1
    for d in shape:
        numel *= d
    return numel * torch.tensor([], dtype=tmeta.dtype).element_size()

def _build_chunk_plan(K: int,
                      chunk_size: int,
                      max_chunks: Optional[int] = None,
                      min_chunk_k: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    返回 [(start_k, len_k), ...]，会合并过小尾块并限制分块个数。
    """
    if K <= 0:
        return []

    max_chunks = MAX_CHUNKS if max_chunks is None else max_chunks
    min_chunk_k = MIN_CHUNK_K if min_chunk_k is None else min_chunk_k

    eff_chunk = max(chunk_size, math.ceil(K / max_chunks))
    plan: List[Tuple[int, int]] = []

    cur = 0
    while cur < K:
        length = min(eff_chunk, K - cur)
        # 尾块太小则和前一块合并，避免极碎通信
        if cur + length >= K and plan and length < max(min_chunk_k, eff_chunk // 2):
            last_s, last_l = plan[-1]
            plan[-1] = (last_s, last_l + length)
            break
        plan.append((cur, length))
        cur += length

    # 进一步限制块数，必要时自尾部向前合并
    while len(plan) > max_chunks:
        prev_s, prev_l = plan[-2]
        last_s, last_l = plan[-1]
        plan[-2] = (prev_s, prev_l + last_l)
        plan.pop()

    return plan


def _pack_comm_groups(chunks: List[Tuple[int, int]],
                      max_packed: Optional[int] = None,
                      max_comm_group_k: Optional[int] = None) -> List[dict]:
    """
    将 compute chunk 打包成通信 group，仅合并长度一致的连续块。
    返回 [{start_k, chunk_len, total_k, chunks:[(s,l), ...]}]
    """
    max_packed = MAX_PACKED_COMM_CHUNKS if max_packed is None else max_packed
    max_comm_group_k = MAX_COMM_GROUP_K if max_comm_group_k is None else max_comm_group_k
    groups = []
    idx = 0
    while idx < len(chunks):
        start_k, length = chunks[idx]
        group_chunks = [(start_k, length)]
        total_k = length
        idx += 1
        while (idx < len(chunks)
               and len(group_chunks) < max_packed
               and chunks[idx][1] == length
               and total_k + length <= max_comm_group_k):
            group_chunks.append(chunks[idx])
            total_k += length
            idx += 1

        groups.append({
            "start_k": start_k,
            "chunk_len": length,
            "total_k": total_k,
            "chunks": group_chunks,
        })
    return groups


def _build_profile_maps(profiling_results, graph_id: int, bwd: bool):
    profile = profiling_results.get(graph_id, None) if profiling_results is not None else None
    if profile is None:
        return {}, {}, {}
    time_list = profile.bwd_time if bwd else profile.fwd_time
    size_list = profile.bwd_tensor_sizes if bwd else profile.fwd_tensor_sizes
    mem_list = profile.bwd_mem if bwd else profile.fwd_mem
    time_list = time_list if time_list is not None else []
    size_list = size_list if size_list is not None else []
    mem_list = mem_list if mem_list is not None else []
    time_map = {name: (device_time, wall_time) for name, device_time, wall_time in time_list}
    size_map = {name: size for name, size in size_list}
    mem_map = {name: (alloc, peak) for name, alloc, delta, peak in mem_list}
    return time_map, size_map, mem_map


def _unwrap_weight_to_allgather(node: Node):
    # 尝试从当前节点开始向上查找 allgather 节点。
    # 如果找到 allgather 节点，则返回 (allgather_node, transforms)；否则返回 None。
    #
    # 允许穿透的算子（保持张量值不变或仅做 dtype/布局变换）包括：
    #   - dc.wait_allgather
    #   - dtype cast（prims.convert_element_type / aten._to_copy(dtype=...)）
    #   - aten.permute
    #   - aten.view / aten.reshape
    #   - aten.contiguous
    #
    # transforms 记录从 allgather 输出到原始权重的变换，按“靠近权重 -> 靠近 allgather”的顺序
    transforms = []
    chain_nodes = []
    cur = node
    for _ in range(3):
        if cur.op != "call_function":
            break

        tgt = cur.target
        chain_nodes.append(cur)

        # dc.wait_allgather（单独处理，不计入 transforms）
        if tgt == torch.ops.dc.wait_allgather.default:
            cur = cur.args[0]
            continue

        # dtype cast
        is_cast, cast_dtype = is_cast_op(cur)
        if is_cast:
            transforms.append({"type": "cast", "target": tgt, "dtype": cast_dtype})
            cur = cur.args[0]
            continue

        # 纯布局/形状变换：permute/view/reshape/contiguous
        if tgt in (
            torch.ops.aten.permute.default,
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.contiguous.default,
        ):
            meta = {"type": tgt._schema.name.split("::")[-1]}  # view/reshape/_unsafe_view/permute/contiguous
            if tgt == torch.ops.aten.permute.default:
                meta["dims"] = cur.args[1]
            elif tgt in (
                torch.ops.aten.view.default,
                torch.ops.aten.reshape.default,
                torch.ops.aten._unsafe_view.default,
            ):
                meta["shape"] = cur.args[1]
                meta["target"] = tgt
            transforms.append(meta)
            cur = cur.args[0]
            continue

        # 其他算子认为是边界
        break

    if cur.op == "call_function" and cur.target == torch.ops.dc.allgather_param.default:
        chain_nodes.append(cur)
        return cur, transforms, chain_nodes
    return None

def _extract_linear_pattern(node: Node):
    """
    Try to unify different linear-ish patterns to (x, w, bias, beta, alpha, mm_like_node).
    mm_like_node 是真正的 gemm 节点（用于在外层 add 场景下删除冗余 gemm）。
    """
    bias = None
    beta = 1.0
    alpha = 1.0

    # Bias add on top of mm/bmm/matmul
    if node.target == torch.ops.aten.add.Tensor and len(node.args) >= 2:
        a, b = node.args[:2]
        mm_like = None
        bias_candidate = None
        for first, second in ((a, b), (b, a)):
            # first/second可能是常量float/int，需要先判断是否是Node
            if isinstance(first, Node) and first.op == "call_function" and first.target in {
                    torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default,
                    torch.ops.aten.matmul.default
            }:
                mm_like = first
                bias_candidate = second
                break
        if mm_like is None:
            return None
        inner = _extract_linear_pattern(mm_like)
        if inner is None:
            return None
        x, w, inner_bias, beta, alpha, _ = inner
        bias = inner_bias if inner_bias is not None else bias_candidate
        # mm_like 是外层 add 里的真正 gemm 节点
        return x, w, bias, beta, alpha, mm_like

    if node.target == torch.ops.aten.mm.default and len(node.args) == 2:
        x, w = node.args
        return x, w, bias, beta, alpha, node

    if node.target == torch.ops.aten.addmm.default and len(node.args) >= 3:
        bias, x, w = node.args[:3]
        if len(node.args) >= 4:
            beta = float(node.args[3])
        if len(node.args) >= 5:
            alpha = float(node.args[4])
        return x, w, bias, beta, alpha, node

    if node.target == torch.ops.aten.bmm.default and len(node.args) == 2:
        x, w = node.args
        return x, w, bias, beta, alpha, node

    if node.target == torch.ops.aten.matmul.default and len(node.args) == 2:
        x, w = node.args
        x_meta = x.meta.get("tensor_meta", None)
        w_meta = w.meta.get("tensor_meta", None)
        if x_meta is None or w_meta is None:
            return None
        if len(x_meta.shape) in (2, 3) and len(w_meta.shape) in (2, 3):
            return x, w, bias, beta, alpha, node
        return None

    return None

def _replace_uses(old: Node, new: Node):
    # 替换old节点的所有使用者，将其输入从old替换为new
    for user in list(old.users):
        user.replace_input_with(old, new)


def _adjust_shape_for_chunk(shape, k_full: int, chunk_len: int, elements_per_k: int):
    # 将 view/reshape 的目标 shape 中与 k_full 对应的维度替换为 chunk_len。
    # 如果存在 -1，则根据 chunk 总元素数推导。
    flat_shape = []
    unknown_idx = None
    known_prod = 1
    for i, d in enumerate(shape):
        if isinstance(d, int):
            if d == -1:
                unknown_idx = i
                flat_shape.append(-1)
                continue
            new_d = chunk_len if d == k_full else d
            flat_shape.append(new_d)
            known_prod *= new_d
        else:
            return None

    total_elems = chunk_len * elements_per_k
    if unknown_idx is not None:
        if known_prod == 0 or total_elems % known_prod != 0:
            return None
        inferred = total_elems // known_prod
        flat_shape[unknown_idx] = inferred
    return flat_shape


def _apply_transforms_on_chunk(graph,
                               chunk_node: Node,
                               transforms,
                               w_shape,
                               w_dim,
                               chunk_len: int,
                               base_shape,
                               base_k_dim: int,
                               elements_per_k: int):
    # 根据记录的 transforms，从 allgather chunk 的输出重建到 mm 权重的形状/布局
    if w_dim < 0 or w_dim >= len(w_shape):
        return None
    if base_k_dim < 0 or base_k_dim >= len(base_shape):
        return None
    k_full = w_shape[w_dim]
    target_shape = list(w_shape)
    target_shape[w_dim] = chunk_len

    # 先将扁平 chunk reshape 成 allgather 输出的形状（只在 K 轴替换 chunk_len）
    chunk_base_shape = list(base_shape)
    if any(not isinstance(d, int) for d in chunk_base_shape):
        return None
    if any(d <= 0 for d in chunk_base_shape):
        return None
    chunk_base_shape[base_k_dim] = chunk_len

    chunk_total = chunk_len * elements_per_k
    prod = 1
    for d in chunk_base_shape:
        prod *= d
    if prod != chunk_total:
        return None

    cur = graph.create_node("call_function",
                            torch.ops.aten.view.default,
                            args=(chunk_node, tuple(chunk_base_shape)),
                            kwargs={})
    current_rank = len(chunk_base_shape)
    for t in reversed(transforms):
        t_type = t.get("type")
        if t_type == "cast":
            target = t["target"]
            dtype = t["dtype"]
            if target == torch.ops.prims.convert_element_type.default:
                cur = graph.create_node("call_function", target, args=(cur, dtype), kwargs={})
            else:
                cur = graph.create_node("call_function", target, args=(cur,), kwargs={"dtype": dtype})
        elif t_type == "permute":
            dims = t["dims"]
            if len(dims) != current_rank:
                return None
            cur = graph.create_node("call_function", 
                                    torch.ops.aten.permute.default, 
                                    args=(cur, dims), kwargs={})
        elif t_type in ("view", "reshape", "_unsafe_view"):
            target = t.get("target", torch.ops.aten.view.default)
            target_shape = t.get("shape", None)
            if target_shape is None:
                return None
            adjusted_shape = _adjust_shape_for_chunk(target_shape, k_full, chunk_len, elements_per_k)
            if adjusted_shape is None:
                return None
            cur = graph.create_node("call_function", target, args=(cur, tuple(adjusted_shape)), kwargs={})
            current_rank = len(adjusted_shape)
        elif t_type == "contiguous":
            cur = graph.create_node("call_function", 
                                    torch.ops.aten.contiguous.default, 
                                    args=(cur,), 
                                    kwargs={})
        else:
            return None
    # 最终 reshape 成与原始权重相同的布局（chunk 维度替换为 chunk_len）
    cur = graph.create_node("call_function", 
                            torch.ops.aten.view.default, 
                            args=(cur, tuple(target_shape)), 
                            kwargs={})
    return cur


def _prune_unused_call_function_nodes(graph, candidates):
    # 清理掉指定集合中没有使用者的 call_function 节点，避免冗余 allgather/transform 被执行。
    removed = True
    candidates = set(candidates)
    while removed:
        removed = False
        for n in list(candidates):
            if n.op == "call_function" and len(n.users) == 0:
                graph.erase_node(n)
                candidates.remove(n)
                removed = True


def _chunk_allgather_and_gemm(graph, mm_node: Node, x: Node, w: Node, bias: Optional[Node], beta: float,
                              alpha: float, mm_like: Node, ag_node: Node, transforms, chunk_size: int,
                              graph_id: int, lookahead: int = CHUNK_LOOKAHEAD, max_chunks: int = MAX_CHUNKS,
                              max_packed: int = MAX_PACKED_COMM_CHUNKS, min_chunk_k: int = MIN_CHUNK_K,
                              cross_prefetch_anchor: Optional[Node] = None, time_map=None, size_map=None,
                              mem_map=None, max_mem_budget: Optional[int] = None, guard: bool = True) -> Tuple[bool, Optional[Node]]:
    def map_k_dim_to_ag_axis(w_dim: int, transforms):
        axis = w_dim
        for t in transforms:
            t_type = t.get("type")
            if t_type in ("cast", "contiguous"):
                continue
            if t_type == "permute":
                dims = t["dims"]
                if axis >= len(dims):
                    return None
                axis = dims[axis]
            elif t_type in ("view", "reshape", "_unsafe_view"):
                # 形状变化无法安全映射，回退
                return None
            else:
                return None
        return axis

    x_meta = x.meta.get("tensor_meta", None)
    w_meta = w.meta.get("tensor_meta", None)
    if x_meta is None or w_meta is None:
        return False

    w_shape = tuple(w_meta.shape)
    if any((not isinstance(d, int)) or d <= 0 for d in w_shape):
        return False
    w_dim = len(w_shape) - 2
    if w_dim < 0:
        return False
    K = w_shape[w_dim]
    if K is None:
        return False

    ag_meta = ag_node.meta.get("tensor_meta") if hasattr(ag_node, "meta") else None
    base_shape = tuple(ag_meta.shape) if ag_meta is not None and hasattr(ag_meta, "shape") else None
    if base_shape is None:
        # 如果没有 meta，尝试仅由 permute/cast/contiguous 推回 allgather 的 shape。
        only_permute_like = all(t.get("type") in ("permute", "cast", "contiguous") for t in transforms)
        if only_permute_like:
            inferred = list(w_shape)
            for t in transforms:
                if t.get("type") == "permute":
                    dims = t["dims"]
                    if len(dims) != len(inferred):
                        return False
                    inv = [0] * len(dims)
                    for i, d in enumerate(dims):
                        inv[d] = i
                    inferred = [inferred[i] for i in inv]
            base_shape = tuple(inferred)
        else:
            # 有 view/reshape 且缺少 meta，无法安全处理
            if any(t.get("type") == "permute" for t in transforms):
                return False
            base_shape = w_shape

    base_k_dim = map_k_dim_to_ag_axis(w_dim, transforms)
    if base_k_dim is None or base_k_dim < 0 or base_k_dim >= len(base_shape):
        return False
    base_k = base_shape[base_k_dim]
    if not isinstance(base_k, int) or base_k != K:
        # K 轴大小与 allgather 输出不一致，无法安全分块
        return False
    # 计算每个 K 元素对应的连续元素数（假设 allgather 输出是行主序连续的）。
    if any((not isinstance(d, int)) or d <= 0 for d in base_shape):
        return False
    prefix = 1
    for d in base_shape[:base_k_dim]:
        prefix *= d
    suffix = 1
    for d in base_shape[base_k_dim + 1:]:
        suffix *= d
    elems_per_k = prefix * suffix

    # 如果前缀>1，使用 stride/chunk_count 方式逐行提取 K 段
    chunk_count = prefix
    stride_elems = base_k * suffix if chunk_count > 1 else 0

    ds_id = ag_node.args[2]
    dtype_kw = ag_node.kwargs.get("dtype", None)
    ag_input = ag_node.args[0]
    ag_graph_id = ag_node.args[1] if len(ag_node.args) > 1 else graph_id
    x_dim = len(x_meta.shape) - 1
    matmul_target = torch.ops.aten.mm.default if len(x_meta.shape) == 2 \
                    and len(w_shape) == 2 else torch.ops.aten.bmm.default

    acc = None
    max_chunk_size = max(chunk_size, math.ceil(K / max_chunks)) if K > 0 else chunk_size
    if K < max(max_chunk_size, CHUNK_ENABLE_K_THRESHOLD):
        return False, None

    chunk_plan = _build_chunk_plan(K, max_chunk_size, max_chunks=max_chunks, min_chunk_k=min_chunk_k)
    if len(chunk_plan) <= 1:
        return False, None
    comm_groups = _pack_comm_groups(chunk_plan,
                                    max_packed=max_packed,
                                    max_comm_group_k=max_chunk_size * max_packed)

    if guard:
        # 预判通信占比与显存约束，低收益或风险 OOM 时直接跳过
        mm_time = time_map.get(mm_like.name, (0.0, 0.0))[0] if time_map else 0.0
        ag_time = time_map.get(ag_node.name, (0.0, 0.0))[0] if time_map else 0.0
        if ag_time == 0.0 and time_map:
            wait_name = f"{ag_node.name}_wait"
            ag_time = time_map.get(wait_name, (0.0, 0.0))[0]

        if mm_time > 0.0 and ag_time > 0.0:
            ratio = ag_time / max(mm_time, 1e-6)
            if ratio < CHUNK_COMM_RATIO_THRESHOLD:
                return False, None
        else:
            ag_size = size_map.get(ag_node.name, 0) if size_map else 0
            if ag_size == 0:
                ag_size = base_k * elems_per_k * torch.tensor([], dtype=w_meta.dtype).element_size()
            if ag_size < CHUNK_MIN_AG_BYTES:
                return False, None

    elem_size = torch.tensor([], dtype=w_meta.dtype).element_size()
    max_group_k = max(g["total_k"] for g in comm_groups) if len(comm_groups) > 0 else 0
    extra_bytes = lookahead * max_group_k * elems_per_k * elem_size
    if max_mem_budget is None:
        total_mem = get_accelerator().total_memory()
        max_mem_budget = int(total_mem * (1 - CHUNK_MEM_MARGIN))
    current_peak = mem_map.get(mm_like.name, (0, 0))[1] if mem_map else 0
    full_ag_bytes = size_map.get(ag_node.name, 0) if size_map else 0
    if full_ag_bytes == 0:
        full_ag_bytes = base_k * elems_per_k * elem_size
    # chunk 后理论上不需要一次性持有 full_ag_bytes，而是最多持有 extra_bytes（lookahead 窗口内）
    est_peak_after_chunk = max(current_peak - full_ag_bytes, 0) + extra_bytes
    if est_peak_after_chunk > max_mem_budget:
        return False, None

    def make_kwargs():
        return {"dtype": dtype_kw} if dtype_kw is not None else {}

    with graph.inserting_before(mm_node):
        created_nodes = []
        prefetch_q = deque()
        group_idx = 0

        def emit_group(g_idx: int, anchor: Optional[Node] = None):
            group = comm_groups[g_idx]
            offset_elems = group["start_k"] * suffix
            length_elems = group["total_k"] * suffix
            ctx = graph.inserting_before(anchor) if anchor is not None else nullcontext()
            with ctx:
                ag_chunk = graph.create_node("call_function",
                                             torch.ops.dc.allgather_param_chunk.default,
                                             args=(ag_input, ag_graph_id, ds_id, offset_elems, length_elems,
                                                   stride_elems, chunk_count),
                                             kwargs=make_kwargs())
            created_nodes.append(ag_chunk)
            return {
                "group": group,
                "offset": offset_elems,
                "length": length_elems,
                "ag": ag_chunk,
            }

        # 跨层预发：在上一层 anchor 处抢先发首组
        if cross_prefetch_anchor is not None and len(comm_groups) > 0:
            prefetched = emit_group(0, anchor=cross_prefetch_anchor)
            prefetch_q.append(prefetched)
            group_idx = 1

        # 预发 lookahead 个通信组
        while group_idx < len(comm_groups) and len(prefetch_q) < lookahead:
            info = emit_group(group_idx)
            prefetch_q.append(info)
            group_idx += 1

        # 消费队列，同时保持窗口内通信在跑
        while prefetch_q:
            cur = prefetch_q.popleft()

            # 尽量保持窗口填满
            while group_idx < len(comm_groups) and len(prefetch_q) < lookahead:
                info = emit_group(group_idx)
                prefetch_q.append(info)
                group_idx += 1

            wait_chunk = graph.create_node("call_function",
                                           torch.ops.dc.wait_allgather_chunk.default,
                                           args=(cur["ag"], ag_graph_id, ds_id, cur["offset"], cur["length"],
                                                 stride_elems, chunk_count),
                                           kwargs={})
            created_nodes.append(wait_chunk)

            group_len_elems = cur["length"]
            grouped_view = graph.create_node("call_function",
                                             torch.ops.aten.view.default,
                                             args=(wait_chunk, (chunk_count, group_len_elems)),
                                             kwargs={})
            created_nodes.append(grouped_view)

            group_start_k = cur["group"]["start_k"]
            for chunk_start_k, chunk_k in cur["group"]["chunks"]:
                local_offset = (chunk_start_k - group_start_k) * suffix
                local_end = local_offset + chunk_k * suffix

                chunk_slice = graph.create_node("call_function",
                                                torch.ops.aten.slice.Tensor,
                                                args=(grouped_view, 1, local_offset, local_end),
                                                kwargs={})
                created_nodes.append(chunk_slice)

                chunk_flat = graph.create_node("call_function",
                                               torch.ops.aten.reshape.default,
                                               args=(chunk_slice, (chunk_k * elems_per_k, )),
                                               kwargs={})
                created_nodes.append(chunk_flat)

                w_chunk = _apply_transforms_on_chunk(graph, chunk_flat, transforms, w_shape, w_dim, chunk_k,
                                                     base_shape, base_k_dim, elems_per_k)
                if w_chunk is None:
                    # 按逆序清理本轮临时节点，避免因依赖关系残留导致 erase 抛错
                    for n in reversed(created_nodes):
                        if n in graph.nodes:
                            graph.erase_node(n)
                    return False
                created_nodes.append(w_chunk)

                x_slice = graph.create_node("call_function",
                                            torch.ops.aten.slice.Tensor,
                                            args=(x, x_dim, chunk_start_k, chunk_start_k + chunk_k),
                                            kwargs={})
                created_nodes.append(x_slice)

                y_i = graph.create_node("call_function", matmul_target, args=(x_slice, w_chunk), kwargs={})
                created_nodes.append(y_i)

                if acc is None:
                    acc = y_i
                else:
                    acc = graph.create_node("call_function",
                                            torch.ops.aten.add.Tensor,
                                            args=(acc, y_i),
                                            kwargs={})
                    created_nodes.append(acc)

        if alpha != 1.0:
            acc = graph.create_node("call_function",
                                    torch.ops.aten.mul.Tensor,
                                    args=(acc, alpha),
                                    kwargs={})

        if bias is not None or beta != 0.0:
            if bias is None:
                bias_scaled = graph.create_node("call_function",
                                                torch.ops.aten.mul.Tensor,
                                                args=(acc, 0.0),
                                                kwargs={})
            else:
                if beta != 1.0:
                    bias_scaled = graph.create_node("call_function",
                                                    torch.ops.aten.mul.Tensor,
                                                    args=(bias, beta),
                                                    kwargs={})
                else:
                    bias_scaled = bias
            acc = graph.create_node("call_function", 
                                    torch.ops.aten.add.Tensor, 
                                    args=(bias_scaled, acc), 
                                    kwargs={})

    _replace_uses(mm_node, acc)
    graph.erase_node(mm_node)
    if mm_like is not mm_node and len(mm_like.users) == 0:
        graph.erase_node(mm_like)
    return True, acc

def chunk_gemm(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
               create_inputs_fn, mem_budget: float, 
               param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    # 先只对前向图做chunk
    if bwd:
        return gm
    
    # NOTE: 打印graph
    # if dist.get_rank() == 0:
    #     print(f"[chunk_gemm] graph_id={graph_id} BEFORE:\n{gm.graph}")

    graph = gm.graph
    rewritten = False
    time_map, size_map, mem_map = _build_profile_maps(profiling_results, graph_id, bwd)
    total_mem = get_accelerator().total_memory()
    max_mem_budget = int(total_mem * (1 - CHUNK_MEM_MARGIN))
    try:
        vals = torch.tensor([max_mem_budget], device=torch.device(get_accelerator().current_device()))
        dist.all_reduce(vals, dist.ReduceOp.MIN)
        max_mem_budget = int(vals.item())
    except Exception:
        pass
    # 通信预测器（size -> seconds），用于在无 prefetch 时做保守选择/参数搜索
    try:
        from ..profilers.comm_profile import create_predictor
        comm_predictor = create_predictor()
    except Exception:
        comm_predictor = None

    ds_id_to_bytes = {}
    for n in graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.dc.allgather_param.default:
            ds_id = n.args[2]
            sz = size_map.get(n.name, 0)
            if sz == 0:
                inferred = _infer_tensor_bytes_from_meta(n)
                sz = inferred if inferred is not None else 0
            ds_id_to_bytes[ds_id] = sz
    cross_prefetch_anchor = None

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        extracted = _extract_linear_pattern(node)
        if extracted is None:
            continue
        x, w, bias, beta, alpha, mm_like = extracted

        unwrap_res = _unwrap_weight_to_allgather(w)
        if unwrap_res is None:
            continue
        ag, transforms, chain_nodes = unwrap_res
        ds_id = ag.args[2]
        weight_bytes = ds_id_to_bytes.get(ds_id, 0)
        mm_time_ms = time_map.get(mm_like.name, (0.0, 0.0))[0]
        w_meta = w.meta.get("tensor_meta", None)
        K = w_meta.shape[-2] if w_meta is not None and hasattr(w_meta, "shape") and len(w_meta.shape) >= 2 else None
        if (comm_predictor is None or not isinstance(K, int) or K <= 0 or weight_bytes <= 0 or mm_time_ms <= 0.0):
            # 无法做可靠决策时回退到固定策略（由 _chunk_allgather_and_gemm 的 guard 做进一步过滤）
            plan = {
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "lookahead": CHUNK_LOOKAHEAD,
                "max_chunks": MAX_CHUNKS,
                "max_packed": MAX_PACKED_COMM_CHUNKS,
                "min_chunk_k": MIN_CHUNK_K,
            }
            use_guard = True
        else:
            # 无 prefetch 时的“残余等待”≈全通信
            comm_total_ms = float(comm_predictor(weight_bytes)) * 1000.0
            plan = _choose_chunk_plan_for_mm(K=K,
                                             weight_bytes=weight_bytes,
                                             mm_time_ms=mm_time_ms,
                                             prefetch_residual_wait_ms=comm_total_ms,
                                             comm_predictor=comm_predictor,
                                             mem_peak_bytes=mem_map.get(mm_like.name, (0, 0))[1],
                                             max_mem_budget=max_mem_budget)
            if plan is None:
                continue
            use_guard = False

        did, acc = _chunk_allgather_and_gemm(graph,
                                             node,
                                             x,
                                             w,
                                             bias,
                                             beta,
                                             alpha,
                                             mm_like,
                                             ag,
                                             transforms,
                                             plan["chunk_size"],
                                             graph_id,
                                             plan["lookahead"],
                                             max_chunks=plan["max_chunks"],
                                             max_packed=plan["max_packed"],
                                             min_chunk_k=plan["min_chunk_k"],
                                             cross_prefetch_anchor=cross_prefetch_anchor,
                                             time_map=time_map,
                                             size_map=size_map,
                                             mem_map=mem_map,
                                             max_mem_budget=max_mem_budget,
                                             guard=use_guard)
        if did:
            # 移除旧的 allgather / wait / view 链路（仅在它们没有其他使用者时）
            _prune_unused_call_function_nodes(graph, chain_nodes)
            cross_prefetch_anchor = acc

        rewritten = rewritten or did

    if rewritten:
        graph.lint()
        gm.recompile()
        
        # NOTE: 打印chunk后graph
        # if dist.get_rank() == 0:
        #     print(f"[chunk_gemm] graph_id={graph_id} AFTER:\n{gm.graph}")

    return gm


def _plan_prefetch_forward_elems(graph, graph_id: int, profiling_results, bwd: bool, max_mem: int):
    """
    复用 prefetch.pass 的核心分组/插入点逻辑，但只返回“带 prefetch 插入点”的前向元素序列：
    List[Union[Node, List[Node]]]，其中 List[Node] 表示会插入一次 prefetch_params_fused。
    """
    try:
        from ..profilers.comm_profile import create_predictor
    except Exception:
        return [], {}, None

    profile = profiling_results.get(graph_id, None) if profiling_results is not None else None
    if profile is None:
        return [], {}, None

    mem = profile.bwd_mem if bwd else profile.fwd_mem
    tensor_sizes = profile.bwd_tensor_sizes if bwd else profile.fwd_tensor_sizes
    mem = mem if mem is not None else []
    tensor_sizes = tensor_sizes if tensor_sizes is not None else []

    mem_dict = {name: (alloc_mem, peak) for name, alloc_mem, delta, peak in mem}
    tensor_size_dict = {name: size for name, size in tensor_sizes}

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
            if i >= len(order_rev) - 1:
                break
            next_node = order_rev[i + 1]
            next_alloc_mem, next_peak = mem_dict.get(next_node.name, (0, 0))

            while next_peak + ag_tensor_size_sum > max_mem or ag_tensor_size_sum > PREFETCH_MAX_BUFFERED_SIZE:
                if len(prefetch_ag_groups) > 0:
                    fused_ag_nodes = prefetch_ag_groups.pop(0)
                    total_ag_tensor_size = sum([tensor_size_dict.get(ag_node.name, 0) for ag_node in fused_ag_nodes])
                    ag_tensor_size_sum -= total_ag_tensor_size
                    new_order_rev.append(fused_ag_nodes)
                elif len(prefetch_ags) > 0:
                    prefetch_ag_groups.append(prefetch_ags)
                    prefetch_ags = []
                else:
                    break

            if node.target == torch.ops.dc.allgather_param.default:
                node_size = tensor_size_dict.get(node.name, 0)
                if node_size <= 0:
                    # 缺少尺寸信息时不参与预取分组（保持默认行为）
                    pass
                else:
                    current_ag_size = sum([tensor_size_dict.get(ag_node.name, 0) for ag_node in prefetch_ags])
                    pred_time_current = comm_predictor(current_ag_size)
                    pred_time_next = comm_predictor(node_size)
                    pred_time_fused = comm_predictor(current_ag_size + node_size)

                    do_fuse = max(pred_time_current, pred_time_next) * 1.2 > pred_time_fused and (
                        current_ag_size + node_size) < PREFETCH_MAX_FUSE_SIZE

                    if len(prefetch_ags) > 0 and not do_fuse:
                        prefetch_ag_groups.append(prefetch_ags)
                        prefetch_ags = []
                    prefetch_ags.append(node)
                    ag_tensor_size_sum += node_size

        new_order_rev.append(node)

        if (node.op != "placeholder"
                and node.target != torch.ops.dc.reload_parameter) and order_rev[i + 1].op == "placeholder":
            for ag_group in prefetch_ag_groups:
                new_order_rev.append(ag_group)
                ag_tensor_size_sum -= sum([tensor_size_dict.get(ag_node.name, 0) for ag_node in ag_group])
            if len(prefetch_ags) > 0:
                new_order_rev.append(prefetch_ags)
                ag_tensor_size_sum -= sum([tensor_size_dict.get(ag_node.name, 0) for ag_node in prefetch_ags])
            ag_tensor_size_sum = 0

    return list(reversed(new_order_rev)), tensor_size_dict, comm_predictor


def _simulate_prefetch_residual_waits(forward_elems, tensor_size_dict, comm_predictor, time_map):
    """
    基于“按 prefetch.pass 的插入点”做一个粗略时间线模拟，输出每个 ds_id 在其 wait_allgather 处的残余等待（ms）。
    注意：这是近似模型，用于做 chunk 是否值得的保守决策。
    """
    if comm_predictor is None:
        return {}

    ds_id_to_bytes = {}
    for elem in forward_elems:
        if isinstance(elem, Node) and elem.op == "call_function" and elem.target == torch.ops.dc.allgather_param.default:
            ds_id = elem.args[2]
            ds_id_to_bytes[ds_id] = tensor_size_dict.get(elem.name, 0)

    t_ms = 0.0
    prefetch_done_at_ms = {}
    residual_wait_ms = {}

    for elem in forward_elems:
        if isinstance(elem, list):
            group_bytes = sum([tensor_size_dict.get(n.name, 0) for n in elem])
            group_ms = float(comm_predictor(group_bytes)) * 1000.0 if group_bytes > 0 else 0.0
            for ag_node in elem:
                ds_id = ag_node.args[2]
                prefetch_done_at_ms[ds_id] = t_ms + group_ms
            continue

        node = elem
        if not isinstance(node, Node) or node.op != "call_function":
            t_ms += time_map.get(getattr(node, "name", ""), (0.0, 0.0))[0]
            continue

        if node.target == torch.ops.dc.allgather_param.default:
            # 视作异步 launch，避免与 wait 重复计入通信时间
            continue

        if node.target == torch.ops.dc.wait_allgather.default:
            ds_id = node.args[2]
            param_bytes = ds_id_to_bytes.get(ds_id, 0)
            comm_ms = float(comm_predictor(param_bytes)) * 1000.0 if param_bytes > 0 else 0.0
            if ds_id in prefetch_done_at_ms:
                wait_ms = max(0.0, prefetch_done_at_ms[ds_id] - t_ms)
            else:
                wait_ms = comm_ms
            residual_wait_ms[ds_id] = wait_ms
            t_ms += wait_ms
            continue

        t_ms += time_map.get(node.name, (0.0, 0.0))[0]

    return residual_wait_ms


def _choose_chunk_plan_for_mm(K: int,
                              weight_bytes: int,
                              mm_time_ms: float,
                              prefetch_residual_wait_ms: float,
                              comm_predictor,
                              mem_peak_bytes: int,
                              max_mem_budget: int):
    if comm_predictor is None:
        return None
    if K < CHUNK_ENABLE_K_THRESHOLD:
        return None
    if weight_bytes <= 0:
        return None
    if mm_time_ms <= COMBO_MM_TIME_FLOOR_MS:
        return None

    comm_total_ms = float(comm_predictor(weight_bytes)) * 1000.0
    if comm_total_ms <= 0.0:
        return None
    # 通信占比太低时不要 chunk（chunk 会引入额外 GEMM/accumulate 开销）
    if (comm_total_ms / max(mm_time_ms, 1e-6)) < COMBO_COMM_RATIO_THRESHOLD:
        return None

    base_cost = mm_time_ms + prefetch_residual_wait_ms
    if prefetch_residual_wait_ms <= 0.0:
        return None
    # 如果 prefetch 已经覆盖了大部分通信，chunk 的边际收益很小，通常得不偿失
    if prefetch_residual_wait_ms < comm_total_ms * COMBO_PREFETCH_REMAIN_FRACTION:
        return None

    best = None
    for n_chunks in COMBO_CANDIDATE_N_CHUNKS:
        if n_chunks > MAX_CHUNKS:
            continue
        chunk_k = math.ceil(K / n_chunks)
        if chunk_k < MIN_CHUNK_K or chunk_k >= K:
            continue

        chunk_bytes = int(weight_bytes * (chunk_k / K))
        comm_chunk_ms = float(comm_predictor(chunk_bytes)) * 1000.0 if chunk_bytes > 0 else 0.0

        penalty = 1.0 + COMBO_COMP_PENALTY_PER_EXTRA_CHUNK * (n_chunks - 1)
        comp_chunk_ms = (mm_time_ms / n_chunks) * penalty
        total_comp_ms = comp_chunk_ms * n_chunks

        residual_per_extra = max(0.0, comm_chunk_ms - comp_chunk_ms)
        chunk_total_ms = comm_chunk_ms + total_comp_ms + (n_chunks - 1) * residual_per_extra + COMBO_LAUNCH_OVERHEAD_MS * n_chunks

        est_peak_after_chunk = max(mem_peak_bytes - weight_bytes, 0) + chunk_bytes * CHUNK_LOOKAHEAD
        if est_peak_after_chunk > max_mem_budget:
            continue

        if best is None or chunk_total_ms < best["cost_ms"]:
            best = {
                "cost_ms": chunk_total_ms,
                "chunk_size": chunk_k,
                "max_chunks": n_chunks,
                "lookahead": CHUNK_LOOKAHEAD,
                "max_packed": MAX_PACKED_COMM_CHUNKS,
                "min_chunk_k": MIN_CHUNK_K,
            }

    if best is None:
        return None

    gain_ms = base_cost - best["cost_ms"]
    if gain_ms < max(COMBO_MIN_GAIN_MS, base_cost * COMBO_SPEEDUP_MARGIN):
        return None

    return best


def chunk_gemm_combo(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
                     create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    """
    当与 prefetch 同时启用时使用的组合优化版本：
      1) 先用 prefetch.pass 的同款策略估算每个参数的“prefetch 后残余 wait”
      2) 只对残余 wait 足够大且模型预测能带来净收益的层启用 chunk，并自动挑选 chunk 参数
      3) 不做跨层 cross_prefetch（交给 prefetch.pass 负责层间重叠）
    """
    if bwd:
        return gm

    graph = gm.graph
    time_map, size_map, mem_map = _build_profile_maps(profiling_results, graph_id, bwd)
    if not time_map:
        return gm

    # 与 prefetch.pass 一致的显存预算（全 rank 取最小）
    max_mem = int(get_accelerator().total_memory() * (1 - CHUNK_MEM_MARGIN))
    max_mem = _min_all_reduce_int(max_mem)

    forward_elems, tensor_size_dict, comm_predictor = _plan_prefetch_forward_elems(
        graph, graph_id, profiling_results, bwd, max_mem)
    residual_wait_ms = _simulate_prefetch_residual_waits(forward_elems, tensor_size_dict, comm_predictor, time_map)

    # 建立 ds_id -> allgather bytes
    ds_id_to_bytes = {}
    for n in graph.nodes:
        if n.op == "call_function" and n.target == torch.ops.dc.allgather_param.default:
            ds_id = n.args[2]
            sz = size_map.get(n.name, 0)
            if sz == 0:
                sz = tensor_size_dict.get(n.name, 0)
            if sz == 0:
                inferred = _infer_tensor_bytes_from_meta(n)
                sz = inferred if inferred is not None else 0
            ds_id_to_bytes[ds_id] = sz

    # 先决策：哪些 ds_id 的 GEMM 需要 chunk，以及参数
    selected = {}
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        extracted = _extract_linear_pattern(node)
        if extracted is None:
            continue
        x, w, bias, beta, alpha, mm_like = extracted

        unwrap_res = _unwrap_weight_to_allgather(w)
        if unwrap_res is None:
            continue
        ag, transforms, chain_nodes = unwrap_res
        ds_id = ag.args[2]

        w_meta = w.meta.get("tensor_meta", None)
        if w_meta is None or not hasattr(w_meta, "shape") or len(w_meta.shape) < 2:
            continue
        K = w_meta.shape[-2]
        if not isinstance(K, int):
            continue

        mm_time_ms = time_map.get(mm_like.name, (0.0, 0.0))[0]
        if mm_time_ms <= 0.0:
            continue

        weight_bytes = ds_id_to_bytes.get(ds_id, 0)
        if weight_bytes <= 0:
            continue

        pref_wait = residual_wait_ms.get(ds_id, 0.0)
        # 如果 prefetch 模拟已经能基本覆盖通信，就不要再 chunk（避免引入额外 GEMM/accumulate）
        if pref_wait <= COMBO_MIN_GAIN_MS:
            continue

        mem_peak = mem_map.get(mm_like.name, (0, 0))[1]
        plan = _choose_chunk_plan_for_mm(
            K=K,
            weight_bytes=weight_bytes,
            mm_time_ms=mm_time_ms,
            prefetch_residual_wait_ms=pref_wait,
            comm_predictor=comm_predictor,
            mem_peak_bytes=mem_peak,
            max_mem_budget=max_mem,
        )
        if plan is None:
            continue
        selected[ds_id] = plan

    if not selected:
        return gm

    rewritten = False
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        extracted = _extract_linear_pattern(node)
        if extracted is None:
            continue
        x, w, bias, beta, alpha, mm_like = extracted

        unwrap_res = _unwrap_weight_to_allgather(w)
        if unwrap_res is None:
            continue
        ag, transforms, chain_nodes = unwrap_res
        ds_id = ag.args[2]
        plan = selected.get(ds_id, None)
        if plan is None:
            continue

        did, _ = _chunk_allgather_and_gemm(graph,
                                          node,
                                          x,
                                          w,
                                          bias,
                                          beta,
                                          alpha,
                                          mm_like,
                                          ag,
                                          transforms,
                                          plan["chunk_size"],
                                          graph_id,
                                          plan["lookahead"],
                                          max_chunks=plan["max_chunks"],
                                          max_packed=plan["max_packed"],
                                          min_chunk_k=plan["min_chunk_k"],
                                          cross_prefetch_anchor=None,
                                          time_map=time_map,
                                          size_map=size_map,
                                          mem_map=mem_map,
                                          max_mem_budget=max_mem,
                                          guard=False)
        if did:
            _prune_unused_call_function_nodes(graph, chain_nodes)
        rewritten = rewritten or did

    if rewritten:
        graph.lint()
        gm.recompile()
        if dist.get_rank() == 0:
            print(f"[chunk_gemm] graph_id={graph_id} AFTER:\n{gm.graph}")

    return gm
