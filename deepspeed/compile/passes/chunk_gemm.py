"""
Chunk linear pass: 对 allgather + gemm 进行按 K 维度分块并流水化。
"""
from typing import List, Tuple, Optional

import torch
from torch.fx import GraphModule, Node

from ..util import is_cast_op
from ..graph_param import DSGraphParamManager
import deepspeed.comm as dist

NAME = "chunk_gemm"

# TODO: 先手动设置chunksize，后面需要加上cost model来自动确定chunk size
DEFAULT_CHUNK_SIZE = 4096

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
            cur = graph.create_node("call_function", torch.ops.aten.permute.default, args=(cur, dims), kwargs={})
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
            cur = graph.create_node("call_function", torch.ops.aten.contiguous.default, args=(cur,), kwargs={})
        else:
            return None
    # 最终 reshape 成与原始权重相同的布局（chunk 维度替换为 chunk_len）
    cur = graph.create_node("call_function", torch.ops.aten.view.default, args=(cur, tuple(target_shape)), kwargs={})
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
        
def _chunk_dim_k(graph, mm_node: Node, x: Node, w: Node, bias: Optional[Node], beta: float, alpha: float,
                 chunk_size: int) -> bool:
    """
    重写一个2D matmul/addmm为k-dimension chunks
    returns true if rewritten, false otherwise
    """
    x_meta = x.meta.get("tensor_meta", None)
    w_meta = w.meta.get("tensor_meta", None)
    if x_meta is None or w_meta is None:
        return False

    if len(x_meta.shape) < 2 or len(w_meta.shape) < 2:
        return False

    K = x_meta.shape[-1]
    K2 = w_meta.shape[-2]
    if K != K2:
        return False
    if K <= chunk_size:
        return False

    x_dim = len(x_meta.shape) - 1
    w_dim = len(w_meta.shape) - 2
    matmul_target = torch.ops.aten.mm.default if len(x_meta.shape) == 2 \
                    and len(w_meta.shape) == 2 else torch.ops.aten.bmm.default

    with mm_node.graph.inserting_before(mm_node):
        acc = None
        start = 0
        while start < K:
            length = min(chunk_size, K - start)

            x_slice = mm_node.graph.create_node("call_function",
                                                torch.ops.aten.slice.Tensor,
                                                args=(x, x_dim, start, start + length),
                                                kwargs={})
            w_slice = mm_node.graph.create_node("call_function",
                                                torch.ops.aten.slice.Tensor,
                                                args=(w, w_dim, start, start + length),
                                                kwargs={})

            y_i = mm_node.graph.create_node(
                                            "call_function", 
                                            matmul_target, 
                                            args=(x_slice, w_slice), 
                                            kwargs={}
                                            )

            if acc is None:
                acc = y_i
            else:
                acc = mm_node.graph.create_node(
                                                "call_function", 
                                                torch.ops.aten.add.Tensor, 
                                                args=(acc, y_i), 
                                                kwargs={}
                                                )

            start += length

        if alpha != 1.0:
            acc = mm_node.graph.create_node("call_function",
                                            torch.ops.aten.mul.Tensor,
                                            args=(acc, alpha),
                                            kwargs={})

        if bias is not None or beta != 0.0:
            if bias is None:
                bias_scaled = mm_node.graph.create_node("call_function",
                                                        torch.ops.aten.mul.Tensor,
                                                        args=(acc, 0.0),
                                                        kwargs={})
            else:
                if beta != 1.0:
                    bias_scaled = mm_node.graph.create_node("call_function",
                                                            torch.ops.aten.mul.Tensor,
                                                            args=(bias, beta),
                                                            kwargs={})
                else:
                    bias_scaled = bias
            acc = mm_node.graph.create_node(
                                            "call_function", 
                                            torch.ops.aten.add.Tensor, 
                                            args=(bias_scaled, acc), 
                                            kwargs={}
                                            )

    _replace_uses(mm_node, acc)
    mm_node.graph.erase_node(mm_node)
    return True


def _chunk_allgather_and_gemm(graph, mm_node: Node, x: Node, w: Node, bias: Optional[Node], beta: float,
                              alpha: float, mm_like: Node, ag_node: Node, transforms, chunk_size: int,
                              graph_id: int) -> bool:
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
    if K is None or K <= chunk_size:
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
    start_k = 0

    def make_kwargs():
        return {"dtype": dtype_kw} if dtype_kw is not None else {}

    with graph.inserting_before(mm_node):
        next_ag = None
        created_nodes = []
        while start_k < K:
            chunk_k = min(chunk_size, K - start_k)
            offset_elems = start_k * suffix
            length_elems = chunk_k * suffix

            if next_ag is None:
                ag_chunk = graph.create_node("call_function",
                                             torch.ops.dc.allgather_param_chunk.default,
                                             args=(ag_input, ag_graph_id, ds_id, offset_elems, length_elems, stride_elems, chunk_count),
                                             kwargs=make_kwargs())
                created_nodes.append(ag_chunk)
            else:
                ag_chunk = next_ag

            next_start = start_k + chunk_k
            if next_start < K:
                next_len = min(chunk_size, K - next_start) * suffix
                next_offset = next_start * suffix
                next_ag = graph.create_node("call_function",
                                            torch.ops.dc.allgather_param_chunk.default,
                                            args=(ag_input, ag_graph_id, ds_id, next_offset, next_len, stride_elems, chunk_count),
                                            kwargs=make_kwargs())
                created_nodes.append(next_ag)
            else:
                next_ag = None

            wait_chunk = graph.create_node("call_function",
                                           torch.ops.dc.wait_allgather_chunk.default,
                                           args=(ag_chunk, ag_graph_id, ds_id, offset_elems, length_elems, stride_elems, chunk_count),
                                           kwargs={})
            created_nodes.append(wait_chunk)

            w_chunk = _apply_transforms_on_chunk(graph, wait_chunk, transforms, w_shape, w_dim, chunk_k,
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
                                        args=(x, x_dim, start_k, start_k + chunk_k),
                                        kwargs={})
            created_nodes.append(x_slice)

            y_i = graph.create_node("call_function", matmul_target, args=(x_slice, w_chunk), kwargs={})
            created_nodes.append(y_i)

            if acc is None:
                acc = y_i
            else:
                acc = graph.create_node("call_function", torch.ops.aten.add.Tensor, args=(acc, y_i), kwargs={})
                created_nodes.append(acc)

            start_k += chunk_k

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
            acc = graph.create_node("call_function", torch.ops.aten.add.Tensor, args=(bias_scaled, acc), kwargs={})

    _replace_uses(mm_node, acc)
    graph.erase_node(mm_node)
    if mm_like is not mm_node and len(mm_like.users) == 0:
        graph.erase_node(mm_like)
    return True

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

        did = _chunk_allgather_and_gemm(graph, node, x, w, bias, beta, alpha, mm_like, ag, transforms,
                                        DEFAULT_CHUNK_SIZE, graph_id)
        if not did:
            did = _chunk_dim_k(graph, node, x, w, bias, beta, alpha, DEFAULT_CHUNK_SIZE)
            if did and mm_like is not node and len(mm_like.users) == 0:
                graph.erase_node(mm_like)
        else:
            # 移除旧的 allgather / wait / view 链路（仅在它们没有其他使用者时）
            _prune_unused_call_function_nodes(graph, chain_nodes)

        rewritten = rewritten or did

    if rewritten:
        graph.lint()
        gm.recompile()
        
        # NOTE: 打印chunk后graph
        # if dist.get_rank() == 0:
        #     print(f"[chunk_gemm] graph_id={graph_id} AFTER:\n{gm.graph}")

    return gm
