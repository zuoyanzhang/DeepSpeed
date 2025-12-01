"""
chunk only linear pass
allgather先不进行chunk,目前先支持仅对计算图中的linear进行chunk操作
"""
from typing import List, Tuple, Optional

import torch
from torch.fx import GraphModule, Node

from ..util import is_cast_op
from ..graph_param import DSGraphParamManager

NAME = "chunk_gemm"

# NOTE: 将chunk的大小设置成1024，后面需要加上cost model来自动确定chunk size
DEFAULT_CHUNK_SIZE = 1024

def _unwrap_weight_to_allgather(node: Node) -> Optional[Node]:
    # 尝试从当前节点开始向上查找allgather节点
    # 如果找到allgather节点，则返回该节点。否则返回None
    # dc.allgather_param -> dc.wait_allgather -> cast(fp16) -> 作为某个算子的输入
    cur = node
    for _ in range(3):
        if cur.op == "call_function" and cur.target == torch.ops.dc.wait_allgather.default:
            cur = cur.args[0]
            continue
        if cur.op == "call_function":
            is_cast, _ = is_cast_op(cur)
            if is_cast:
                cur = cur.args[0]
                continue
        break

    if cur.op == "call_function" and cur.target == torch.ops.dc.allgather_param.default:
        return cur
    return None

def _extract_linear_pattern(node: Node):
    """
    Try to unify different linear-ish patterns to (x, w, bias, beta, alpha).
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
            if first.op == "call_function" and first.target in {
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
        x, w, inner_bias, beta, alpha = inner
        bias = inner_bias if inner_bias is not None else bias_candidate
        return x, w, bias, beta, alpha

    if node.target == torch.ops.aten.mm.default and len(node.args) == 2:
        x, w = node.args
        return x, w, bias, beta, alpha

    if node.target == torch.ops.aten.addmm.default and len(node.args) >= 3:
        bias, x, w = node.args[:3]
        if len(node.args) >= 4:
            beta = float(node.args[3])
        if len(node.args) >= 5:
            alpha = float(node.args[4])
        return x, w, bias, beta, alpha

    if node.target == torch.ops.aten.bmm.default and len(node.args) == 2:
        x, w = node.args
        return x, w, bias, beta, alpha

    if node.target == torch.ops.aten.matmul.default and len(node.args) == 2:
        x, w = node.args
        x_meta = x.meta.get("tensor_meta", None)
        w_meta = w.meta.get("tensor_meta", None)
        if x_meta is None or w_meta is None:
            return None
        if len(x_meta.shape) in (2, 3) and len(w_meta.shape) in (2, 3):
            return x, w, bias, beta, alpha
        return None

    return None

def _replace_uses(old: Node, new: Node):
    # 替换old节点的所有使用者，将其输入从old替换为new
    for user in list(old.users):
        user.replace_input_with(old, new)
        
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
    matmul_target = torch.ops.aten.mm.default if len(x_meta.shape) == 2 and len(w_meta.shape) == 2 else torch.ops.aten.bmm.default

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

            y_i = mm_node.graph.create_node("call_function", matmul_target, args=(x_slice, w_slice), kwargs={})

            if acc is None:
                acc = y_i
            else:
                acc = mm_node.graph.create_node("call_function", torch.ops.aten.add.Tensor, args=(acc, y_i), kwargs={})

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
            acc = mm_node.graph.create_node("call_function", torch.ops.aten.add.Tensor, args=(bias_scaled, acc), kwargs={})

    _replace_uses(mm_node, acc)
    mm_node.graph.erase_node(mm_node)
    return True

def chunk_gemm(gm: GraphModule, graph_id: int, graph_order: List[Tuple[int, bool]], profiling_results,
               create_inputs_fn, mem_budget: float, param_manager: DSGraphParamManager, bwd: bool) -> GraphModule:
    # 先只对前向图做chunk
    if bwd:
        return gm

    graph = gm.graph
    rewritten = False

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue

        extracted = _extract_linear_pattern(node)
        if extracted is None:
            continue
        x, w, bias, beta, alpha = extracted

        ag = _unwrap_weight_to_allgather(w)
        if ag is None:
            continue

        did = _chunk_dim_k(graph, node, x, w, bias, beta, alpha, DEFAULT_CHUNK_SIZE)
        rewritten = rewritten or did

    if rewritten:
        graph.lint()
        gm.recompile()

    return gm
