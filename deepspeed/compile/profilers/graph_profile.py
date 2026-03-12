# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
from typing import Any, Tuple, Dict, Optional
import statistics
import os

import torch
from torch.fx import GraphModule, Interpreter
from torch.fx.node import map_aggregate

try:
    from torch.utils._pytree import tree_all, tree_leaves
    from torch._subclasses.fake_tensor import unset_fake_temporarily, is_fake
except ImportError:
    # Unsupported torch version
    pass

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from ..util import is_comm_op, is_release_node, get_deepcompile_handle


_INPLACE_FUSE_STACK_SUBSTRINGS = ("cross_entropy", "fixed_cross_entropy")
_INPLACE_FUSE_MIN_BYTES = int(os.getenv("DS_DEEPCOMPILE_PROF_INPLACE_FUSE_MIN_BYTES", 256 * 1024 * 1024))
_CE_SPARSE_FUSE_ENABLED = os.getenv("DS_DEEPCOMPILE_PROF_CE_SPARSE_FUSE", "1") not in ("0", "false", "False")
_CE_SPARSE_FUSE_MIN_BYTES = int(os.getenv("DS_DEEPCOMPILE_PROF_CE_SPARSE_FUSE_MIN_BYTES", 256 * 1024 * 1024))
_LARGE_DENSE_GRAD_FUSE_ENABLED = os.getenv("DS_DEEPCOMPILE_PROF_LARGE_DENSE_GRAD_FUSE", "1") not in (
    "0", "false", "False")
_LARGE_DENSE_GRAD_FUSE_MIN_BYTES = int(
    os.getenv("DS_DEEPCOMPILE_PROF_LARGE_DENSE_GRAD_FUSE_MIN_BYTES", 256 * 1024 * 1024))
_LARGE_DENSE_GRAD_STACK_SUBSTRINGS = ("embed_tokens", "word_embeddings", ".wte")


class _CEFullNoAlloc:

    __slots__ = ("shape", "fill_value", "dtype", "device")

    def __init__(self, shape, fill_value, dtype, device):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        self.device = device

    def __repr__(self) -> str:
        return f"_CEFullNoAlloc(shape={self.shape}, fill_value={self.fill_value}, dtype={self.dtype}, device={self.device})"


class _CEScatterValueNoAlloc:

    __slots__ = ("shape", "dim", "index", "value")

    def __init__(self, shape, dim: int, index: torch.Tensor, value):
        self.shape = shape
        self.dim = dim
        self.index = index
        self.value = value

    def __repr__(self) -> str:
        return f"_CEScatterValueNoAlloc(shape={self.shape}, dim={self.dim}, index_shape={tuple(self.index.shape)}, value={self.value})"


class _CESparseScaledNoAlloc:

    __slots__ = ("shape", "dim", "index", "scale", "value")

    def __init__(self, shape, dim: int, index: torch.Tensor, scale: torch.Tensor, value):
        self.shape = shape
        self.dim = dim
        self.index = index
        self.scale = scale
        self.value = value

    def __repr__(self) -> str:
        return (f"_CESparseScaledNoAlloc(shape={self.shape}, dim={self.dim}, index_shape={tuple(self.index.shape)}, "
                f"scale_shape={tuple(self.scale.shape)}, value={self.value})")


class _CEDenseMulNoAlloc:

    __slots__ = ("exp", "sum", "exp_node", "node")

    def __init__(self, exp: torch.Tensor, sum: torch.Tensor, exp_node: Optional[torch.fx.Node], node: torch.fx.Node):
        self.exp = exp
        self.sum = sum
        self.exp_node = exp_node
        self.node = node

    def __repr__(self) -> str:
        exp_shape = tuple(self.exp.shape) if torch.is_tensor(self.exp) else None
        sum_shape = tuple(self.sum.shape) if torch.is_tensor(self.sum) else None
        return f"_CEDenseMulNoAlloc(exp_shape={exp_shape}, sum_shape={sum_shape}, exp_node={getattr(self.exp_node, 'name', None)})"


class _LargeDenseGradFullNoAlloc:

    __slots__ = ("shape", "fill_value", "dtype", "device", "reserve_bytes", "kind")

    def __init__(self, shape, fill_value, dtype, device, reserve_bytes: int, kind: str):
        self.shape = shape
        self.fill_value = fill_value
        self.dtype = dtype
        self.device = device
        self.reserve_bytes = int(reserve_bytes)
        self.kind = str(kind)

    def __repr__(self) -> str:
        return (f"_LargeDenseGradFullNoAlloc(shape={self.shape}, dtype={self.dtype}, device={self.device}, "
                f"reserve_bytes={self.reserve_bytes}, kind={self.kind})")


class _LargeDenseGradNoAlloc:

    __slots__ = ("shape", "dtype", "device", "reserve_bytes", "kind")

    def __init__(self, shape, dtype, device, reserve_bytes: int, kind: str):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.reserve_bytes = int(reserve_bytes)
        self.kind = str(kind)

    def __repr__(self) -> str:
        return (f"_LargeDenseGradNoAlloc(shape={self.shape}, dtype={self.dtype}, device={self.device}, "
                f"reserve_bytes={self.reserve_bytes}, kind={self.kind})")


def _to_int_tuple(v) -> Optional[Tuple[int, ...]]:
    if not isinstance(v, (tuple, list)):
        return None
    dims = []
    sym_int_ty = getattr(torch, "SymInt", None)
    for x in v:
        if sym_int_ty is not None and isinstance(x, sym_int_ty):
            try:
                dims.append(int(x.node.hint))
                continue
            except Exception:
                pass
        try:
            dims.append(int(x))
        except Exception:
            return None
    return tuple(dims)


def _dtype_element_size(dtype) -> int:
    try:
        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        return 0


def _full_default_requested_nbytes(args, kwargs) -> int:
    if not args:
        return 0
    shape = _to_int_tuple(args[0])
    if not shape:
        return 0
    dtype = kwargs.get("dtype", None)
    if not isinstance(dtype, torch.dtype):
        return 0
    es = _dtype_element_size(dtype)
    if es <= 0:
        return 0
    numel = 1
    for d in shape:
        numel *= int(d)
    return int(numel * es)


def _match_ce_sparse_decomp(full_node: torch.fx.Node):
    if full_node.op != "call_function" or full_node.target != torch.ops.aten.full.default:
        return None
    if not _has_any_stack_substring(full_node, _INPLACE_FUSE_STACK_SUBSTRINGS):
        return None
    users = list(full_node.users)
    if len(users) != 1:
        return None

    scatter_node = users[0]
    if scatter_node.op != "call_function" or scatter_node.target != torch.ops.aten.scatter.value:
        return None
    if not scatter_node.args or scatter_node.args[0] is not full_node:
        return None
    if len(scatter_node.args) < 4:
        return None
    try:
        dim = int(scatter_node.args[1])
        value = float(scatter_node.args[3])
    except Exception:
        return None
    if dim != 1 or value != -1.0:
        return None
    scatter_users = list(scatter_node.users)
    if len(scatter_users) != 1:
        return None

    mul_sparse = scatter_users[0]
    if mul_sparse.op != "call_function" or mul_sparse.target != torch.ops.aten.mul.Tensor:
        return None
    if not (mul_sparse.args and (mul_sparse.args[0] is scatter_node or mul_sparse.args[1] is scatter_node)):
        return None

    sum_nodes = [u for u in mul_sparse.users if u.op == "call_function" and u.target == torch.ops.aten.sum.dim_IntList]
    if len(sum_nodes) != 1:
        return None
    sum_node = sum_nodes[0]
    if not sum_node.args or sum_node.args[0] is not mul_sparse:
        return None
    if len(sum_node.args) < 3:
        return None
    try:
        if list(sum_node.args[1]) != [1] or sum_node.args[2] is not True:
            return None
    except Exception:
        return None
    sum_users = list(sum_node.users)
    if len(sum_users) != 1:
        return None

    mul_dense = sum_users[0]
    if mul_dense.op != "call_function" or mul_dense.target != torch.ops.aten.mul.Tensor:
        return None
    if not (mul_dense.args and (mul_dense.args[0] is sum_node or mul_dense.args[1] is sum_node)):
        return None
    mul_dense_users = list(mul_dense.users)
    if len(mul_dense_users) != 1:
        return None

    # Expect sub.Tensor(mul_sparse, mul_dense)
    sub_nodes = [
        u for u in mul_sparse.users
        if u.op == "call_function" and u.target == torch.ops.aten.sub.Tensor and u.args and u.args[0] is mul_sparse
        and u.args[1] is mul_dense
    ]
    if len(sub_nodes) != 1:
        return None
    if mul_dense_users[0] is not sub_nodes[0]:
        return None

    return (full_node, scatter_node, mul_sparse, sum_node, mul_dense, sub_nodes[0])


def _mark_explicit_temp_reserve(node: torch.fx.Node, reserve_bytes: int, kind: str):
    meta = node.meta
    prev = int(meta.get("explicit_temp_reserve_bytes", 0))
    if int(reserve_bytes) > prev:
        meta["explicit_temp_reserve_bytes"] = int(reserve_bytes)
    meta["explicit_temp_reserve_kind"] = str(kind)


def _match_large_dense_grad_decomp(full_node: torch.fx.Node):
    if not _LARGE_DENSE_GRAD_FUSE_ENABLED:
        return None
    if full_node.op != "call_function" or full_node.target != torch.ops.aten.full.default:
        return None
    requested_nbytes = _full_default_requested_nbytes(full_node.args, full_node.kwargs)
    if requested_nbytes < _LARGE_DENSE_GRAD_FUSE_MIN_BYTES:
        return None

    shape = _to_int_tuple(full_node.args[0]) if full_node.args else None
    if not shape or len(shape) != 2:
        return None

    dtype = full_node.kwargs.get("dtype", None)
    if dtype != torch.float32:
        return None

    if not _has_any_stack_substring(full_node, _LARGE_DENSE_GRAD_STACK_SUBSTRINGS):
        return None

    users = list(full_node.users)
    if len(users) != 1:
        return None

    index_put_node = users[0]
    if index_put_node.op != "call_function" or index_put_node.target != torch.ops.aten.index_put.default:
        return None
    if not index_put_node.args or index_put_node.args[0] is not full_node:
        return None

    accumulate = None
    if len(index_put_node.args) >= 4:
        accumulate = index_put_node.args[3]
    elif "accumulate" in index_put_node.kwargs:
        accumulate = index_put_node.kwargs["accumulate"]
    if accumulate not in (True, 1):
        return None

    return (full_node, index_put_node, int(requested_nbytes))


def _build_ce_sparse_fusion_role_map(graph) -> Dict[torch.fx.Node, str]:
    if not _CE_SPARSE_FUSE_ENABLED:
        return {}

    roles: Dict[torch.fx.Node, str] = {}
    for n in graph.nodes:
        m = _match_ce_sparse_decomp(n)
        if not m:
            continue
        full_node, scatter_node, mul_sparse, sum_node, mul_dense, sub_node = m
        # Avoid fusing small cases; the motivation is to prevent huge allocations/OOM.
        if _full_default_requested_nbytes(full_node.args, full_node.kwargs) < _CE_SPARSE_FUSE_MIN_BYTES:
            continue
        roles[full_node] = "ce_full"
        roles[scatter_node] = "ce_scatter"
        roles[mul_sparse] = "ce_mul_sparse"
        roles[sum_node] = "ce_sum"
        roles[mul_dense] = "ce_mul_dense"
        roles[sub_node] = "ce_sub"

    return roles


def _build_large_dense_grad_role_map(graph) -> Dict[torch.fx.Node, str]:
    if not _LARGE_DENSE_GRAD_FUSE_ENABLED:
        return {}

    roles: Dict[torch.fx.Node, str] = {}
    for n in graph.nodes:
        matched = _match_large_dense_grad_decomp(n)
        if not matched:
            continue
        full_node, index_put_node, reserve_bytes = matched
        roles[full_node] = "dense_full"
        roles[index_put_node] = "dense_index_put"
        _mark_explicit_temp_reserve(full_node, reserve_bytes, "embedding_dense_grad_fp32")
        _mark_explicit_temp_reserve(index_put_node, reserve_bytes, "embedding_dense_grad_fp32")

    return roles


def _has_any_stack_substring(n: torch.fx.Node, substrings) -> bool:
    st = getattr(n, "stack_trace", None) or ""
    return any(s in st for s in substrings)


def _tensor_nbytes(v) -> int:
    if not torch.is_tensor(v):
        return 0
    try:
        return int(v.numel() * v.element_size())
    except Exception:
        return 0


def _is_non_view_tensor(v) -> bool:
    # Best-effort aliasing guard: only inplace-update tensors that are not views.
    if not torch.is_tensor(v):
        return False
    return getattr(v, "_base", None) is None


def _should_try_inplace_fuse(interp: Interpreter, n: torch.fx.Node, args) -> bool:
    if not getattr(interp, "garbage_collect_values", False):
        return False
    if n.op != "call_function":
        return False
    if not _has_any_stack_substring(n, _INPLACE_FUSE_STACK_SUBSTRINGS):
        return False
    return _tensor_nbytes(args[0]) >= _INPLACE_FUSE_MIN_BYTES if args else False


def _maybe_inplace_fuse_call(interp: Interpreter, n: torch.fx.Node, args, kwargs):
    """
    Best-effort inplace fusion for the cross-entropy backward decomposition.

    When profiling FX graphs node-by-node, some decomposed primitives allocate multiple
    huge [B*Seq, Vocab] fp32 intermediates (e.g., one-hot/scatter + exp + sub), causing
    OOM even though real execution is fused and fits. Here we opportunistically reuse
    dead buffers (based on FX last-use analysis) to avoid extra allocations.
    """

    if not _should_try_inplace_fuse(interp, n, args):
        return None

    last_uses = getattr(interp, "user_to_last_uses", {}).get(n, [])

    # scatter.value(self, dim, index, value) -> Tensor
    if n.target == torch.ops.aten.scatter.value:
        if len(args) == 4 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses and _is_non_view_tensor(args[0]):
            out = args[0]
            out.scatter_(int(args[1]), args[2], args[3])
            return out

    # mul.Tensor(self, other) -> Tensor  (broadcasted)
    if n.target == torch.ops.aten.mul.Tensor:
        if (len(args) == 2 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses and _is_non_view_tensor(args[0])
                and torch.is_tensor(args[1])):
            out = args[0]
            out.mul_(args[1])
            return out

    # sub.Tensor(self, other) -> Tensor
    if n.target == torch.ops.aten.sub.Tensor:
        if (len(args) == 2 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses and _is_non_view_tensor(args[0])
                and torch.is_tensor(args[1])):
            out = args[0]
            out.sub_(args[1])
            return out

    return None


def _maybe_ce_sparse_fuse_call(interp: Interpreter, n: torch.fx.Node, args, kwargs):
    role_map = getattr(interp, "_ce_sparse_fusion_roles", None)
    if not role_map:
        return None
    role = role_map.get(n)
    if role is None:
        return None

    if role == "ce_full":
        shape = _to_int_tuple(args[0]) if args else None
        if not shape:
            return None
        fill_value = args[1] if len(args) > 1 else None
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)
        if _full_default_requested_nbytes(args, kwargs) < _CE_SPARSE_FUSE_MIN_BYTES:
            return None
        return _CEFullNoAlloc(shape=shape, fill_value=fill_value, dtype=dtype, device=device)

    if role == "ce_scatter":
        if len(args) != 4 or not isinstance(args[0], _CEFullNoAlloc):
            return None
        dim = int(args[1])
        index = args[2]
        value = args[3]
        if dim != 1 or not torch.is_tensor(index):
            return None
        return _CEScatterValueNoAlloc(shape=args[0].shape, dim=dim, index=index, value=value)

    if role == "ce_mul_sparse":
        if len(args) != 2:
            return None
        if isinstance(args[0], _CEScatterValueNoAlloc) and torch.is_tensor(args[1]):
            scatter, scale = args[0], args[1]
        elif isinstance(args[1], _CEScatterValueNoAlloc) and torch.is_tensor(args[0]):
            scatter, scale = args[1], args[0]
        else:
            return None
        return _CESparseScaledNoAlloc(shape=scatter.shape, dim=scatter.dim, index=scatter.index, scale=scale,
                                      value=scatter.value)

    if role == "ce_sum":
        if not args or not isinstance(args[0], _CESparseScaledNoAlloc):
            return None
        sparse = args[0]
        if len(args) < 3:
            return None
        dim_list = args[1]
        keepdim = args[2]
        if list(dim_list) != [sparse.dim]:
            return None
        if keepdim not in (True, False):
            return None
        out = sparse.scale.mul(float(sparse.value))
        if not keepdim and torch.is_tensor(out) and out.dim() == 2 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    if role == "ce_mul_dense":
        if len(args) != 2 or not (torch.is_tensor(args[0]) and torch.is_tensor(args[1])):
            return None

        if _tensor_nbytes(args[0]) >= _CE_SPARSE_FUSE_MIN_BYTES and _tensor_nbytes(args[1]) < _CE_SPARSE_FUSE_MIN_BYTES:
            exp, sum_ = args[0], args[1]
            exp_node = n.args[0] if isinstance(n.args[0], torch.fx.Node) else None
        elif _tensor_nbytes(args[1]) >= _CE_SPARSE_FUSE_MIN_BYTES and _tensor_nbytes(args[0]) < _CE_SPARSE_FUSE_MIN_BYTES:
            exp, sum_ = args[1], args[0]
            exp_node = n.args[1] if isinstance(n.args[1], torch.fx.Node) else None
        else:
            return None

        return _CEDenseMulNoAlloc(exp=exp, sum=sum_, exp_node=exp_node, node=n)

    if role == "ce_sub":
        if len(args) != 2 or not (isinstance(args[0], _CESparseScaledNoAlloc) and isinstance(args[1], _CEDenseMulNoAlloc)):
            return None
        sparse, dense = args[0], args[1]
        if sparse.dim != 1:
            return None
        if not (torch.is_tensor(sparse.index) and torch.is_tensor(dense.exp) and torch.is_tensor(dense.sum)):
            return None

        can_reuse_exp = _is_non_view_tensor(dense.exp) and dense.exp_node is not None and (
            dense.exp_node in getattr(interp, "user_to_last_uses", {}).get(dense.node, []))

        out = dense.exp if can_reuse_exp else dense.exp.clone()
        neg_sum = dense.sum.neg()
        out.mul_(neg_sum)
        out.scatter_add_(sparse.dim, sparse.index, dense.sum)
        return out

    return None


def _maybe_large_dense_grad_noalloc_call(interp: Interpreter, n: torch.fx.Node, args, kwargs):
    role_map = getattr(interp, "_large_dense_grad_roles", None)
    role = role_map.get(n) if role_map else None

    if role == "dense_full":
        shape = _to_int_tuple(args[0]) if args else None
        if not shape:
            return None
        fill_value = args[1] if len(args) > 1 else None
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)
        reserve_bytes = int(getattr(n, "meta", {}).get("explicit_temp_reserve_bytes",
                                                       _full_default_requested_nbytes(args, kwargs)))
        kind = str(getattr(n, "meta", {}).get("explicit_temp_reserve_kind", "large_dense_grad"))
        return _LargeDenseGradFullNoAlloc(shape=shape,
                                          fill_value=fill_value,
                                          dtype=dtype,
                                          device=device,
                                          reserve_bytes=reserve_bytes,
                                          kind=kind)

    if role == "dense_index_put":
        if len(args) < 3 or not isinstance(args[0], _LargeDenseGradFullNoAlloc):
            return None
        full = args[0]
        return _LargeDenseGradNoAlloc(shape=full.shape,
                                      dtype=full.dtype,
                                      device=full.device,
                                      reserve_bytes=full.reserve_bytes,
                                      kind=full.kind)

    if args and isinstance(args[0], _LargeDenseGradNoAlloc):
        dense = args[0]
        if n.target == torch.ops.prims.convert_element_type.default:
            if len(args) < 2 or not isinstance(args[1], torch.dtype):
                return None
            return _LargeDenseGradNoAlloc(shape=dense.shape,
                                          dtype=args[1],
                                          device=dense.device,
                                          reserve_bytes=dense.reserve_bytes,
                                          kind=dense.kind)
        if n.target == torch.ops.aten._to_copy.default:
            dtype = kwargs.get("dtype", None)
            if not isinstance(dtype, torch.dtype):
                return None
            return _LargeDenseGradNoAlloc(shape=dense.shape,
                                          dtype=dtype,
                                          device=dense.device,
                                          reserve_bytes=dense.reserve_bytes,
                                          kind=dense.kind)
        if n.target == torch.ops.dc.reduce_grad.default:
            dtype = dense.dtype if isinstance(dense.dtype, torch.dtype) else torch.float32
            device = dense.device if dense.device is not None else getattr(interp, "device", None)
            with unset_fake_temporarily():
                return torch.empty([0], dtype=dtype, device=device)

    return None


def _can_inplace_fuse_call(interp: Interpreter, n: torch.fx.Node, args, kwargs) -> bool:
    if not _should_try_inplace_fuse(interp, n, args):
        return False
    last_uses = getattr(interp, "user_to_last_uses", {}).get(n, [])
    if n.target == torch.ops.aten.scatter.value:
        return (len(args) == 4 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses
                and _is_non_view_tensor(args[0]))
    if n.target == torch.ops.aten.mul.Tensor:
        return (len(args) == 2 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses
                and _is_non_view_tensor(args[0]) and torch.is_tensor(args[1]))
    if n.target == torch.ops.aten.sub.Tensor:
        return (len(args) == 2 and isinstance(n.args[0], torch.fx.Node) and n.args[0] in last_uses
                and _is_non_view_tensor(args[0]) and torch.is_tensor(args[1]))
    return False


def _maybe_fuse_call(interp: Interpreter, n: torch.fx.Node, args, kwargs):
    ret = _maybe_ce_sparse_fuse_call(interp, n, args, kwargs)
    if ret is not None:
        return ret
    ret = _maybe_large_dense_grad_noalloc_call(interp, n, args, kwargs)
    if ret is not None:
        return ret
    return _maybe_inplace_fuse_call(interp, n, args, kwargs)


def _all_real_if_tensor(args):
    return tree_all(lambda x: not torch.is_tensor(x) or not is_fake(x), args)


def _to(v, device):
    if torch.is_tensor(v):
        with unset_fake_temporarily():
            return v.to(device)
    return v


def _args_to_key(v):

    def _tensor_to_key(v) -> str:
        if torch.is_tensor(v):
            if v.numel() == 1:
                try:
                    return f"{v.dtype}{v.device}{v.item()}"
                except Exception as e:
                    return f"{v.dtype}{v.device}ptr{v.data_ptr()}"
            else:
                return f"{v.dtype}{v.device}{v.shape}"
        return str(v)

    return map_aggregate(v, _tensor_to_key)


def _node_size(out):
    return sum([v.element_size() * v.numel() for v in tree_leaves(out) if torch.is_tensor(v)])


def _get_mem_usage_out_of_torch():

    adjust = 0
    try:
        import pynvml
        pynvml.nvmlInit()

        current_dev_id = get_accelerator().current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(current_dev_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        torch_alloc = get_accelerator().memory_allocated()
        adjust = info.used - torch_alloc
    except:
        # pynvml not available
        pass

    return adjust


# https://pytorch.org/tutorials/intermediate/fx_profiling_tutorial.html
class ProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, iteration: int = 10, warmup: int = 5, debug_log=False):
        super().__init__(gm)

        self.nz3 = get_deepcompile_handle()

        assert iteration > 0
        assert warmup >= 0
        self.iteration = iteration
        self.warmup = warmup
        self.device = torch.device(get_accelerator().current_device())
        self.cache: Dict[Tuple, Any] = {}
        self.distributed = dist.is_initialized()
        self.allgather_mem: Dict[int, int] = {}
        self.debug_log = debug_log
        self.mem_usage_out_of_torch = 0
        self._ce_sparse_fusion_roles = _build_ce_sparse_fusion_role_map(gm.graph)
        self._large_dense_grad_roles = _build_large_dense_grad_role_map(gm.graph)

    def run(self, *args) -> Any:
        """Run the graph with profiling enabled.

        args: inputs to the graph. Tensors in the inpusts must be real tensors, not fake tensors. args can contain ds parameters.
        returns: The output of the graph. Tensor in the output is real tensors.
        """
        return_val = None
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()
                    with torch.no_grad():
                        return_val = super().run(*args)
        except Exception as e:
            msg = e.msg if "msg" in dir(e) else str(e)
            print(f"Profiling error {msg}")
        finally:
            self.nz3.clear_all_gathered_params()
            self.nz3.enable_profiling(False)
        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:

        if n.op in {"placeholder", "output"}:
            n.meta["device_time"] = 0.0
            n.meta["wall_time"] = 0.0
            n.meta["alloc_mem"] = 0
            n.meta["max_mem"] = 0
            n.meta["tensor_size"] = _node_size(n)
            return super().run_node(n)

        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        partitioned_params = {}

        def rebuild_param_if_necessary(v):
            if hasattr(v, "ds_id"):
                v.all_gather(param_list=[v])
                if hasattr(v, "ds_target_dtype"):
                    casted = v.to(v.ds_target_dtype)
                    partitioned_params[id(casted)] = v
                    return casted
            return v

        args = map_aggregate(args, lambda x: rebuild_param_if_necessary(x))

        args = map_aggregate(args, lambda x: _to(x, self.device))
        kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))

        cache_key = (n.target, _args_to_key(args), _args_to_key(kwargs))
        cache_hit = cache_key in self.cache

        cache_hit_flag = torch.tensor([0 if cache_hit else 1], device=self.device, dtype=torch.int)
        if self.distributed:
            dist.all_reduce(cache_hit_flag, dist.ReduceOp.SUM)
        cache_hit = cache_hit_flag.item() == 0

        if cache_hit:
            device_time, wall_time, alloc_mem, max_mem, tensor_size = self.cache[cache_key]
            n.meta["device_time"] = device_time
            n.meta["wall_time"] = wall_time
            n.meta["alloc_mem"] = alloc_mem
            n.meta["max_mem"] = max_mem
            n.meta["tensor_size"] = tensor_size

        is_release_op = is_release_node(n)
        # Inplace fusion mutates buffers; do not run multiple warmup/iterations.
        run_only_once = cache_hit or is_release_op or (self._ce_sparse_fusion_roles.get(n) == "ce_sub"
                                                       or _can_inplace_fuse_call(self, n, args, kwargs))
        iteration = 1 if run_only_once else self.iteration
        accelerator = get_accelerator()
        start_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]
        end_events = [accelerator.Event(enable_timing=True) for _ in range(iteration)]

        get_accelerator().reset_peak_memory_stats()
        alloc_mem_start = get_accelerator().memory_allocated()
        max_mem_start = get_accelerator().max_memory_allocated()

        if not run_only_once:
            for i in range(self.warmup):
                out = _maybe_fuse_call(self, n, args, kwargs)
                if out is None:
                    out = getattr(self, n.op)(n.target, args, kwargs)

        if is_comm_op(n):
            assert self.distributed, f"Distributed environment is not initialized but comm operator {n.name} {n.target} is used."
            dist.barrier()

        start = time.time()
        for i in range(iteration):
            start_events[i].record()
            out = _maybe_fuse_call(self, n, args, kwargs)
            if out is None:
                out = getattr(self, n.op)(n.target, args, kwargs)
            end_events[i].record()
        accelerator.synchronize()
        walltime_sum = time.time() - start

        if is_comm_op(n):
            dist.barrier()

        alloc_mem = get_accelerator().memory_allocated() - alloc_mem_start + self.mem_usage_out_of_torch
        max_memory = get_accelerator().max_memory_allocated() - max_mem_start + self.mem_usage_out_of_torch
        tensor_size = _node_size(out)

        def partition_param_if_necessary(v):
            if id(v) in partitioned_params:
                v = partitioned_params[id(v)]
            if hasattr(v, "ds_id") and not v.ds_persist:
                v.partition(param_list=[v], has_been_updated=False)
            return v

        args = map_aggregate(args, lambda x: partition_param_if_necessary(x))

        if not cache_hit:
            device_time = statistics.mean([s.elapsed_time(e) for s, e in zip(start_events, end_events)])
            wall_time = walltime_sum / iteration * 1000

            with unset_fake_temporarily():
                vals_to_bcast = torch.tensor([device_time, wall_time, alloc_mem, max_memory, tensor_size],
                                             device=self.device)
                if self.distributed:
                    dist.all_reduce(vals_to_bcast, dist.ReduceOp.AVG)
                n.meta["device_time"] = vals_to_bcast[0].item()
                n.meta["wall_time"] = vals_to_bcast[1].item()
                n.meta["alloc_mem"] = int(vals_to_bcast[2].item())
                n.meta["max_mem"] = int(vals_to_bcast[3].item())
                n.meta["tensor_size"] = int(vals_to_bcast[4].item())
                self.cache[cache_key] = (n.meta["device_time"], n.meta["wall_time"], n.meta["alloc_mem"],
                                         n.meta["max_mem"], n.meta["tensor_size"])

            if is_release_op:
                n.meta["alloc_mem"] = -self.allgather_mem.get(args[2], 0)

            if dist.get_rank() == 0 and self.debug_log:
                print(
                    f"{n.target} {n.meta['device_time']:.2f}ms {n.meta['wall_time']:.2f}ms alloc_mem={n.meta['alloc_mem'] / 1024 / 1024:.2f}MB max_mem={n.meta['max_mem'] / 1024 / 1024:.2f}MB tensor_size={n.meta['tensor_size']}"
                )

        if n.target == torch.ops.dc.allgather_param.default:
            out = args[0]
            assert hasattr(out, "ds_id")
            if not out.ds_persist:
                self.nz3.invalidate_gathered_param(args[2])
            if "dtype" in n.kwargs:
                setattr(out, "ds_target_dtype", n.kwargs["dtype"])
            self.allgather_mem[out.ds_id] = n.meta["alloc_mem"]

        return out


class MemoryProfilingInterpreter(Interpreter):

    def __init__(self, gm: GraphModule, debug_log=False):
        super().__init__(gm)
        self.nz3 = get_deepcompile_handle()
        self.device = torch.device(get_accelerator().current_device())
        self.mem_record = []
        self.last_alloc = get_accelerator().memory_allocated()
        self._ce_sparse_fusion_roles = _build_ce_sparse_fusion_role_map(gm.graph)
        self._large_dense_grad_roles = _build_large_dense_grad_role_map(gm.graph)

        self.node_counter = 0
        self.node_num = len(gm.graph.nodes)
        self.debug_log = debug_log

    def run(self, *args) -> Any:
        return_val = None
        try:
            assert _all_real_if_tensor(args), "Inputs must be real tensors"
            self.nz3.enable_profiling(True)
            self.mem_usage_out_of_torch = _get_mem_usage_out_of_torch()

            with unset_fake_temporarily():
                with get_accelerator().random().fork_rng(devices=[self.device]):
                    return_val = super().run(*args)
        except Exception as e:
            print(f"MemoryProfiling error {e}")
        finally:
            self.nz3.enable_profiling(False)

        return return_val

    def run_node(self, n: torch.fx.Node) -> Any:
        get_accelerator().reset_peak_memory_stats()

        if n.op in {"placeholder", "output"}:
            ret = super().run_node(n)
        else:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            args = map_aggregate(args, lambda x: _to(x, self.device))
            kwargs = map_aggregate(kwargs, lambda x: _to(x, self.device))
            ret = _maybe_fuse_call(self, n, args, kwargs)
            if ret is None:
                ret = getattr(self, n.op)(n.target, args, kwargs)

            del args, kwargs

        current_alloc = get_accelerator().memory_allocated() + self.mem_usage_out_of_torch
        max_alloc = get_accelerator().max_memory_allocated() + self.mem_usage_out_of_torch
        vals_to_bcast = torch.tensor([current_alloc, max_alloc], device=self.device, dtype=torch.int64)
        dist.all_reduce(vals_to_bcast, dist.ReduceOp.MAX)
        current_alloc = vals_to_bcast[0].item()
        max_alloc = vals_to_bcast[1].item()

        self.mem_record.append((n.name, current_alloc, current_alloc - self.last_alloc, max_alloc))

        self.node_counter += 1
        if self.debug_log and dist.get_rank() == 0:
            print(
                f"Mem prof Node {self.node_counter}/{self.node_num} {n.name} memory {current_alloc / 1024 / 1024:.2f}MB delta {(current_alloc - self.last_alloc) / 1024 / 1024:.2f}MB"
            )

        self.last_alloc = current_alloc

        return ret

    def dump(self, path):
        import pandas as pd
        df = pd.DataFrame(self.mem_record, columns=["node", "memory", "delta", "max_mem"])
        df.to_csv(path, index=False)
