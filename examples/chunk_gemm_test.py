# Minimal demo for the chunk_gemm pass (Step 1: chunk compute only).
#
# This does not run DeepSpeed end-to-end; it builds a tiny FX graph that
# includes a dc.allgather_param -> aten.mm pattern, runs the chunk pass,
# and prints the graph before/after so you can see the rewrite.

import torch
from torch.fx import Graph, GraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.library import Library

from deepspeed.compile.passes import chunk_gemm


def ensure_dc_ops():
    """
    If DeepCompile custom ops are not available, try to build/load them.
    As a last resort, register lightweight CPU fallbacks so the FX graph
    can be built和打印（不会做真实通信）。
    """
    if hasattr(torch.ops, "dc") and hasattr(torch.ops.dc, "allgather_param"):
        return

    # 尝试编译/加载 DeepCompile 自定义算子
    try:
        from deepspeed.ops.op_builder.dc import DeepCompileBuilder

        DeepCompileBuilder().load()
        if hasattr(torch.ops, "dc") and hasattr(torch.ops.dc, "allgather_param"):
            return
    except Exception as e:
        print(f"[chunk_linear_demo] Failed to load dc ops, fallback to stub ops: {e}")

    # 最后兜底：注册 CPU stub，便于构图/打印
    lib = Library("dc", "DEF")
    lib.define("allgather_param(Tensor a, int graph_id, int ds_id, ScalarType? dtype=None) -> Tensor")
    lib.define(
        "allgather_param_chunk(Tensor a, int graph_id, int ds_id, int offset, int length, int stride=0, int chunk_count=1, ScalarType? dtype=None) -> Tensor"
    )
    lib.define("wait_allgather(Tensor a, int graph_id, int ds_id) -> Tensor")
    lib.define(
        "wait_allgather_chunk(Tensor a, int graph_id, int ds_id, int offset, int length, int stride=0, int chunk_count=1) -> Tensor"
    )

    lib_impl = Library("dc", "IMPL", "CPU")
    lib_impl.impl("allgather_param", lambda a, graph_id, ds_id, dtype=None: a)
    def _fake_allgather_chunk(a, graph_id, ds_id, offset, length, stride=0, chunk_count=1, dtype=None):
        if chunk_count <= 0:
            raise ValueError("chunk_count must be positive")
        if stride < 0:
            raise ValueError("stride must be non-negative")
        if chunk_count > 1 and stride <= 0:
            raise ValueError("stride must be positive when chunk_count > 1")

        flat = a.flatten()
        if dtype is not None:
            flat = flat.to(dtype)

        stride_elems = stride if chunk_count > 1 else (stride if stride > 0 else length)
        pieces = []
        for idx in range(chunk_count):
            start = offset + idx * stride_elems
            if start >= flat.numel():
                break
            take = min(length, flat.numel() - start)
            pieces.append(flat.narrow(0, start, take))
        if not pieces:
            return torch.empty((0,), device=flat.device, dtype=flat.dtype)
        return torch.cat(pieces)

    lib_impl.impl("allgather_param_chunk", _fake_allgather_chunk)
    lib_impl.impl("wait_allgather", lambda a, graph_id, ds_id: a)
    lib_impl.impl("wait_allgather_chunk", lambda a, graph_id, ds_id, offset, length, stride=0, chunk_count=1: a)


def build_dummy_graph():
    ensure_dc_ops()
    g = Graph()

    # Inputs with tensor_meta so the pass knows shapes.
    x = g.placeholder("x")
    w = g.placeholder("w")

    x.meta["tensor_meta"] = TensorMetadata(
        shape=torch.Size([4, 2048]),
        dtype=torch.float32,
        requires_grad=False,
        stride=(2048, 1),
        memory_format=torch.contiguous_format,
        is_quantized=False,
        qparams=None,
    )
    w.meta["tensor_meta"] = TensorMetadata(
        shape=torch.Size([2048, 4096]),
        dtype=torch.float32,
        requires_grad=False,
        stride=(4096, 1),
        memory_format=torch.contiguous_format,
        is_quantized=False,
        qparams=None,
    )

    # Simulate ZeRO-3 allgather; this node is never executed in the demo.
    ag = g.call_function(torch.ops.dc.allgather_param.default, args=(w, 0, 0), kwargs={"dtype": torch.float32})
    # 给 allgather 节点也填上 tensor_meta，便于 chunk pass 获取形状
    ag.meta["tensor_meta"] = w.meta["tensor_meta"]

    mm = g.call_function(torch.ops.aten.mm.default, args=(x, ag))
    g.output(mm)

    return GraphModule(torch.nn.Module(), g)


def main():
    gm = build_dummy_graph()
    print("=== Original graph ===")
    print(gm.graph)

    graph_id = id(gm.graph)
    graph_order = [(graph_id, False)]
    gm = chunk_gemm.chunk_gemm(gm,
                               graph_id=graph_id,
                               graph_order=graph_order,
                               profiling_results={},
                               create_inputs_fn=lambda: (),
                               mem_budget=0.0,
                               param_manager=None,
                               bwd=False)

    print("\n=== After chunk_gemm ===")
    print(gm.graph)


if __name__ == "__main__":
    main()
