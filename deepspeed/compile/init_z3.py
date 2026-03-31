# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import InsertPostInitMethodToModuleSubClasses
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
from deepspeed.utils.torch import register_grad_hook

from .passes import (zero3_compile, prefetch, selective_gather, offload_parameters, global_layer_scheduler,
                     selective_activation_recompute)
from .backend import make_backend, launch_compile_passes, init_schedule, set_z3_param_ds_ids_excluded_from_graph
from .patch_fake_tensor import patch_fake_tensor
from .util import get_deepcompile_handle, add_pre_backward_hook

WARMUP = 5


def init_z3(engine, backend, compile_config, compile_kwargs, schedule=None):

    optimizer = engine.optimizer
    use_opt = not isinstance(optimizer, DeepSpeedZeRoOffload)

    if use_opt and hasattr(optimizer, "ipg_buckets"):
        optimizer.ipg_buckets.clear()
        from deepspeed.runtime.zero.stage3 import IPGBucketZ3
        optimizer.ipg_buckets = {optimizer.communication_data_type: IPGBucketZ3()}
        if optimizer.contiguous_gradients:
            for dtype, bucket in optimizer.ipg_buckets.items():
                bucket.buffer = torch.empty(optimizer.reduce_bucket_size,
                                            dtype=dtype,
                                            device=get_accelerator().current_device_name())
        get_accelerator().empty_cache()

    dc = get_deepcompile_handle()
    dc.init(engine.data_parallel_group, compile_config, engine.zero_reduce_bucket_size())

    # Unset hooks
    for m in engine.module.modules():
        m._parameters = m._original_parameters

    if use_opt:
        optimizer.parameter_offload._remove_module_hooks()

        for hook in optimizer._grad_acc_hooks:
            hook.remove()
        optimizer._grad_acc_hooks.clear()

        for hook in getattr(optimizer, "_leaf_module_hooks", []):
            hook.remove()
        if hasattr(optimizer, "_leaf_module_hooks"):
            optimizer._leaf_module_hooks.clear()

    # Unpatch linear
    if hasattr(InsertPostInitMethodToModuleSubClasses, "linear_bk"):
        torch.nn.functional.linear = InsertPostInitMethodToModuleSubClasses.linear_bk

    if compile_config.symmetric_memory:
        group_name = engine.data_parallel_group.group_name
        dist.enable_symm_mem_for_group(group_name)

    for p in engine.module.parameters():
        if not p.requires_grad:
            continue
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]

        # Disable persistent param
        p.ds_persist = False
        dc.register_z3_param(p.ds_id, p.ds_shape, p.ds_tensor, grad_buffer, p.ds_persist)

    # MoE expert routing is data-dependent, so the set of parameters observed in
    # compiled graphs can differ across ranks and across batches. Handle those
    # parameters consistently outside graph-level ZeRO communication insertion.
    eager_param_candidates = []
    eager_param_ds_ids = set()
    module_by_param_name = {}
    for module_name, submodule in engine.module.named_modules():
        for param_name, _ in submodule.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            module_by_param_name[full_name] = submodule

    has_moe_experts = any(".experts." in name or name.endswith("shared_expert_gate.weight")
                          for name, _ in engine.module.named_parameters())
    if has_moe_experts:
        for module_name, submodule in engine.module.named_modules():
            if ".experts." in module_name and hasattr(submodule, "forward"):
                submodule.forward = torch._dynamo.disable(submodule.forward)

        for name, param in engine.module.named_parameters():
            if not hasattr(param, "ds_id"):
                continue

            owner_module = module_by_param_name.get(name)
            is_embedding_weight = isinstance(owner_module, torch.nn.Embedding) and name.endswith(".weight")
            is_expert_param = ".experts." in name
            is_shared_expert_gate = name.endswith("shared_expert_gate.weight")

            if not (is_embedding_weight or is_expert_param or is_shared_expert_gate):
                continue
            if param.ds_id in eager_param_ds_ids:
                continue

            eager_param_ds_ids.add(param.ds_id)
            eager_param_candidates.append(param)

    engine._deepcompile_eager_param_candidates = tuple(eager_param_candidates)
    engine._deepcompile_eager_param_ds_ids = set(eager_param_ds_ids)
    engine._deepcompile_pending_eager_release_params = ()
    set_z3_param_ds_ids_excluded_from_graph(eager_param_ds_ids)

    # 没有显式传schedule时，设置默认schedule
    if schedule is None:
        schedule = []
        if (compile_config.offload_parameters):
            if getattr(compile_config, "selective_activation_recompute", False):
                raise RuntimeError("selective_activation_recompute is not supported together with offload_parameters")
            schedule.append((0, [zero3_compile.add_z3_gather_release, offload_parameters.offload_parameter_fwd]))
        else:
            if getattr(compile_config, "global_layer_scheduler", False):
                if getattr(compile_config, "selective_activation_recompute", False):
                    raise RuntimeError("selective_activation_recompute + global_layer_scheduler is not implemented yet")
                schedule.append((0, [zero3_compile.add_z3_gather_release, global_layer_scheduler.plan]))
                schedule.append((WARMUP, [zero3_compile.add_z3_gather_release, global_layer_scheduler.apply]))
            elif getattr(compile_config, "selective_activation_recompute", False):
                schedule.append((0, [zero3_compile.add_z3_gather_release, selective_activation_recompute.plan]))
                schedule.append((WARMUP,
                                 [zero3_compile.add_z3_gather_release, selective_activation_recompute.apply]))
            else:
                schedule.append((0, [zero3_compile.add_z3_gather_release]))
                schedule.append((WARMUP, [
                    zero3_compile.add_z3_gather_release, prefetch.schedule_prefetch, selective_gather.selective_gather
                ]))

    global_layer_scheduler.maybe_init_layer_mapping(engine.module, compile_config, schedule)
    selective_activation_recompute.maybe_init_layer_mapping(engine.module, compile_config, schedule)

    init_schedule(schedule)

    if use_opt:

        def set_grad_buffer():
            for i, sub_group in enumerate(optimizer.fp16_groups):
                optimizer.averaged_gradients[i] = [
                    optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[param.ds_id]
                    if param.requires_grad else torch.zeros_like(param.ds_tensor) for param in sub_group
                ]

        add_pre_backward_hook(set_grad_buffer)

        optimizer._deepcompile_eager_grad_pending = False
        optimizer._deepcompile_eager_grad_hook_param_ids = set()

        def register_deepcompile_eager_grad_hook(param):
            if not param.requires_grad:
                return
            if id(param) in optimizer._deepcompile_eager_grad_hook_param_ids:
                return

            def reduce_partition_and_remove_grads(*unused, _param=param):
                optimizer._deepcompile_eager_grad_pending = True
                optimizer.reduce_ready_partitions_and_remove_grads(_param)

            optimizer._grad_acc_hooks.append(register_grad_hook(param, reduce_partition_and_remove_grads))
            optimizer._deepcompile_eager_grad_hook_param_ids.add(id(param))

        def flush_deepcompile_eager_gradients():
            if not optimizer._deepcompile_eager_grad_pending:
                return
            optimizer.overlapping_partition_gradients_reduce_epilogue()
            optimizer._deepcompile_eager_grad_pending = False

        optimizer.register_deepcompile_eager_grad_hook = register_deepcompile_eager_grad_hook
        optimizer.flush_deepcompile_eager_gradients = flush_deepcompile_eager_gradients
        for param in eager_param_candidates:
            register_deepcompile_eager_grad_hook(param)

        # offloading opt states need additional setup
        from .passes.offload_adam_states import move_opt_states, move_opt_states_sync, init_offload_opt_states
        for _, passes in schedule:
            if move_opt_states in passes or move_opt_states_sync in passes:
                init_offload_opt_states(optimizer, dc)

    engine.launch_compile_passes = launch_compile_passes

    patch_fake_tensor()
    torch._inductor.config.size_asserts = False

    return make_backend(backend, compile_config, compile_kwargs=compile_kwargs)
