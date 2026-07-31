# Modified by Chorus contributors for Chorus backends and portable data/model loading.

import os
import argparse
import time
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, enable_full_determinism
from datasets import load_dataset, DownloadConfig
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler

from datasets.utils.logging import disable_progress_bar


def distributed_max_int(value: int) -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(value)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([int(value)], device=device, dtype=torch.int64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset_name", "--dataset-name", type=str,
                        default="timdettmers/openassistant-guanaco",
                        help="Hugging Face dataset used when --dataset-path is not supplied and the bundled dataset is unavailable.")
    parser.add_argument("--dataset_path", "--dataset-path", type=str, default=None,
                        help="Local JSON/JSONL dataset file or directory. Defaults to the dataset bundled beside this script.")
    parser.add_argument("--num_layers", type=int, default=0)
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--passes", type=str, default=None)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument("--distributed_backend", "--distributed-backend", type=str, default="auto")
    parser.add_argument("--simplefsdp_reshard_after_forward", "--simplefsdp-reshard-after-forward",
                        action="store_true",
                        help="Compatibility alias for --simplefsdp_reshard_after_forward_policy always.")
    parser.add_argument("--simplefsdp_reshard_after_forward_policy", "--simplefsdp-reshard-after-forward-policy",
                        choices=("default", "always", "never"), default="default",
                        help="TorchTitan-style SimpleFSDP reshard policy. Default follows TorchTitan smart defaults.")
    parser.add_argument("--simplefsdp_reshard_after_backward_policy", "--simplefsdp-reshard-after-backward-policy",
                        choices=("default", "always", "never"), default="default",
                        help="Native FSDP2 backward reshard policy. 'never' keeps unsharded params after backward to trade memory for fewer next-forward all-gathers.")
    parser.add_argument("--simplefsdp_layer_group_size", "--simplefsdp-layer-group-size",
                        type=int, default=1,
                        help="Group this many consecutive decoder layers into one native fully_shard unit to trade memory/overlap for fewer collectives.")
    parser.add_argument("--simplefsdp_disable_inductor_comm_overlap", "--simplefsdp-disable-inductor-comm-overlap",
                        action="store_true",
                        help="Disable TorchInductor compute/communication reordering for SimpleFSDP. Default enables it as the native compiler-side scheduling path.")
    parser.add_argument("--simplefsdp_enable_compiled_autograd", "--simplefsdp-enable-compiled-autograd",
                        action="store_true",
                        help="Enable TorchDynamo compiled autograd for SimpleFSDP so backward communication can be optimized by the compiler.")
    parser.add_argument("--simplefsdp_comm_overlap_policy", "--simplefsdp-comm-overlap-policy",
                        choices=("nvlink", "aggressive", "balanced", "conservative", "off"),
                        default="nvlink",
                        help="SimpleFSDP Inductor comm-overlap schedule. nvlink uses a lighter raise/sink pass order; aggressive also reorders compute for slower or more communication-bound fabrics.")
    parser.add_argument("--simplefsdp_comm_overlap_passes", "--simplefsdp-comm-overlap-passes",
                        type=str, default=None,
                        help="Comma-separated TorchInductor comm-overlap scheduler passes for SimpleFSDP ablations. Overrides --simplefsdp_comm_overlap_policy.")
    parser.add_argument("--simplefsdp_coalesce_bucket_mb", "--simplefsdp-coalesce-bucket-mb",
                        type=int, default=128,
                        help="Native SimpleFSDP c10d coalescing bucket size in MiB. 0 disables local graph bucketing.")
    parser.add_argument("--simplefsdp_enable_chorus", "--simplefsdp-enable-chorus",
                        action="store_true",
                        help="Enable Chorus-style graph scheduling for SimpleFSDP all-gather collectives.")
    parser.add_argument("--simplefsdp_chorus_prefetch_groups", "--simplefsdp-chorus-prefetch-groups",
                        type=int, default=-1,
                        help="Number of SimpleFSDP all-gather buckets Chorus may raise before earlier consumer buckets. -1 auto-selects a workload-aware window; non-negative values are manual ablations.")
    parser.add_argument("--simplefsdp_chorus_live_mb", "--simplefsdp-chorus-live-mb",
                        type=int, default=-1,
                        help="Approximate extra live all-gather budget for SimpleFSDP-Chorus graph scheduling in MiB. -1 uses the automatic safety budget; 0 disables graph-retained all-gather buffers; positive values are manual ablations.")
    parser.add_argument("--simplefsdp_chorus_global_retention_mb", "--simplefsdp-chorus-global-retention-mb",
                        type=int, default=-1,
                        help="Runtime-footprint budget for SimpleFSDP-Chorus global retention in MiB. -1 auto-selects from actual GPU memory with a safety margin; 0 disables persistent retention; positive values are manual ablations.")
    parser.add_argument("--simplefsdp_chorus_persistent_usable_fraction", "--simplefsdp-chorus-persistent-usable-fraction",
                        type=float, default=0.90,
                        help="Safety margin for automatic SimpleFSDP-Chorus persistent retention. The auto budget targets at most this fraction of total GPU memory.")
    parser.add_argument("--simplefsdp_chorus_persistent_baseline_param_multiplier", "--simplefsdp-chorus-persistent-baseline-param-multiplier",
                        type=float, default=1.45,
                        help="Conservative multiplier from loaded fp32 model parameter bytes to estimated non-persistent memory peak for automatic SimpleFSDP-Chorus budgeting.")
    parser.add_argument("--simplefsdp_chorus_persistent_cost_multiplier", "--simplefsdp-chorus-persistent-cost-multiplier",
                        type=float, default=20.00,
                        help="Runtime footprint multiplier for each extra persistent parameter byte, covering grads, optimizer state, foreach temporaries, and allocator slack.")
    parser.add_argument("--simplefsdp_chorus_persistent_static_margin_mb", "--simplefsdp-chorus-persistent-static-margin-mb",
                        type=int, default=0,
                        help="Additional fixed safety margin in MiB for automatic SimpleFSDP-Chorus persistent retention budgeting.")
    parser.add_argument("--simplefsdp_chorus_global_retention_max_layers", "--simplefsdp-chorus-global-retention-max-layers",
                        type=int, default=0,
                        help="Optional max number of transformer layers selected for SimpleFSDP-Chorus global retention. 0 means budget-only.")
    parser.add_argument("--simplefsdp_chorus_milp_time_limit_s", "--simplefsdp-chorus-milp-time-limit-s",
                        type=float, default=2.0,
                        help="Time limit for the SimpleFSDP-Chorus global-retention MILP planner.")
    parser.add_argument("--simplefsdp_chorus_enable_cross_graph_retention", "--simplefsdp-chorus-enable-cross-graph-retention",
                        action="store_true",
                        help="Compatibility no-op: SimpleFSDP-Chorus now enables cross-graph retention by default.")
    parser.add_argument("--simplefsdp_chorus_disable_cross_graph_retention", "--simplefsdp-chorus-disable-cross-graph-retention",
                        action="store_true",
                        help="Disable SimpleFSDP-Chorus cross-graph retention for debugging ablations.")
    parser.add_argument("--simplefsdp_replicate_small_param_numel", "--simplefsdp-replicate-small-param-numel",
                        type=int, default=16384,
                        help="Replicate SimpleFSDP parameters with at most this many elements to remove tiny all-gathers. 0 disables this memory-for-latency policy.")
    parser.add_argument("--simplefsdp_enable_explicit_prefetch", "--simplefsdp-enable-explicit-prefetch",
                        action="store_true",
                        help="Enable non-native explicit FSDP2 prefetch distance tuning for SimpleFSDP ablations.")
    parser.add_argument("--simplefsdp_keep_gradient_division", "--simplefsdp-keep-gradient-division",
                        action="store_true",
                        help="Keep FSDP automatic gradient division. Default disables it like TorchTitan.")
    parser.add_argument("--simplefsdp_activation_checkpointing", "--simplefsdp-activation-checkpointing",
                        action="store_true",
                        help="Keep HF full activation checkpointing enabled for SimpleFSDP. This is the default when --activation_checkpointing is set.")
    parser.add_argument("--simplefsdp_checkpoint_every_n_layers", "--simplefsdp-checkpoint-every-n-layers",
                        type=int, default=0,
                        help="Optional SimpleFSDP selective activation checkpoint interval. 0 keeps the normal full activation-checkpointing policy.")
    parser.add_argument("--simplefsdp_enable_fused_optimizer", "--simplefsdp-enable-fused-optimizer",
                        action="store_true",
                        help="Force fused AdamW for SimpleFSDP when available. Fused AdamW is the default CUDA fast path unless disabled.")
    parser.add_argument("--simplefsdp_disable_fused_optimizer", "--simplefsdp-disable-fused-optimizer",
                        action="store_true",
                        help="Disable the SimpleFSDP fused AdamW fast path for ablations.")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile_dir", type=str, default=None)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Compatibility option: local root directory containing --model_name.")
    parser.add_argument("--model_dir", "--model-dir", type=str, default=None,
                        help="Exact local model directory. When omitted, --model_name is also used as a Hugging Face identifier.")
    parser.add_argument("--disable_fsdp2_prefetch", "--disable-fsdp2-prefetch", action="store_true",
                        help="Disable explicit FSDP2 module prefetching.")
    parser.add_argument("--fsdp2_forward_prefetch_distance", "--fsdp2-forward-prefetch-distance",
                        type=int, default=0,
                        help="Number of following FSDP2 layers to prefetch in forward.")
    parser.add_argument("--fsdp2_backward_prefetch_distance", "--fsdp2-backward-prefetch-distance",
                        type=int, default=1,
                        help="Number of preceding FSDP2 layers to prefetch in backward.")
    parser.add_argument("--fsdp2_enable_chorus", "--fsdp2-enable-chorus", action="store_true",
                        help="Enable Chorus-style nonuniform FSDP2 prefetch lists using per-layer parameter-byte budgets.")
    parser.add_argument("--fsdp2_chorus_live_mb", "--fsdp2-chorus-live-mb", type=int, default=4096,
                        help="Approximate extra live full-parameter budget for FSDP2-Chorus prefetching in MiB. 0 disables the cap.")

    return parser.parse_args()



def _is_accelerate_fsdp2(accelerator: Accelerator) -> bool:
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    try:
        return int(getattr(fsdp_plugin, "fsdp_version", 1) or 1) == 2
    except (TypeError, ValueError):
        return False


def _fsdp_base_class_name(module: torch.nn.Module) -> str:
    class_name = module.__class__.__name__
    if class_name.startswith("FSDP"):
        class_name = class_name[len("FSDP"):]
    return class_name


def _is_transformer_layer_class(module: torch.nn.Module) -> bool:
    wrapped = getattr(module, "_checkpoint_wrapped_module", None)
    if wrapped is not None:
        return _is_transformer_layer_class(wrapped)
    class_name = _fsdp_base_class_name(module)
    return class_name.endswith("DecoderLayer") or class_name == "BaichuanLayer"


def _replace_child_module(parent: torch.nn.Module, name: str, new_child: torch.nn.Module) -> None:
    if isinstance(parent, torch.nn.ModuleList):
        parent[int(name)] = new_child
    else:
        setattr(parent, name, new_child)


def _collect_transformer_layer_children(model: torch.nn.Module):
    entries = []
    for parent in model.modules():
        if getattr(parent, "_checkpoint_wrapped_module", None) is not None:
            continue
        for name, child in parent.named_children():
            if _is_transformer_layer_class(child):
                entries.append((parent, name, child))
    return entries


def apply_simplefsdp_selective_checkpointing(model: torch.nn.Module, every_n_layers: int) -> dict:
    every_n_layers = int(every_n_layers)
    if every_n_layers <= 0:
        return {"enabled": False, "layers": 0, "every_n_layers": 0}

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    entries = _collect_transformer_layer_children(model)

    wrapped_layers = 0
    for idx, (parent, name, child) in enumerate(entries):
        if (idx + 1) % every_n_layers != 0:
            continue
        wrapped = checkpoint_wrapper(
            child,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            preserve_rng_state=False,
        )
        _replace_child_module(parent, name, wrapped)
        wrapped_layers += 1

    return {
        "enabled": wrapped_layers > 0,
        "layers": wrapped_layers,
        "every_n_layers": every_n_layers,
    }


def _is_fsdp2_transformer_layer(module: torch.nn.Module) -> bool:
    if not (hasattr(module, "set_modules_to_forward_prefetch") and
            hasattr(module, "set_modules_to_backward_prefetch")):
        return False
    return _is_transformer_layer_class(module)


def configure_fsdp2_explicit_prefetch(
    model: torch.nn.Module,
    forward_distance: int,
    backward_distance: int,
) -> dict:
    fsdp_layers = [module for module in model.modules() if _is_fsdp2_transformer_layer(module)]
    if not fsdp_layers:
        return {
            "enabled": False,
            "layers": 0,
            "forward_distance": 0,
            "backward_distance": 0,
            "chorus_enabled": False,
            "chorus_live_mb": 0,
        }

    forward_distance = max(0, int(forward_distance))
    backward_distance = max(0, int(backward_distance))
    for idx, layer in enumerate(fsdp_layers):
        if forward_distance > 0:
            layer.set_modules_to_forward_prefetch(fsdp_layers[idx + 1:idx + 1 + forward_distance])
        if backward_distance > 0:
            start = max(0, idx - backward_distance)
            layer.set_modules_to_backward_prefetch(list(reversed(fsdp_layers[start:idx])))
    return {
        "enabled": True,
        "layers": len(fsdp_layers),
        "forward_distance": forward_distance,
        "backward_distance": backward_distance,
        "chorus_enabled": False,
        "chorus_live_mb": 0,
    }


def _module_param_nbytes(module: torch.nn.Module) -> int:
    nbytes = 0
    for param in module.parameters(recurse=True):
        try:
            nbytes += int(param.numel()) * int(param.element_size())
        except Exception:
            continue
    return int(nbytes)


def _budgeted_prefetch_list(layers, sizes, start_idx: int, step: int, max_distance: int, budget_bytes: int):
    selected = []
    live_bytes = 0
    idx = int(start_idx)
    for _ in range(max(0, int(max_distance))):
        if idx < 0 or idx >= len(layers):
            break
        size = max(1, int(sizes[idx]))
        if budget_bytes > 0 and selected and live_bytes + size > budget_bytes:
            break
        selected.append(layers[idx])
        live_bytes += size
        idx += int(step)
    return selected


def configure_fsdp2_chorus_prefetch(
    model: torch.nn.Module,
    forward_max_distance: int,
    backward_max_distance: int,
    live_budget_mb: int,
) -> dict:
    fsdp_layers = [module for module in model.modules() if _is_fsdp2_transformer_layer(module)]
    if not fsdp_layers:
        return {
            "enabled": False,
            "layers": 0,
            "forward_distance": 0,
            "backward_distance": 0,
            "chorus_enabled": False,
            "chorus_live_mb": 0,
        }

    forward_max_distance = max(0, int(forward_max_distance))
    backward_max_distance = max(0, int(backward_max_distance))
    budget_bytes = max(0, int(live_budget_mb)) * 1024 * 1024
    layer_sizes = [_module_param_nbytes(layer) for layer in fsdp_layers]

    total_forward_prefetches = 0
    total_backward_prefetches = 0
    for idx, layer in enumerate(fsdp_layers):
        forward_modules = _budgeted_prefetch_list(
            fsdp_layers, layer_sizes, idx + 1, 1, forward_max_distance, budget_bytes
        )
        backward_modules = _budgeted_prefetch_list(
            fsdp_layers, layer_sizes, idx - 1, -1, backward_max_distance, budget_bytes
        )
        if forward_modules:
            layer.set_modules_to_forward_prefetch(forward_modules)
        if backward_modules:
            layer.set_modules_to_backward_prefetch(backward_modules)
        total_forward_prefetches += len(forward_modules)
        total_backward_prefetches += len(backward_modules)

    return {
        "enabled": True,
        "layers": len(fsdp_layers),
        "forward_distance": forward_max_distance,
        "backward_distance": backward_max_distance,
        "chorus_enabled": True,
        "chorus_live_mb": int(live_budget_mb),
        "chorus_forward_prefetches": int(total_forward_prefetches),
        "chorus_backward_prefetches": int(total_backward_prefetches),
    }


def make_adamw_optimizer(params, lr: float, fused: bool = False):
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr=lr, fused=True), "fused"
        except TypeError:
            pass
    return torch.optim.AdamW(params, lr=lr), "default"



def _module_tree_param_nbytes(module: torch.nn.Module) -> int:
    seen = set()
    nbytes = 0
    for param in module.parameters(recurse=True):
        if param is None:
            continue
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        try:
            nbytes += int(param.numel()) * int(param.element_size())
        except Exception:
            continue
    return int(nbytes)


def _module_tree_param_numel(module: torch.nn.Module) -> int:
    seen = set()
    numel = 0
    for param in module.parameters(recurse=True):
        if param is None:
            continue
        ident = id(param)
        if ident in seen:
            continue
        seen.add(ident)
        try:
            numel += int(param.numel())
        except Exception:
            continue
    return int(numel)


def _mark_module_tree(module: torch.nn.Module, attr: str, value: bool) -> None:
    for submodule in module.modules():
        setattr(submodule, attr, bool(value))


def estimate_simplefsdp_chorus_auto_persistent_budget(
    model: torch.nn.Module,
    usable_fraction: float,
    baseline_param_multiplier: float,
    runtime_cost_multiplier: float,
    static_margin_mb: int,
    world_size: int = 1,
) -> dict:
    total_mem = 0
    current_alloc = 0
    current_reserved = 0
    if torch.cuda.is_available():
        try:
            total_mem = int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory)
            current_alloc = int(torch.cuda.memory_allocated())
            current_reserved = int(torch.cuda.memory_reserved())
        except Exception:
            total_mem = 0
    usable_fraction = max(0.0, min(1.0, float(usable_fraction)))
    baseline_param_multiplier = max(0.0, float(baseline_param_multiplier))
    runtime_cost_multiplier = max(1.0, float(runtime_cost_multiplier))
    static_margin_bytes = max(0, int(static_margin_mb)) * 1024 * 1024
    model_param_bytes = _module_tree_param_nbytes(model)
    model_param_numel = _module_tree_param_numel(model)
    safe_total = int(float(total_mem) * usable_fraction)
    world_size = max(1, int(world_size))
    if world_size <= 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_size = max(1, int(torch.distributed.get_world_size()))
        except Exception:
            world_size = 1
    # SimpleFSDP converts loaded fp32 parameters into sharded bf16 DTensors.
    # A full fp32-parameter multiplier is too pessimistic for the compiled
    # SimpleFSDP steady state and can collapse the Chorus graph budget to zero.
    target_param_bytes = int(model_param_numel) * 2
    local_target_param_bytes = int((target_param_bytes + world_size - 1) // world_size)
    optimizer_grad_floor = int(local_target_param_bytes * 4)
    activation_margin_bytes = max(
        2 * 1024 * 1024 * 1024,
        min(6 * 1024 * 1024 * 1024, int(target_param_bytes // 3)),
    )
    estimated_simplefsdp_runtime_floor = (
        int(optimizer_grad_floor)
        + int(target_param_bytes)
        + int(activation_margin_bytes)
        + int(static_margin_bytes)
    )
    observed_floor = max(int(current_alloc), int(current_reserved))
    estimated_baseline = max(int(estimated_simplefsdp_runtime_floor), int(observed_floor) + int(static_margin_bytes))
    runtime_headroom = max(0, int(safe_total) - int(estimated_baseline))
    if total_mem > 0:
        # Cross-graph retained AGs are transient per-step buffers guarded by the
        # runtime 0.9 safety check. Use a GPU-size-proportional cap and round up
        # to a coarse bucket so 40 GB cards get a 4 GiB window while 80 GB cards
        # can naturally use a larger one.
        live_quantum = 512 * 1024 * 1024
        raw_live_cap = max(1024 * 1024 * 1024, int(float(total_mem) * 0.10))
        rounded_live_cap = ((raw_live_cap + live_quantum - 1) // live_quantum) * live_quantum
        runtime_live_cap = min(16 * 1024 * 1024 * 1024, int(rounded_live_cap))
    else:
        runtime_live_cap = runtime_headroom
    runtime_budget = max(0, min(int(runtime_headroom), int(runtime_live_cap)))
    return {
        "enabled": True,
        "total_mem_bytes": int(total_mem),
        "usable_fraction": float(usable_fraction),
        "safe_total_bytes": int(safe_total),
        "model_param_bytes": int(model_param_bytes),
        "model_param_numel": int(model_param_numel),
        "target_param_bytes": int(target_param_bytes),
        "local_target_param_bytes": int(local_target_param_bytes),
        "baseline_param_multiplier": float(baseline_param_multiplier),
        "runtime_cost_multiplier": float(runtime_cost_multiplier),
        "static_margin_bytes": int(static_margin_bytes),
        "activation_margin_bytes": int(activation_margin_bytes),
        "current_alloc_bytes": int(current_alloc),
        "current_reserved_bytes": int(current_reserved),
        "observed_floor_bytes": int(observed_floor),
        "estimated_simplefsdp_runtime_floor_bytes": int(estimated_simplefsdp_runtime_floor),
        "estimated_baseline_bytes": int(estimated_baseline),
        "runtime_headroom_bytes": int(runtime_headroom),
        "runtime_live_cap_bytes": int(runtime_live_cap),
        "runtime_budget_bytes": int(runtime_budget),
        "runtime_budget_mb": int(runtime_budget // (1024 * 1024)),
    }


def _collect_layer_param_retention_entries(layer_entries, replicate_small_param_numel: int):
    entries = []
    seen = set()
    min_numel = max(0, int(replicate_small_param_numel))
    for layer_idx, (_, _, child) in enumerate(layer_entries):
        for module_path, module in child.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param is None:
                    continue
                ident = id(param)
                if ident in seen:
                    continue
                seen.add(ident)
                try:
                    numel = int(param.numel())
                    nbytes = int(numel * param.element_size())
                except Exception:
                    continue
                if numel <= min_numel or nbytes <= 0:
                    continue
                # Persistent retention is only useful for real matrix/vector
                # parameters. Tiny tensors are already covered by the small-param
                # replication policy and should not consume MILP budget.
                if getattr(param, "ndim", 0) < 1:
                    continue
                qualified = f"{layer_idx}:{module_path}.{param_name}" if module_path else f"{layer_idx}:{param_name}"
                entries.append({
                    "layer_idx": int(layer_idx),
                    "module": module,
                    "param_name": param_name,
                    "qualified": qualified,
                    "nbytes": int(nbytes),
                    "numel": int(numel),
                })
    return entries


def plan_simplefsdp_chorus_global_retention(
    layer_entries,
    world_size: int,
    retention_budget_mb: int,
    max_layers: int,
    activation_checkpointing: bool,
    milp_time_limit_s: float,
    replicate_small_param_numel: int = 0,
    runtime_cost_multiplier: float = 1.0,
) -> dict:
    import math

    param_entries = _collect_layer_param_retention_entries(layer_entries, replicate_small_param_numel)
    num_items = len(param_entries)
    budget_bytes = max(0, int(retention_budget_mb)) * 1024 * 1024
    world_size = max(1, int(world_size))
    if num_items <= 0 or budget_bytes <= 0 or world_size <= 1:
        return {
            "enabled": False,
            "method": "none",
            "selected_layers": [],
            "selected_layers_csv": "",
            "selected_params": [],
            "selected_param_names_csv": "",
            "selected_count": 0,
            "selected_param_count": 0,
            "budget_mb": int(retention_budget_mb),
            "extra_bytes": 0,
            "runtime_extra_bytes": 0,
            "global_bytes": 0,
            "binary_vars": 0,
            "constraints": 0,
            "solve_time_s": 0.0,
            "final_gap": 0.0,
            "status": -1,
            "status_msg": "disabled",
        }

    sizes = [int(entry["nbytes"]) for entry in param_entries]
    extra = [max(0, int(math.ceil(float(size) * (1.0 - 1.0 / float(world_size))))) for size in sizes]
    runtime_cost_multiplier = max(1.0, float(runtime_cost_multiplier))
    # Python-level persistent full-param caching has a fixed per-parameter cost
    # (cache lookup, version check, tensor lifetime, allocator pressure) in
    # addition to bytes. Accounting for that fixed cost keeps Chorus from using
    # the whole memory budget on many small attention matrices and instead
    # favors fewer high-byte all-gathers such as MLP projections.
    per_param_runtime_overhead = 64 * 1024 * 1024 if activation_checkpointing else 32 * 1024 * 1024
    runtime_costs = [
        max(0, int(math.ceil(float(cost) * runtime_cost_multiplier)) + int(per_param_runtime_overhead))
        for cost in extra
    ]
    reuse_factor = 3.0 if activation_checkpointing else 2.0
    max_layer_idx = max(1, max(int(entry["layer_idx"]) for entry in param_entries))
    values = []
    for entry, size in zip(param_entries, sizes):
        layer_bias = 1.0 + 0.03 * float(entry["layer_idx"]) / float(max_layer_idx)
        name = str(entry["qualified"])
        role_bias = 1.0
        if "down_proj" in name or "gate_proj" in name or "up_proj" in name:
            role_bias = 1.35
        elif "W_pack" in name:
            role_bias = 1.15
        elif "q_proj" in name or "o_proj" in name:
            role_bias = 1.02
        elif "k_proj" in name or "v_proj" in name:
            role_bias = 0.90
        values.append(float(size) * reuse_factor * layer_bias * role_bias)
    # Topology-aware admission: persistent retention is only useful when saved
    # communication exceeds the added live-state/runtime pressure. This prevents
    # NVLink runs from filling all spare memory with persistent full params when
    # the communication is already cheap. Lower runtime_cost_multiplier values on
    # slower fabrics naturally admit more parameters.
    min_value_per_runtime_cost = 0.35 if activation_checkpointing else 0.25
    net_values = [
        float(value) - min_value_per_runtime_cost * float(cost)
        for value, cost in zip(values, runtime_costs)
    ]

    selected = []
    method = "greedy"
    status = -1
    status_msg = "greedy_fallback"
    final_gap = 0.0
    solve_time_s = 0.0
    constraints = 1 + (1 if int(max_layers) > 0 else 0)
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        rows = []
        cols = []
        data = []
        lb = []
        ub = []
        row = 0
        for idx, cost in enumerate(runtime_costs):
            rows.append(row)
            cols.append(idx)
            data.append(float(cost))
        lb.append(0.0)
        ub.append(float(budget_bytes))
        row += 1
        if int(max_layers) > 0:
            for idx in range(num_items):
                rows.append(row)
                cols.append(idx)
                data.append(1.0)
            lb.append(0.0)
            ub.append(float(max(0, int(max_layers))))
            row += 1
        A = coo_matrix((np.array(data, dtype=float), (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                       shape=(row, num_items)).tocsr()
        c = -np.array(net_values, dtype=float)
        import time as _time
        start = _time.perf_counter()
        res = milp(
            c,
            integrality=np.ones(num_items, dtype=int),
            bounds=Bounds(np.zeros(num_items, dtype=float), np.ones(num_items, dtype=float)),
            constraints=LinearConstraint(A, np.array(lb, dtype=float), np.array(ub, dtype=float)),
            options={"time_limit": float(milp_time_limit_s), "mip_rel_gap": 0.001, "presolve": True},
        )
        solve_time_s = float(_time.perf_counter() - start)
        if res.x is not None and int(getattr(res, "status", -1)) in (0, 1):
            selected = [idx for idx, value in enumerate(res.x) if float(value) >= 0.5 and net_values[idx] > 0.0]
            method = "milp_param"
            status = int(getattr(res, "status", -1))
            status_msg = str(getattr(res, "message", ""))
            raw_gap = getattr(res, "mip_gap", 0.0)
            final_gap = float(raw_gap) if raw_gap is not None else 0.0
    except Exception as exc:
        status_msg = f"greedy_fallback: {exc}"

    if not selected:
        order = sorted(range(num_items), key=lambda idx: (net_values[idx] / max(1, runtime_costs[idx]), net_values[idx]), reverse=True)
        used = 0
        for idx in order:
            if runtime_costs[idx] <= 0 or net_values[idx] <= 0.0:
                continue
            if int(max_layers) > 0 and len(selected) >= int(max_layers):
                break
            if used + runtime_costs[idx] > budget_bytes:
                continue
            selected.append(idx)
            used += int(runtime_costs[idx])

    selected = sorted(selected)
    used_extra = int(sum(extra[idx] for idx in selected))
    used_runtime_extra = int(sum(runtime_costs[idx] for idx in selected))
    selected_global = int(sum(sizes[idx] for idx in selected))
    selected_layers = sorted({int(param_entries[idx]["layer_idx"]) for idx in selected})
    selected_refs = [(param_entries[idx]["module"], str(param_entries[idx]["param_name"])) for idx in selected]
    selected_names = [str(param_entries[idx]["qualified"]) for idx in selected]
    return {
        "enabled": bool(selected),
        "method": method,
        "selected_layers": selected_layers,
        "selected_layers_csv": ",".join(str(idx) for idx in selected_layers),
        "selected_params": selected_refs,
        "selected_param_names_csv": ",".join(selected_names[:32]),
        "selected_count": int(len(selected_layers)),
        "selected_param_count": int(len(selected)),
        "budget_mb": int(retention_budget_mb),
        "extra_bytes": used_extra,
        "runtime_extra_bytes": used_runtime_extra,
        "global_bytes": selected_global,
        "binary_vars": int(num_items),
        "constraints": int(constraints),
        "solve_time_s": float(solve_time_s),
        "final_gap": float(final_gap),
        "status": int(status),
        "status_msg": status_msg,
    }


def configure_native_simplefsdp(
    model: torch.nn.Module,
    accelerator: Accelerator,
    replicate_small_param_numel: int = 0,
    enable_chorus: bool = False,
    chorus_global_retention_mb: int = -1,
    chorus_global_retention_max_layers: int = 0,
    chorus_milp_time_limit_s: float = 2.0,
    activation_checkpointing: bool = False,
    chorus_persistent_usable_fraction: float = 0.90,
    chorus_persistent_baseline_param_multiplier: float = 1.45,
    chorus_persistent_cost_multiplier: float = 20.00,
    chorus_persistent_static_margin_mb: int = 0,
) -> dict:
    from torch.distributed.device_mesh import init_device_mesh
    from native_simplefsdp import (
        SimpleFSDPMixedPrecisionPolicy,
        data_parallel,
        summarize_simplefsdp_parameters,
    )

    if accelerator.num_processes > 1 and (
        not torch.distributed.is_available() or not torch.distributed.is_initialized()
    ):
        raise RuntimeError("Native SimpleFSDP requires an initialized torch.distributed process group")

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(accelerator.local_process_index)
    mesh = init_device_mesh(device_type, (accelerator.num_processes,), mesh_dim_names=("fsdp",))
    mp_policy = SimpleFSDPMixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    layer_entries = _collect_transformer_layer_children(model)
    auto_budget_meta = {
        "enabled": False,
        "total_mem_bytes": 0,
        "usable_fraction": float(chorus_persistent_usable_fraction),
        "safe_total_bytes": 0,
        "model_param_bytes": 0,
        "model_param_numel": 0,
        "target_param_bytes": 0,
        "local_target_param_bytes": 0,
        "baseline_param_multiplier": float(chorus_persistent_baseline_param_multiplier),
        "runtime_cost_multiplier": float(chorus_persistent_cost_multiplier),
        "static_margin_bytes": int(chorus_persistent_static_margin_mb) * 1024 * 1024,
        "activation_margin_bytes": 0,
        "current_alloc_bytes": 0,
        "current_reserved_bytes": 0,
        "observed_floor_bytes": 0,
        "estimated_simplefsdp_runtime_floor_bytes": 0,
        "estimated_baseline_bytes": 0,
        "runtime_headroom_bytes": 0,
        "runtime_live_cap_bytes": 0,
        "runtime_budget_bytes": 0,
        "runtime_budget_mb": 0,
    }
    requested_retention_mb = int(chorus_global_retention_mb) if enable_chorus else 0
    effective_retention_mb = requested_retention_mb
    if enable_chorus and requested_retention_mb < 0:
        auto_budget_meta = estimate_simplefsdp_chorus_auto_persistent_budget(
            model,
            usable_fraction=float(chorus_persistent_usable_fraction),
            baseline_param_multiplier=float(chorus_persistent_baseline_param_multiplier),
            runtime_cost_multiplier=float(chorus_persistent_cost_multiplier),
            static_margin_mb=int(chorus_persistent_static_margin_mb),
            world_size=accelerator.num_processes,
        )
        # Automatic budgeting is used for graph-local Chorus prefetch/retention.
        # Persistent full-parameter retention changes SimpleFSDP parameter
        # placement, so keep it opt-in via an explicit positive value.
        effective_retention_mb = 0
    chorus_plan = plan_simplefsdp_chorus_global_retention(
        layer_entries,
        world_size=accelerator.num_processes,
        retention_budget_mb=max(0, int(effective_retention_mb)) if enable_chorus else 0,
        max_layers=int(chorus_global_retention_max_layers),
        activation_checkpointing=bool(activation_checkpointing),
        milp_time_limit_s=float(chorus_milp_time_limit_s),
        replicate_small_param_numel=int(replicate_small_param_numel),
        runtime_cost_multiplier=float(chorus_persistent_cost_multiplier),
    )
    for module, param_name in chorus_plan.get("selected_params", []):
        retained = set(getattr(module, "_simplefsdp_chorus_persistent_params", set()))
        retained.add(str(param_name))
        setattr(module, "_simplefsdp_chorus_persistent_params", retained)
    data_parallel(
        model,
        mesh,
        mode="fully_shard",
        mp_policy=mp_policy,
        shard_dim=0,
        full_dtensor=False,
        replicate_numel_threshold=int(replicate_small_param_numel),
        replicate_module_attr="_simplefsdp_chorus_global_retain",
        replicate_param_names_attr="_simplefsdp_chorus_global_retain_params",
        persistent_param_names_attr="_simplefsdp_chorus_persistent_params",
        enable_param_tags=bool(enable_chorus),
    )
    param_stats = summarize_simplefsdp_parameters(model)
    return {
        "enabled": True,
        "recipe": "native_dtensor_parametrization",
        "layers": len(layer_entries),
        "layer_group_size": 1,
        "layer_groups": len(layer_entries),
        "sharded_frontend_modules": int(param_stats["wrapped_modules"]),
        "enable_weight_tying": False,
        "reshard_after_forward_policy": "dtensor_parametrization",
        "reshard_after_forward": False,
        "reshard_after_backward_policy": "dtensor_parametrization",
        "reshard_after_backward": False,
        "reshard_after_backward_modules": 0,
        "gradient_division_disabled_modules": 0,
        "explicit_prefetch_ablation": False,
        "prefetch_enabled": False,
        "forward_distance": 0,
        "backward_distance": 0,
        "replicate_small_param_numel": int(replicate_small_param_numel),
        "replicated_params": int(param_stats["replicated_params"]),
        "replicated_global_numel": int(param_stats["replicated_global_numel"]),
        "sharded_params": int(param_stats["sharded_params"]),
        "dtensor_params": int(param_stats["dtensor_params"]),
        "dtensor_local_numel": int(param_stats["local_numel"]),
        "dtensor_global_numel": int(param_stats["global_numel"]),
        "chorus_global_retention_enabled": bool(chorus_plan.get("enabled", False)),
        "chorus_global_retention_method": str(chorus_plan.get("method", "none")),
        "chorus_global_retention_layers": int(chorus_plan.get("selected_count", 0)),
        "chorus_global_retention_layer_ids": str(chorus_plan.get("selected_layers_csv", "")),
        "chorus_global_retention_params": int(chorus_plan.get("selected_param_count", 0)),
        "chorus_global_retention_param_names": str(chorus_plan.get("selected_param_names_csv", "")),
        "chorus_global_retention_budget_mode": "auto" if int(chorus_global_retention_mb) < 0 and enable_chorus else ("manual" if int(chorus_global_retention_mb) > 0 and enable_chorus else "disabled"),
        "chorus_global_retention_requested_mb": int(chorus_global_retention_mb) if enable_chorus else 0,
        "chorus_global_retention_budget_mb": int(chorus_plan.get("budget_mb", 0)),
        "chorus_global_retention_extra_bytes": int(chorus_plan.get("extra_bytes", 0)),
        "chorus_global_retention_runtime_extra_bytes": int(chorus_plan.get("runtime_extra_bytes", 0)),
        "chorus_global_retention_global_bytes": int(chorus_plan.get("global_bytes", 0)),
        "chorus_auto_budget_total_mem_bytes": int(auto_budget_meta.get("total_mem_bytes", 0)),
        "chorus_auto_budget_usable_fraction": float(auto_budget_meta.get("usable_fraction", 0.0)),
        "chorus_auto_budget_safe_total_bytes": int(auto_budget_meta.get("safe_total_bytes", 0)),
        "chorus_auto_budget_model_param_bytes": int(auto_budget_meta.get("model_param_bytes", 0)),
        "chorus_auto_budget_model_param_numel": int(auto_budget_meta.get("model_param_numel", 0)),
        "chorus_auto_budget_target_param_bytes": int(auto_budget_meta.get("target_param_bytes", 0)),
        "chorus_auto_budget_local_target_param_bytes": int(auto_budget_meta.get("local_target_param_bytes", 0)),
        "chorus_auto_budget_activation_margin_bytes": int(auto_budget_meta.get("activation_margin_bytes", 0)),
        "chorus_auto_budget_observed_floor_bytes": int(auto_budget_meta.get("observed_floor_bytes", 0)),
        "chorus_auto_budget_runtime_floor_bytes": int(auto_budget_meta.get("estimated_simplefsdp_runtime_floor_bytes", 0)),
        "chorus_auto_budget_estimated_baseline_bytes": int(auto_budget_meta.get("estimated_baseline_bytes", 0)),
        "chorus_auto_budget_runtime_headroom_bytes": int(auto_budget_meta.get("runtime_headroom_bytes", 0)),
        "chorus_auto_budget_runtime_live_cap_bytes": int(auto_budget_meta.get("runtime_live_cap_bytes", 0)),
        "chorus_auto_budget_runtime_budget_bytes": int(auto_budget_meta.get("runtime_budget_bytes", 0)),
        "chorus_auto_budget_runtime_cost_multiplier": float(auto_budget_meta.get("runtime_cost_multiplier", chorus_persistent_cost_multiplier)),
        "chorus_milp_binary_vars": int(chorus_plan.get("binary_vars", 0)),
        "chorus_milp_constraints": int(chorus_plan.get("constraints", 0)),
        "chorus_milp_solve_time_s": float(chorus_plan.get("solve_time_s", 0.0)),
        "chorus_milp_final_gap": float(chorus_plan.get("final_gap", 0.0)),
        "chorus_milp_status": int(chorus_plan.get("status", -1)),
        "chorus_milp_status_msg": str(chorus_plan.get("status_msg", "")),
    }


def make_schedule(passes: List[str], warmup):
    from deepspeed.compile.passes import (zero3_compile, prefetch, selective_gather, offload_adam_states,
                                          global_layer_scheduler, selective_activation_recompute)

    schedule = []

    if "offload_adam_states" in passes:
        assert len(passes) == 1, "offload_adam_states should be the only pass"
        schedule.append((0, [offload_adam_states.offload_adam_states_for_init, zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states_sync]))
        schedule.append((5, [offload_adam_states.offload_adam_states_for_init, zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states]))
    elif "offload_adam_states_sync" in passes:
        assert len(passes) == 1, "offload_adam_states_sync should be the only pass"
        schedule.append((0, [zero3_compile.add_z3_gather_release, offload_adam_states.move_opt_states_sync]))
    elif "selective_activation_recompute" in passes:
        assert len(passes) == 1, "selective_activation_recompute should be the only pass in the MVP"
        schedule.append((0, [zero3_compile.add_z3_gather_release, selective_activation_recompute.plan]))
        schedule.append((warmup, [zero3_compile.add_z3_gather_release, selective_activation_recompute.apply]))
    else:
        if "global_layer_scheduler" in passes:
            assert "prefetch" not in passes and "selective_gather" not in passes, \
                "global_layer_scheduler should not be combined with prefetch/selective_gather in the same schedule"
            schedule.append((0, [zero3_compile.add_z3_gather_release, global_layer_scheduler.plan]))
            schedule.append((warmup, [zero3_compile.add_z3_gather_release, global_layer_scheduler.apply]))
        else:
            schedule.append((0, [zero3_compile.add_z3_gather_release]))
            second_opt = [zero3_compile.add_z3_gather_release]
            if "prefetch" in passes:
                second_opt.append(prefetch.schedule_prefetch)
            if "selective_gather" in passes:
                second_opt.append(selective_gather.selective_gather)
            schedule.append((warmup, second_opt))
    return schedule


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # to suppress tokenizer parallelism warning
    args = get_args()
    # When --num_layers is set, build a smaller randomly initialized model from
    # the local config instead of loading pretrained weights.
    if args.num_layers > 0:
        args.load_weights = False
    print(args)

    is_simplefsdp_arg = args.distributed_backend == "simplefsdp"

    activation_checkpointing_enabled = bool(args.activation_checkpointing)
    simplefsdp_selective_checkpoint_every = 0
    if (
        is_simplefsdp_arg
        and args.activation_checkpointing
        and not args.simplefsdp_activation_checkpointing
        and int(args.simplefsdp_checkpoint_every_n_layers) > 0
    ):
        activation_checkpointing_enabled = False
        simplefsdp_selective_checkpoint_every = int(args.simplefsdp_checkpoint_every_n_layers)

    if args.passes is not None and "offload_adam_states" in args.passes:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    if args.deterministic:
        enable_full_determinism(1)
        from torch._inductor import config
        config.fallback_random = True

    simplefsdp_overlap_policy_passes = {
        "nvlink": ["raise_comms", "sink_waits"],
        "aggressive": ["reorder_compute_for_overlap", "sink_waits", "raise_comms"],
        "balanced": ["reorder_compute_for_overlap", "sink_waits"],
        "conservative": ["sink_waits"],
    }
    simplefsdp_inductor_comm_overlap = bool(
        is_simplefsdp_arg
        and args.compile
        and not args.simplefsdp_disable_inductor_comm_overlap
        and args.simplefsdp_comm_overlap_policy != "off"
    )
    if simplefsdp_inductor_comm_overlap:
        from torch._inductor import config
        config.reorder_for_compute_comm_overlap = True
        if args.simplefsdp_comm_overlap_passes:
            config.reorder_for_compute_comm_overlap_passes = [
                pass_name.strip()
                for pass_name in args.simplefsdp_comm_overlap_passes.split(",")
                if pass_name.strip()
            ]
        else:
            config.reorder_for_compute_comm_overlap_passes = list(
                simplefsdp_overlap_policy_passes[args.simplefsdp_comm_overlap_policy]
            )
        if args.simplefsdp_enable_chorus:
            # Installed after SimpleFSDP wrapping, once the automatic Chorus memory
            # budget has been computed from the actual model and GPU.
            pass
        elif int(args.simplefsdp_coalesce_bucket_mb) > 0:
            from native_simplefsdp import simplefsdp_coalesce_collectives_graph_pass

            bucket_bytes = int(args.simplefsdp_coalesce_bucket_mb) * 1024 * 1024
            config.post_grad_custom_pre_pass = (
                lambda graph: simplefsdp_coalesce_collectives_graph_pass(graph, bucket_bytes)
            )
    simplefsdp_compiled_autograd = bool(
        is_simplefsdp_arg and args.compile and args.simplefsdp_enable_compiled_autograd
    )
    if simplefsdp_compiled_autograd:
        from torch._dynamo import config as dynamo_config
        dynamo_config.compiled_autograd = True

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    is_deepspeed = accelerator.state.deepspeed_plugin is not None
    is_simplefsdp = is_simplefsdp_arg
    print(f"Running on device: {device} is_deepspeed: {is_deepspeed} distributed_backend: {args.distributed_backend}")

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("Loading model and tokenizer...")

    model_name = args.model_name

    if args.model_dir:
        model_weight_path = os.path.abspath(os.path.expanduser(args.model_dir))
        model_source = model_weight_path
    elif args.model_path:
        model_weight_path = os.path.join(os.path.expanduser(args.model_path), args.model_name)
        model_source = model_weight_path if os.path.exists(model_weight_path) else model_name
    else:
        model_weight_path = model_name
        model_source = model_name
    
    if accelerator.is_main_process:
        print(f"model_source: {model_source}")
    if args.load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_source,
                                                     trust_remote_code=True)
    else:
        model_config = AutoConfig.from_pretrained(model_source,
                                                  attn_implementation=args.attn_impl,
                                                  trust_remote_code=True)
        if args.num_layers > 0:
            print(f"num_hidden_layers: {model_config.num_hidden_layers} -> {args.num_layers}")
            model_config.num_hidden_layers = args.num_layers
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
            
    # 有些buffer类型是float32，使用fsdp+compile的时候需要强制将buffer提前转换成bfloat16，
    # 否则torch.compile的dynamo会触发类型转换错误
    # model = model.to(dtype=torch.bfloat16)
    # for name, buffer in model.named_buffers():
    #     if buffer.dtype == torch.float32:
    #         buffer.data = buffer.data.to(torch.bfloat16)
            
    if accelerator.is_main_process:
        print(f"model is {model}")

    tokenizer = AutoTokenizer.from_pretrained(model_source,
                                              trust_remote_code=True)

    if args.save_weights and accelerator.is_main_process:
        model.save_pretrained(model_weight_path)

    if activation_checkpointing_enabled:
        model.gradient_checkpointing_enable()

    simplefsdp_selective_checkpoint_stats = {"enabled": False, "layers": 0, "every_n_layers": 0}
    if is_simplefsdp and simplefsdp_selective_checkpoint_every > 0:
        simplefsdp_selective_checkpoint_stats = apply_simplefsdp_selective_checkpointing(
            model,
            simplefsdp_selective_checkpoint_every,
        )

    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")
    else:
        disable_progress_bar()
        
    bundled_dataset = Path(__file__).resolve().parent / "datasets" / "openassistant_best_replies_de_train.jsonl"
    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else bundled_dataset
    if dataset_path.exists():
        if dataset_path.is_dir():
            data_files = sorted(str(path) for pattern in ("*.json", "*.jsonl") for path in dataset_path.glob(pattern))
            if not data_files:
                raise FileNotFoundError(f"No JSON or JSONL files found in dataset directory: {dataset_path}")
        else:
            data_files = [str(dataset_path)]
        dataset = load_dataset(
            "json",
            data_files={"train": data_files},
            split="train[:100%]",
            download_config=DownloadConfig(disable_tqdm=True),
        )
    elif args.dataset_path:
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    else:
        dataset = load_dataset(
            args.dataset_name,
            split="train[:100%]",
            download_config=DownloadConfig(disable_tqdm=True),
        )

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=args.seq_length, truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    sampler = DistributedSampler(tokenized_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=8,
        persistent_workers=True,
    )

    simplefsdp_stats = {
        "enabled": False,
        "recipe": "none",
        "layers": 0,
        "layer_group_size": 1,
        "layer_groups": 0,
        "sharded_frontend_modules": 0,
        "enable_weight_tying": False,
        "reshard_after_forward_policy": "default",
        "reshard_after_forward": False,
        "reshard_after_backward_policy": "default",
        "reshard_after_backward": True,
        "reshard_after_backward_modules": 0,
        "gradient_division_disabled_modules": 0,
        "explicit_prefetch_ablation": False,
        "prefetch_enabled": False,
        "forward_distance": 0,
        "backward_distance": 0,
        "activation_checkpointing": False,
        "full_activation_checkpointing": False,
        "selective_checkpoint_layers": 0,
        "selective_checkpoint_every_n_layers": 0,
        "optimizer": "default",
        "inductor_comm_overlap": False,
        "compiled_autograd": False,
        "comm_overlap_policy": "off",
        "comm_overlap_passes": "",
        "chorus_enabled": False,
        "chorus_prefetch_groups": 0,
        "chorus_live_mb": 0,
        "chorus_global_retention_enabled": False,
        "chorus_global_retention_method": "none",
        "chorus_global_retention_layers": 0,
        "chorus_global_retention_layer_ids": "",
        "chorus_global_retention_params": 0,
        "chorus_global_retention_param_names": "",
        "chorus_global_retention_budget_mode": "disabled",
        "chorus_global_retention_requested_mb": 0,
        "chorus_global_retention_budget_mb": 0,
        "chorus_global_retention_extra_bytes": 0,
        "chorus_global_retention_runtime_extra_bytes": 0,
        "chorus_global_retention_global_bytes": 0,
        "chorus_auto_budget_total_mem_bytes": 0,
        "chorus_auto_budget_usable_fraction": 0.0,
        "chorus_auto_budget_safe_total_bytes": 0,
        "chorus_auto_budget_model_param_bytes": 0,
        "chorus_auto_budget_model_param_numel": 0,
        "chorus_auto_budget_target_param_bytes": 0,
        "chorus_auto_budget_local_target_param_bytes": 0,
        "chorus_auto_budget_activation_margin_bytes": 0,
        "chorus_auto_budget_observed_floor_bytes": 0,
        "chorus_auto_budget_runtime_floor_bytes": 0,
        "chorus_auto_budget_estimated_baseline_bytes": 0,
        "chorus_auto_budget_runtime_headroom_bytes": 0,
        "chorus_auto_budget_runtime_live_cap_bytes": 0,
        "chorus_auto_budget_runtime_budget_bytes": 0,
        "chorus_auto_budget_runtime_cost_multiplier": 0.0,
        "chorus_milp_binary_vars": 0,
        "chorus_milp_constraints": 0,
        "chorus_milp_solve_time_s": 0.0,
        "chorus_milp_final_gap": 0.0,
        "chorus_milp_status": -1,
        "chorus_milp_status_msg": "",
        "chorus_runtime_max_live_bytes": 0,
        "chorus_runtime_safe_total_bytes": 0,
        "chorus_runtime_static_margin_bytes": 0,
        "replicate_small_param_numel": 0,
        "replicated_params": 0,
        "replicated_global_numel": 0,
        "sharded_params": 0,
        "dtensor_params": 0,
        "dtensor_local_numel": 0,
        "dtensor_global_numel": 0,
        "coalesce_bucket_mb": 0,
    }
    clear_simplefsdp_cross_graph_cache = None
    if is_simplefsdp:
        simplefsdp_stats = configure_native_simplefsdp(
            model,
            accelerator,
            replicate_small_param_numel=args.simplefsdp_replicate_small_param_numel,
            enable_chorus=args.simplefsdp_enable_chorus,
            chorus_global_retention_mb=args.simplefsdp_chorus_global_retention_mb,
            chorus_global_retention_max_layers=args.simplefsdp_chorus_global_retention_max_layers,
            chorus_milp_time_limit_s=args.simplefsdp_chorus_milp_time_limit_s,
            activation_checkpointing=bool(
                activation_checkpointing_enabled or simplefsdp_selective_checkpoint_stats.get("enabled", False)
            ),
            chorus_persistent_usable_fraction=args.simplefsdp_chorus_persistent_usable_fraction,
            chorus_persistent_baseline_param_multiplier=args.simplefsdp_chorus_persistent_baseline_param_multiplier,
            chorus_persistent_cost_multiplier=args.simplefsdp_chorus_persistent_cost_multiplier,
            chorus_persistent_static_margin_mb=args.simplefsdp_chorus_persistent_static_margin_mb,
        )
        simplefsdp_stats.setdefault("chorus_runtime_max_live_bytes", 0)
        simplefsdp_stats.setdefault("chorus_runtime_safe_total_bytes", 0)
        simplefsdp_stats.setdefault("chorus_runtime_static_margin_bytes", 0)
        simplefsdp_use_fused_optimizer = (
            bool(args.simplefsdp_enable_fused_optimizer)
            or (torch.cuda.is_available() and not bool(args.simplefsdp_disable_fused_optimizer))
        )
        optimizer, optimizer_name = make_adamw_optimizer(
            model.parameters(),
            lr=args.learning_rate,
            fused=simplefsdp_use_fused_optimizer,
        )
        simplefsdp_stats["optimizer"] = optimizer_name
        simplefsdp_stats["inductor_comm_overlap"] = bool(simplefsdp_inductor_comm_overlap)
        simplefsdp_stats["compiled_autograd"] = bool(simplefsdp_compiled_autograd)
        simplefsdp_stats["comm_overlap_policy"] = (
            args.simplefsdp_comm_overlap_policy if simplefsdp_inductor_comm_overlap else "off"
        )
        simplefsdp_stats["comm_overlap_passes"] = ",".join(
            getattr(config, "reorder_for_compute_comm_overlap_passes", [])
        ) if simplefsdp_inductor_comm_overlap else ""
        effective_chorus_live_mb = int(args.simplefsdp_chorus_live_mb)
        if simplefsdp_inductor_comm_overlap and args.simplefsdp_enable_chorus:
            if effective_chorus_live_mb < 0:
                auto_live_mb = int(simplefsdp_stats.get("chorus_auto_budget_runtime_budget_bytes", 0)) // (1024 * 1024)
                if auto_live_mb <= 0:
                    auto_live_mb = int(simplefsdp_stats.get("chorus_global_retention_budget_mb", 0))
                # Graph-local scheduling and persistent full-param retention have
                # different runtime costs. Use the SimpleFSDP-aware headroom for
                # graph-local live AGs instead of tying it to the persistent MILP
                # budget, which may intentionally be zero.
                if auto_live_mb > 0:
                    if int(args.seq_length) >= 2048:
                        # Long-sequence runs already have high activation and optimizer
                        # pressure. Use a small, high-confidence cross-graph budget so
                        # selected retain_get nodes hit without carrying too many
                        # long-lived full parameters through high-activation regions.
                        effective_chorus_live_mb = max(512, min(auto_live_mb // 2, 2048))
                    else:
                        effective_chorus_live_mb = max(1024, min(auto_live_mb, 16384))
                else:
                    # No measured/estimated safe budget means no graph-retained
                    # full-parameter buffers. Retain_get falls back to all-gather.
                    effective_chorus_live_mb = 0
            from native_simplefsdp import (
                clear_simplefsdp_chorus_cross_graph_cache,
                configure_simplefsdp_chorus_runtime_memory_budget,
                simplefsdp_chorus_collectives_graph_pass,
            )

            bucket_bytes = int(args.simplefsdp_coalesce_bucket_mb) * 1024 * 1024
            live_bytes = max(0, int(effective_chorus_live_mb)) * 1024 * 1024
            runtime_budget_stats = configure_simplefsdp_chorus_runtime_memory_budget(
                live_bytes,
                usable_fraction=float(args.simplefsdp_chorus_persistent_usable_fraction),
                static_margin_mb=int(args.simplefsdp_chorus_persistent_static_margin_mb),
            )
            clear_simplefsdp_cross_graph_cache = clear_simplefsdp_chorus_cross_graph_cache
            simplefsdp_stats["chorus_runtime_max_live_bytes"] = int(runtime_budget_stats.get("max_live_bytes", 0))
            simplefsdp_stats["chorus_runtime_safe_total_bytes"] = int(runtime_budget_stats.get("safe_total_bytes", 0))
            simplefsdp_stats["chorus_runtime_static_margin_bytes"] = int(runtime_budget_stats.get("static_margin_bytes", 0))
            effective_chorus_prefetch_groups = int(args.simplefsdp_chorus_prefetch_groups)
            if effective_chorus_prefetch_groups < 0:
                # Long-sequence NVLink runs are usually compute-bound; avoid moving
                # all-gathers unless the user explicitly asks for the ablation.
                effective_chorus_prefetch_groups = 0 if int(args.seq_length) >= 2048 else 2
            prefetch_groups = effective_chorus_prefetch_groups
            config.post_grad_custom_pre_pass = (
                lambda graph, bucket_bytes=bucket_bytes, live_bytes=live_bytes, prefetch_groups=prefetch_groups: simplefsdp_chorus_collectives_graph_pass(
                    graph,
                    max_bucket_bytes=bucket_bytes,
                    prefetch_groups=prefetch_groups,
                    max_live_bytes=live_bytes,
                    milp_time_limit_s=float(args.simplefsdp_chorus_milp_time_limit_s),
                    enable_cross_graph_retention=not bool(args.simplefsdp_chorus_disable_cross_graph_retention),
                )
            )
        simplefsdp_stats["coalesce_bucket_mb"] = (
            int(args.simplefsdp_coalesce_bucket_mb) if simplefsdp_inductor_comm_overlap else 0
        )
        simplefsdp_stats["chorus_enabled"] = bool(
            simplefsdp_inductor_comm_overlap and args.simplefsdp_enable_chorus
        )
        simplefsdp_stats["chorus_prefetch_groups"] = (
            int(effective_chorus_prefetch_groups) if simplefsdp_stats["chorus_enabled"] else 0
        )
        simplefsdp_stats["chorus_live_mb"] = (
            int(effective_chorus_live_mb) if simplefsdp_stats["chorus_enabled"] else 0
        )
        simplefsdp_stats["activation_checkpointing"] = bool(
            activation_checkpointing_enabled or simplefsdp_selective_checkpoint_stats.get("enabled", False)
        )
        simplefsdp_stats["full_activation_checkpointing"] = bool(activation_checkpointing_enabled)
        simplefsdp_stats["selective_checkpoint_layers"] = int(simplefsdp_selective_checkpoint_stats.get("layers", 0))
        simplefsdp_stats["selective_checkpoint_every_n_layers"] = int(
            simplefsdp_selective_checkpoint_stats.get("every_n_layers", 0)
        )
        if accelerator.is_main_process:
            print(
                "[simplefsdp] "
                f"recipe={simplefsdp_stats['recipe']} "
                f"layers={simplefsdp_stats['layers']} "
                f"layer_group_size={simplefsdp_stats['layer_group_size']} "
                f"layer_groups={simplefsdp_stats['layer_groups']} "
                f"sharded_frontend_modules={simplefsdp_stats['sharded_frontend_modules']} "
                f"enable_weight_tying={simplefsdp_stats['enable_weight_tying']} "
                f"reshard_after_forward_policy={simplefsdp_stats['reshard_after_forward_policy']} "
                f"reshard_after_forward={simplefsdp_stats['reshard_after_forward']} "
                f"reshard_after_backward_policy={simplefsdp_stats['reshard_after_backward_policy']} "
                f"reshard_after_backward={simplefsdp_stats['reshard_after_backward']} "
                f"reshard_after_backward_modules={simplefsdp_stats['reshard_after_backward_modules']} "
                f"gradient_division_disabled_modules={simplefsdp_stats['gradient_division_disabled_modules']} "
                f"activation_checkpointing={simplefsdp_stats['activation_checkpointing']} "
                f"full_activation_checkpointing={simplefsdp_stats['full_activation_checkpointing']} "
                f"selective_checkpoint_layers={simplefsdp_stats['selective_checkpoint_layers']} "
                f"selective_checkpoint_every_n_layers={simplefsdp_stats['selective_checkpoint_every_n_layers']} "
                f"optimizer={simplefsdp_stats['optimizer']} "
                f"inductor_comm_overlap={simplefsdp_stats['inductor_comm_overlap']} "
                f"compiled_autograd={simplefsdp_stats['compiled_autograd']} "
                f"comm_overlap_policy={simplefsdp_stats['comm_overlap_policy']} "
                f"comm_overlap_passes={simplefsdp_stats['comm_overlap_passes']} "
                f"coalesce_bucket_mb={simplefsdp_stats['coalesce_bucket_mb']} "
                f"chorus_enabled={simplefsdp_stats['chorus_enabled']} "
                f"chorus_prefetch_groups={simplefsdp_stats['chorus_prefetch_groups']} "
                f"chorus_live_mb={simplefsdp_stats['chorus_live_mb']} "
                f"chorus_global_retention_enabled={simplefsdp_stats['chorus_global_retention_enabled']} "
                f"chorus_global_retention_method={simplefsdp_stats['chorus_global_retention_method']} "
                f"chorus_global_retention_layers={simplefsdp_stats['chorus_global_retention_layers']} "
                f"chorus_global_retention_layer_ids={simplefsdp_stats['chorus_global_retention_layer_ids']} "
                f"chorus_global_retention_params={simplefsdp_stats['chorus_global_retention_params']} "
                f"chorus_global_retention_param_names={simplefsdp_stats['chorus_global_retention_param_names']} "
                f"chorus_global_retention_budget_mode={simplefsdp_stats['chorus_global_retention_budget_mode']} "
                f"chorus_global_retention_requested_mb={simplefsdp_stats['chorus_global_retention_requested_mb']} "
                f"chorus_global_retention_budget_mb={simplefsdp_stats['chorus_global_retention_budget_mb']} "
                f"chorus_global_retention_extra_bytes={simplefsdp_stats['chorus_global_retention_extra_bytes']} "
                f"chorus_global_retention_runtime_extra_bytes={simplefsdp_stats['chorus_global_retention_runtime_extra_bytes']} "
                f"chorus_global_retention_global_bytes={simplefsdp_stats['chorus_global_retention_global_bytes']} "
                f"chorus_auto_budget_total_mem_bytes={simplefsdp_stats['chorus_auto_budget_total_mem_bytes']} "
                f"chorus_auto_budget_usable_fraction={simplefsdp_stats['chorus_auto_budget_usable_fraction']:.4f} "
                f"chorus_auto_budget_safe_total_bytes={simplefsdp_stats['chorus_auto_budget_safe_total_bytes']} "
                f"chorus_auto_budget_model_param_bytes={simplefsdp_stats['chorus_auto_budget_model_param_bytes']} "
                f"chorus_auto_budget_target_param_bytes={simplefsdp_stats['chorus_auto_budget_target_param_bytes']} "
                f"chorus_auto_budget_local_target_param_bytes={simplefsdp_stats['chorus_auto_budget_local_target_param_bytes']} "
                f"chorus_auto_budget_runtime_floor_bytes={simplefsdp_stats['chorus_auto_budget_runtime_floor_bytes']} "
                f"chorus_auto_budget_runtime_headroom_bytes={simplefsdp_stats['chorus_auto_budget_runtime_headroom_bytes']} "
                f"chorus_auto_budget_runtime_live_cap_bytes={simplefsdp_stats['chorus_auto_budget_runtime_live_cap_bytes']} "
                f"chorus_auto_budget_estimated_baseline_bytes={simplefsdp_stats['chorus_auto_budget_estimated_baseline_bytes']} "
                f"chorus_auto_budget_runtime_budget_bytes={simplefsdp_stats['chorus_auto_budget_runtime_budget_bytes']} "
                f"chorus_auto_budget_runtime_cost_multiplier={simplefsdp_stats['chorus_auto_budget_runtime_cost_multiplier']:.4f} "
                f"chorus_milp_binary_vars={simplefsdp_stats['chorus_milp_binary_vars']} "
                f"chorus_milp_constraints={simplefsdp_stats['chorus_milp_constraints']} "
                f"chorus_milp_solve_time_s={simplefsdp_stats['chorus_milp_solve_time_s']:.6f} "
                f"chorus_milp_final_gap={simplefsdp_stats['chorus_milp_final_gap']:.6g} "
                f"chorus_milp_status={simplefsdp_stats['chorus_milp_status']} "
                f"replicate_small_param_numel={simplefsdp_stats['replicate_small_param_numel']} "
                f"replicated_params={simplefsdp_stats['replicated_params']} "
                f"replicated_global_numel={simplefsdp_stats['replicated_global_numel']} "
                f"sharded_params={simplefsdp_stats['sharded_params']} "
                f"explicit_prefetch_ablation={simplefsdp_stats['explicit_prefetch_ablation']} "
                f"prefetch_enabled={simplefsdp_stats['prefetch_enabled']} "
                f"forward_distance={simplefsdp_stats['forward_distance']} "
                f"backward_distance={simplefsdp_stats['backward_distance']} "
                f"dtensor_params={simplefsdp_stats['dtensor_params']} "
                f"dtensor_local_numel={simplefsdp_stats['dtensor_local_numel']} "
                f"dtensor_global_numel={simplefsdp_stats['dtensor_global_numel']}"
            )
    else:
        # Prepare optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Prepare everything with accelerator
        model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
        print(f"Model prepared: {model.__class__} optimizer: {optimizer.__class__}")

    fsdp2_prefetch_stats = {
        "enabled": False,
        "layers": 0,
        "forward_distance": 0,
        "backward_distance": 0,
        "chorus_enabled": False,
        "chorus_live_mb": 0,
        "chorus_forward_prefetches": 0,
        "chorus_backward_prefetches": 0,
    }
    if (not is_simplefsdp) and _is_accelerate_fsdp2(accelerator) and not args.disable_fsdp2_prefetch:
        if args.fsdp2_enable_chorus:
            fsdp2_prefetch_stats = configure_fsdp2_chorus_prefetch(
                model,
                forward_max_distance=args.fsdp2_forward_prefetch_distance,
                backward_max_distance=args.fsdp2_backward_prefetch_distance,
                live_budget_mb=args.fsdp2_chorus_live_mb,
            )
        else:
            fsdp2_prefetch_stats = configure_fsdp2_explicit_prefetch(
                model,
                forward_distance=args.fsdp2_forward_prefetch_distance,
                backward_distance=args.fsdp2_backward_prefetch_distance,
            )
        if accelerator.is_main_process:
            print(
                "[fsdp2] explicit prefetch "
                f"enabled={fsdp2_prefetch_stats['enabled']} "
                f"layers={fsdp2_prefetch_stats['layers']} "
                f"forward_distance={fsdp2_prefetch_stats['forward_distance']} "
                f"backward_distance={fsdp2_prefetch_stats['backward_distance']} "
                f"chorus_enabled={fsdp2_prefetch_stats.get('chorus_enabled', False)} "
                f"chorus_live_mb={fsdp2_prefetch_stats.get('chorus_live_mb', 0)} "
                f"chorus_forward_prefetches={fsdp2_prefetch_stats.get('chorus_forward_prefetches', 0)} "
                f"chorus_backward_prefetches={fsdp2_prefetch_stats.get('chorus_backward_prefetches', 0)}"
            )

    if "Mixtral" in model_name or "MoE" in model_name:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True

    if is_deepspeed:
        if args.compile:
            schedule = make_schedule(args.passes.split(","), warmup=5) if args.passes else None
            model.compile(backend=args.backend, schedule=schedule)
    else:
        if is_simplefsdp and not args.compile and accelerator.is_main_process:
            print("[simplefsdp] WARNING: SimpleFSDP is designed for torch.compile; running eager may be slow.")
        if args.compile:
            model = torch.compile(model, backend=args.backend)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = args.model_name.split("/")[-1]
    exp_name = f"{model_name}_np{accelerator.num_processes}ds{1 if is_deepspeed else 0}" \
               f"B{args.backend}z{args.zero_stage}" \
               f"L{0 if args.num_layers is None else args.num_layers}" \
               f"bs{args.batch_size}seq{args.seq_length}acc{args.gradient_accumulation_steps}ac{1 if args.activation_checkpointing else 0}" \
               f"pass_{'none' if args.passes is None else args.passes.replace(',', '_')}_" \
               f"os{1 if args.offload_opt_states else 0}" \
               f"T{timestamp}"
    clear_simplefsdp_cache = None
    if is_simplefsdp and simplefsdp_stats.get("chorus_global_retention_enabled", False):
        from native_simplefsdp import clear_simplefsdp_chorus_persistent_cache
        clear_simplefsdp_cache = clear_simplefsdp_chorus_persistent_cache

    if args.profile_dir:
        if accelerator.is_main_process and args.profile_dir:
            os.makedirs(args.profile_dir, exist_ok=True)
            if args.profile:
                prof_dir = f"{args.profile_dir}/{exp_name}"
                os.makedirs(prof_dir, exist_ok=True)
        accelerator.wait_for_everyone()        
        
    do_profile = args.profile and accelerator.is_main_process
    prof_context = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=10*args.gradient_accumulation_steps, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
    ) if do_profile else nullcontext()

    # Training 
    if args.eval:
        model.eval()
    else:
        model.train()

    global_step = 0
    iter_times = []
    iter_times_by_step = {}
    memory_report_start_step = 7
    memory_peak_reset_done = False

    # See https://github.com/microsoft/DeepSpeed/issues/6793
    acc_context = nullcontext if (is_deepspeed or is_simplefsdp) else accelerator.accumulate

    normal_stop = 11 * args.gradient_accumulation_steps
    profiler_stop = 10 * args.gradient_accumulation_steps + 3
    stop_after_microsteps = max(normal_stop, profiler_stop if args.profile else normal_stop)
    stop = False
    with prof_context as prof:
        step_compute_time = 0.0
        for epoch in range(args.num_epochs):
            for step, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # Keep Chorus persistent full-parameter buffers live across
                # microsteps. Version checks in the parametrization invalidate
                # entries after optimizer updates, while preserving reuse inside
                # activation recompute and gradient accumulation windows.

                # Time only the training compute segment (exclude dataloader/epoch-boundary overhead).
                # Start timing after the batch is already moved to device.
                micro_start = time.time()
                
                # with acc_context(model):
                #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
                #     loss = outputs.loss

                # 解决batch size=1时，triton报错的问题
                with acc_context(model):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    logits = outputs.logits
                    shift_labels = F.pad(input_ids, (0, 1), value=-100)[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        shift_labels.view(-1).to(logits.device),
                        ignore_index=-100,
                    )

                    update_step = (is_deepspeed and model.is_gradient_accumulation_boundary()) \
                        or (not is_deepspeed and accelerator.sync_gradients)
                    accelerator.backward(loss)
                    if clear_simplefsdp_cross_graph_cache is not None:
                        clear_simplefsdp_cross_graph_cache()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if update_step:
                        step_compute_time += time.time() - micro_start
                        step_time = step_compute_time
                        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        if accelerator.is_main_process and global_step % (args.print_interval * args.gradient_accumulation_steps) == 0:
                            print(f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item()} sync: {accelerator.sync_gradients} time: {step_time} alloc_mem: {alloc_gb:.2f} GB peak_mem: {peak_gb:.2f} GB")

                        iter_times.append(step_time)
                        iter_times_by_step[int(global_step)] = float(step_time)
                        step_compute_time = 0.0
                        if (not memory_peak_reset_done) and global_step >= memory_report_start_step - 1:
                            torch.cuda.reset_peak_memory_stats()
                            memory_peak_reset_done = True
                    else:
                        step_compute_time += time.time() - micro_start

                if do_profile:
                    prof.step()

                # Keep the normal benchmark at 11 optimizer-update windows.
                # Profiling needs enough microsteps to finish its 10*GAS
                # warmup plus three active steps and flush the trace.
                stop = global_step >= stop_after_microsteps
                if stop:
                    break
            if stop:
                break

    # Report iteration time as the mean of true times from step 7..11 (5 steps).
    _report_steps = list(range(7, 12))
    report_times = [iter_times_by_step[s] for s in _report_steps if s in iter_times_by_step]
    if not report_times:
        report_times = iter_times

    alloc_bytes = int(torch.cuda.memory_allocated())
    peak_bytes = int(torch.cuda.max_memory_allocated())
    alloc_global_bytes = distributed_max_int(alloc_bytes)
    peak_global_bytes = distributed_max_int(peak_bytes)

    steady_iteration_time = float(sum(report_times) / len(report_times)) if report_times else 0.0
    first_step_time = float(iter_times_by_step.get(1, 0.0))

    if accelerator.is_main_process:
        compile_time_sum = 0.0
        compile_time = 0
        compile_time_source = "none"
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = float(sum(t for _, _, _, t in compile_time))
            compile_time_source = "model.get_compile_time"
        if args.compile and compile_time_sum <= 0.0 and first_step_time > 0.0 and steady_iteration_time > 0.0:
            # Plain torch.compile/Inductor paths, including SimpleFSDP, compile
            # lazily during the first training step. Estimate compile overhead as
            # first-step wall time minus the steady-state iteration time reported
            # from steps 7..11. This keeps compile cost visible without changing
            # the timed steady-state metric.
            compile_time_sum = max(0.0, first_step_time - steady_iteration_time)
            compile_time_source = "first_step_minus_steady"

        is_deepcompile = is_deepspeed and model._config.compile_config.deepcompile
        alloc_gb = float(alloc_bytes) / (1024**3)
        alloc_global_gb = float(alloc_global_bytes) / (1024**3)
        peak_gb = float(peak_bytes) / (1024**3)
        peak_global_gb = float(peak_global_bytes) / (1024**3)

        milp_stats = ""
        try:
            from deepspeed.compile.passes import global_layer_scheduler
            schedule = getattr(global_layer_scheduler, "_LATEST_SCHEDULE", None)
            if schedule:
                meta = schedule.get("meta", {})
                milp_meta = meta.get("milp", {}) or {}
                if milp_meta:
                    milp_units = int(milp_meta.get("units", 0))
                    milp_blocks = int(milp_meta.get("blocks", 0))
                    milp_binary_vars = int(milp_meta.get("binary_vars", 0))
                    milp_constraints = int(milp_meta.get("constraints", 0))
                    milp_solve_time_s = float(milp_meta.get("solve_time_s", 0.0))
                    milp_final_gap_raw = milp_meta.get("final_gap", milp_meta.get("mip_gap", float("nan")))
                    milp_final_gap = float(milp_final_gap_raw) if milp_final_gap_raw is not None else float("nan")
                    milp_status = int(milp_meta.get("status", -1))
                    milp_status_msg = str(milp_meta.get("message", ""))
                    milp_stats = (
                        f" milp_units: {milp_units}"
                        f" milp_blocks: {milp_blocks}"
                        f" milp_binary_vars: {milp_binary_vars}"
                        f" milp_constraints: {milp_constraints}"
                        f" milp_solve_time_s: {milp_solve_time_s:.6f}"
                        f" milp_final_gap: {milp_final_gap:.6g}"
                        f" milp_status: {milp_status}"
                        f" milp_status_msg: {milp_status_msg}"
                    )
        except Exception:
            milp_stats = ""

        simplefsdp_chorus_diag = {
            "cache_trace_hits": 0,
            "cache_trace_misses": 0,
            "cache_eager_hits": 0,
            "cache_eager_misses": 0,
            "cache_no_grad_bypass": 0,
            "cache_entries": 0,
            "cache_persistent_params": 0,
            "graph_ag_before": 0,
            "graph_ag_buckets": 0,
            "graph_ag_retained": 0,
            "graph_ag_coalesced_or_moved": 0,
            "graph_rs_before": 0,
            "graph_rs_coalesced": 0,
            "graph_method": "none",
            "graph_status": -1,
            "graph_history_len": 0,
            "graph_history_ag_before": "",
            "graph_history_rs_before": "",
            "graph_history_ag_retained": "",
            "graph_history_cross_graph_puts": "",
            "graph_history_cross_graph_gets": "",
            "graph_cross_graph_puts": 0,
            "graph_cross_graph_gets": 0,
            "graph_cross_graph_selected": 0,
            "graph_cross_graph_selected_bytes": 0,
            "graph_cross_graph_get_bytes": 0,
            "graph_cross_graph_budget_bytes": 0,
            "graph_cross_graph_method": "none",
            "graph_cross_graph_status": -1,
            "graph_cross_graph_solve_time_s": 0.0,
            "runtime_attempted_puts": 0,
            "runtime_admitted_puts": 0,
            "runtime_admitted_bytes": 0,
            "runtime_retain_aliases": 0,
            "runtime_retain_clones": 0,
            "runtime_clone_mode": 0,
            "runtime_skipped_disabled": 0,
            "runtime_skipped_live_budget": 0,
            "runtime_skipped_memory_pressure": 0,
            "runtime_skipped_oom": 0,
            "runtime_hit_gets": 0,
            "runtime_fallback_gets": 0,
            "runtime_peak_retained_bytes": 0,
            "runtime_peak_driver_used_bytes": 0,
            "runtime_peak_allocated_bytes": 0,
            "runtime_peak_reserved_bytes": 0,
        }
        if simplefsdp_stats.get("enabled", False):
            try:
                import native_simplefsdp as _native_simplefsdp
                cache_diag = _native_simplefsdp.summarize_simplefsdp_chorus_persistent_cache(model)
                runtime_diag = _native_simplefsdp.summarize_simplefsdp_chorus_runtime_retention()
                graph_diag = dict(getattr(_native_simplefsdp, "_LATEST_CHORUS_GRAPH_STATS", {}) or {})
                graph_history = list(getattr(_native_simplefsdp, "_CHORUS_GRAPH_HISTORY", []) or [])
                simplefsdp_chorus_diag.update({
                    "cache_trace_hits": int(cache_diag.get("trace_hits", 0)),
                    "cache_trace_misses": int(cache_diag.get("trace_misses", 0)),
                    "cache_eager_hits": int(cache_diag.get("eager_hits", 0)),
                    "cache_eager_misses": int(cache_diag.get("eager_misses", 0)),
                    "cache_no_grad_bypass": int(cache_diag.get("no_grad_bypass", 0)),
                    "cache_entries": int(cache_diag.get("cache_entries", 0)),
                    "cache_persistent_params": int(cache_diag.get("persistent_params", 0)),
                    "graph_ag_before": int(graph_diag.get("ag_before", 0)),
                    "graph_ag_buckets": int(graph_diag.get("ag_buckets", 0)),
                    "graph_ag_retained": int(graph_diag.get("ag_retained", 0)),
                    "graph_ag_coalesced_or_moved": int(graph_diag.get("ag_coalesced_or_moved", 0)),
                    "graph_rs_before": int(graph_diag.get("rs_before", 0)),
                    "graph_rs_coalesced": int(graph_diag.get("rs_coalesced", 0)),
                    "graph_method": str(graph_diag.get("method", "none")),
                    "graph_status": int(graph_diag.get("status", -1)),
                    "graph_history_len": int(len(graph_history)),
                    "graph_history_ag_before": ",".join(str(int(item.get("ag_before", 0))) for item in graph_history[-8:]),
                    "graph_history_rs_before": ",".join(str(int(item.get("rs_before", 0))) for item in graph_history[-8:]),
                    "graph_history_ag_retained": ",".join(str(int(item.get("ag_retained", 0))) for item in graph_history[-8:]),
                    "graph_history_cross_graph_puts": ",".join(str(int(item.get("cross_graph_puts", 0))) for item in graph_history[-8:]),
                    "graph_history_cross_graph_gets": ",".join(str(int(item.get("cross_graph_gets", 0))) for item in graph_history[-8:]),
                    "graph_cross_graph_puts": int(graph_diag.get("cross_graph_puts", 0)),
                    "graph_cross_graph_gets": int(graph_diag.get("cross_graph_gets", 0)),
                    "graph_cross_graph_selected": int(graph_diag.get("cross_graph_selected", 0)),
                    "graph_cross_graph_selected_bytes": int(graph_diag.get("cross_graph_selected_bytes", 0)),
                    "graph_cross_graph_get_bytes": int(graph_diag.get("cross_graph_get_bytes", 0)),
                    "graph_cross_graph_budget_bytes": int(graph_diag.get("cross_graph_budget_bytes", 0)),
                    "graph_cross_graph_method": str(graph_diag.get("cross_graph_method", "none")),
                    "graph_cross_graph_status": int(graph_diag.get("cross_graph_status", -1)),
                    "graph_cross_graph_solve_time_s": float(graph_diag.get("cross_graph_solve_time_s", 0.0)),
                    "runtime_attempted_puts": int(runtime_diag.get("attempted_puts", 0)),
                    "runtime_admitted_puts": int(runtime_diag.get("admitted_puts", 0)),
                    "runtime_admitted_bytes": int(runtime_diag.get("admitted_bytes", 0)),
                    "runtime_retain_aliases": int(runtime_diag.get("retain_aliases", 0)),
                    "runtime_retain_clones": int(runtime_diag.get("retain_clones", 0)),
                    "runtime_clone_mode": int(runtime_diag.get("clone_mode", 0)),
                    "runtime_skipped_disabled": int(runtime_diag.get("skipped_disabled", 0)),
                    "runtime_skipped_live_budget": int(runtime_diag.get("skipped_live_budget", 0)),
                    "runtime_skipped_memory_pressure": int(runtime_diag.get("skipped_memory_pressure", 0)),
                    "runtime_skipped_oom": int(runtime_diag.get("skipped_oom", 0)),
                    "runtime_hit_gets": int(runtime_diag.get("hit_gets", 0)),
                    "runtime_fallback_gets": int(runtime_diag.get("fallback_gets", 0)),
                    "runtime_peak_retained_bytes": int(runtime_diag.get("peak_retained_bytes", 0)),
                    "runtime_peak_driver_used_bytes": int(runtime_diag.get("peak_driver_used_bytes", 0)),
                    "runtime_peak_allocated_bytes": int(runtime_diag.get("peak_allocated_bytes", 0)),
                    "runtime_peak_reserved_bytes": int(runtime_diag.get("peak_reserved_bytes", 0)),
                })
            except Exception:
                pass

        fsdp2_stats = (
            f" fsdp2_prefetch_enabled: {fsdp2_prefetch_stats['enabled']}"
            f" fsdp2_prefetch_layers: {fsdp2_prefetch_stats['layers']}"
            f" fsdp2_forward_prefetch_distance: {fsdp2_prefetch_stats['forward_distance']}"
            f" fsdp2_backward_prefetch_distance: {fsdp2_prefetch_stats['backward_distance']}"
            f" fsdp2_chorus_enabled: {fsdp2_prefetch_stats.get('chorus_enabled', False)}"
            f" fsdp2_chorus_live_mb: {fsdp2_prefetch_stats.get('chorus_live_mb', 0)}"
            f" fsdp2_chorus_forward_prefetches: {fsdp2_prefetch_stats.get('chorus_forward_prefetches', 0)}"
            f" fsdp2_chorus_backward_prefetches: {fsdp2_prefetch_stats.get('chorus_backward_prefetches', 0)}"
        ) if fsdp2_prefetch_stats.get("enabled", False) else ""
        simplefsdp_msg = (
            f" simplefsdp_enabled: {simplefsdp_stats['enabled']}"
            f" simplefsdp_recipe: {simplefsdp_stats['recipe']}"
            f" simplefsdp_layers: {simplefsdp_stats['layers']}"
            f" simplefsdp_layer_group_size: {simplefsdp_stats['layer_group_size']}"
            f" simplefsdp_layer_groups: {simplefsdp_stats['layer_groups']}"
            f" simplefsdp_sharded_frontend_modules: {simplefsdp_stats['sharded_frontend_modules']}"
            f" simplefsdp_enable_weight_tying: {simplefsdp_stats['enable_weight_tying']}"
            f" simplefsdp_reshard_after_forward_policy: {simplefsdp_stats['reshard_after_forward_policy']}"
            f" simplefsdp_reshard_after_forward: {simplefsdp_stats['reshard_after_forward']}"
            f" simplefsdp_reshard_after_backward_policy: {simplefsdp_stats['reshard_after_backward_policy']}"
            f" simplefsdp_reshard_after_backward: {simplefsdp_stats['reshard_after_backward']}"
            f" simplefsdp_reshard_after_backward_modules: {simplefsdp_stats['reshard_after_backward_modules']}"
            f" simplefsdp_gradient_division_disabled_modules: {simplefsdp_stats['gradient_division_disabled_modules']}"
            f" simplefsdp_activation_checkpointing: {simplefsdp_stats['activation_checkpointing']}"
            f" simplefsdp_full_activation_checkpointing: {simplefsdp_stats['full_activation_checkpointing']}"
            f" simplefsdp_selective_checkpoint_layers: {simplefsdp_stats['selective_checkpoint_layers']}"
            f" simplefsdp_selective_checkpoint_every_n_layers: {simplefsdp_stats['selective_checkpoint_every_n_layers']}"
            f" simplefsdp_optimizer: {simplefsdp_stats['optimizer']}"
            f" simplefsdp_inductor_comm_overlap: {simplefsdp_stats['inductor_comm_overlap']}"
            f" simplefsdp_compiled_autograd: {simplefsdp_stats['compiled_autograd']}"
            f" simplefsdp_comm_overlap_policy: {simplefsdp_stats['comm_overlap_policy']}"
            f" simplefsdp_comm_overlap_passes: {simplefsdp_stats['comm_overlap_passes']}"
            f" simplefsdp_coalesce_bucket_mb: {simplefsdp_stats['coalesce_bucket_mb']}"
            f" simplefsdp_chorus_enabled: {simplefsdp_stats['chorus_enabled']}"
            f" simplefsdp_chorus_prefetch_groups: {simplefsdp_stats['chorus_prefetch_groups']}"
            f" simplefsdp_chorus_live_mb: {simplefsdp_stats['chorus_live_mb']}"
            f" simplefsdp_chorus_global_retention_enabled: {simplefsdp_stats['chorus_global_retention_enabled']}"
            f" simplefsdp_chorus_global_retention_method: {simplefsdp_stats['chorus_global_retention_method']}"
            f" simplefsdp_chorus_global_retention_layers: {simplefsdp_stats['chorus_global_retention_layers']}"
            f" simplefsdp_chorus_global_retention_layer_ids: {simplefsdp_stats['chorus_global_retention_layer_ids']}"
            f" simplefsdp_chorus_global_retention_params: {simplefsdp_stats['chorus_global_retention_params']}"
            f" simplefsdp_chorus_global_retention_param_names: {simplefsdp_stats['chorus_global_retention_param_names']}"
            f" simplefsdp_chorus_global_retention_budget_mode: {simplefsdp_stats['chorus_global_retention_budget_mode']}"
            f" simplefsdp_chorus_global_retention_requested_mb: {simplefsdp_stats['chorus_global_retention_requested_mb']}"
            f" simplefsdp_chorus_global_retention_budget_mb: {simplefsdp_stats['chorus_global_retention_budget_mb']}"
            f" simplefsdp_chorus_global_retention_extra_bytes: {simplefsdp_stats['chorus_global_retention_extra_bytes']}"
            f" simplefsdp_chorus_global_retention_runtime_extra_bytes: {simplefsdp_stats['chorus_global_retention_runtime_extra_bytes']}"
            f" simplefsdp_chorus_global_retention_global_bytes: {simplefsdp_stats['chorus_global_retention_global_bytes']}"
            f" simplefsdp_chorus_auto_budget_total_mem_bytes: {simplefsdp_stats['chorus_auto_budget_total_mem_bytes']}"
            f" simplefsdp_chorus_auto_budget_usable_fraction: {simplefsdp_stats['chorus_auto_budget_usable_fraction']:.4f}"
            f" simplefsdp_chorus_auto_budget_safe_total_bytes: {simplefsdp_stats['chorus_auto_budget_safe_total_bytes']}"
            f" simplefsdp_chorus_auto_budget_model_param_bytes: {simplefsdp_stats['chorus_auto_budget_model_param_bytes']}"
            f" simplefsdp_chorus_auto_budget_target_param_bytes: {simplefsdp_stats['chorus_auto_budget_target_param_bytes']}"
            f" simplefsdp_chorus_auto_budget_local_target_param_bytes: {simplefsdp_stats['chorus_auto_budget_local_target_param_bytes']}"
            f" simplefsdp_chorus_auto_budget_runtime_floor_bytes: {simplefsdp_stats['chorus_auto_budget_runtime_floor_bytes']}"
            f" simplefsdp_chorus_auto_budget_runtime_headroom_bytes: {simplefsdp_stats['chorus_auto_budget_runtime_headroom_bytes']}"
            f" simplefsdp_chorus_auto_budget_runtime_live_cap_bytes: {simplefsdp_stats['chorus_auto_budget_runtime_live_cap_bytes']}"
            f" simplefsdp_chorus_auto_budget_estimated_baseline_bytes: {simplefsdp_stats['chorus_auto_budget_estimated_baseline_bytes']}"
            f" simplefsdp_chorus_auto_budget_runtime_budget_bytes: {simplefsdp_stats['chorus_auto_budget_runtime_budget_bytes']}"
            f" simplefsdp_chorus_auto_budget_runtime_cost_multiplier: {simplefsdp_stats['chorus_auto_budget_runtime_cost_multiplier']:.4f}"
            f" simplefsdp_chorus_milp_binary_vars: {simplefsdp_stats['chorus_milp_binary_vars']}"
            f" simplefsdp_chorus_milp_constraints: {simplefsdp_stats['chorus_milp_constraints']}"
            f" simplefsdp_chorus_milp_solve_time_s: {simplefsdp_stats['chorus_milp_solve_time_s']:.6f}"
            f" simplefsdp_chorus_milp_final_gap: {simplefsdp_stats['chorus_milp_final_gap']:.6g}"
            f" simplefsdp_chorus_milp_status: {simplefsdp_stats['chorus_milp_status']}"
            f" simplefsdp_chorus_runtime_max_live_bytes: {simplefsdp_stats['chorus_runtime_max_live_bytes']}"
            f" simplefsdp_chorus_runtime_safe_total_bytes: {simplefsdp_stats['chorus_runtime_safe_total_bytes']}"
            f" simplefsdp_chorus_runtime_static_margin_bytes: {simplefsdp_stats['chorus_runtime_static_margin_bytes']}"
            f" simplefsdp_replicate_small_param_numel: {simplefsdp_stats['replicate_small_param_numel']}"
            f" simplefsdp_replicated_params: {simplefsdp_stats['replicated_params']}"
            f" simplefsdp_replicated_global_numel: {simplefsdp_stats['replicated_global_numel']}"
            f" simplefsdp_sharded_params: {simplefsdp_stats['sharded_params']}"
            f" simplefsdp_explicit_prefetch_ablation: {simplefsdp_stats['explicit_prefetch_ablation']}"
            f" simplefsdp_prefetch_enabled: {simplefsdp_stats['prefetch_enabled']}"
            f" simplefsdp_forward_prefetch_distance: {simplefsdp_stats['forward_distance']}"
            f" simplefsdp_backward_prefetch_distance: {simplefsdp_stats['backward_distance']}"
            f" simplefsdp_dtensor_params: {simplefsdp_stats['dtensor_params']}"
            f" simplefsdp_dtensor_local_numel: {simplefsdp_stats['dtensor_local_numel']}"
            f" simplefsdp_dtensor_global_numel: {simplefsdp_stats['dtensor_global_numel']}"
            f" simplefsdp_chorus_cache_trace_hits: {simplefsdp_chorus_diag['cache_trace_hits']}"
            f" simplefsdp_chorus_cache_trace_misses: {simplefsdp_chorus_diag['cache_trace_misses']}"
            f" simplefsdp_chorus_cache_eager_hits: {simplefsdp_chorus_diag['cache_eager_hits']}"
            f" simplefsdp_chorus_cache_eager_misses: {simplefsdp_chorus_diag['cache_eager_misses']}"
            f" simplefsdp_chorus_cache_no_grad_bypass: {simplefsdp_chorus_diag['cache_no_grad_bypass']}"
            f" simplefsdp_chorus_cache_entries: {simplefsdp_chorus_diag['cache_entries']}"
            f" simplefsdp_chorus_cache_persistent_params: {simplefsdp_chorus_diag['cache_persistent_params']}"
            f" simplefsdp_chorus_graph_ag_before: {simplefsdp_chorus_diag['graph_ag_before']}"
            f" simplefsdp_chorus_graph_ag_buckets: {simplefsdp_chorus_diag['graph_ag_buckets']}"
            f" simplefsdp_chorus_graph_ag_retained: {simplefsdp_chorus_diag['graph_ag_retained']}"
            f" simplefsdp_chorus_graph_ag_coalesced_or_moved: {simplefsdp_chorus_diag['graph_ag_coalesced_or_moved']}"
            f" simplefsdp_chorus_graph_rs_before: {simplefsdp_chorus_diag['graph_rs_before']}"
            f" simplefsdp_chorus_graph_rs_coalesced: {simplefsdp_chorus_diag['graph_rs_coalesced']}"
            f" simplefsdp_chorus_graph_method: {simplefsdp_chorus_diag['graph_method']}"
            f" simplefsdp_chorus_graph_status: {simplefsdp_chorus_diag['graph_status']}"
            f" simplefsdp_chorus_graph_history_len: {simplefsdp_chorus_diag['graph_history_len']}"
            f" simplefsdp_chorus_graph_history_ag_before: {simplefsdp_chorus_diag['graph_history_ag_before']}"
            f" simplefsdp_chorus_graph_history_rs_before: {simplefsdp_chorus_diag['graph_history_rs_before']}"
            f" simplefsdp_chorus_graph_history_ag_retained: {simplefsdp_chorus_diag['graph_history_ag_retained']}"
            f" simplefsdp_chorus_graph_history_cross_graph_puts: {simplefsdp_chorus_diag['graph_history_cross_graph_puts']}"
            f" simplefsdp_chorus_graph_history_cross_graph_gets: {simplefsdp_chorus_diag['graph_history_cross_graph_gets']}"
            f" simplefsdp_chorus_graph_cross_graph_puts: {simplefsdp_chorus_diag['graph_cross_graph_puts']}"
            f" simplefsdp_chorus_graph_cross_graph_gets: {simplefsdp_chorus_diag['graph_cross_graph_gets']}"
            f" simplefsdp_chorus_graph_cross_graph_selected: {simplefsdp_chorus_diag['graph_cross_graph_selected']}"
            f" simplefsdp_chorus_graph_cross_graph_selected_bytes: {simplefsdp_chorus_diag['graph_cross_graph_selected_bytes']}"
            f" simplefsdp_chorus_graph_cross_graph_get_bytes: {simplefsdp_chorus_diag['graph_cross_graph_get_bytes']}"
            f" simplefsdp_chorus_graph_cross_graph_budget_bytes: {simplefsdp_chorus_diag['graph_cross_graph_budget_bytes']}"
            f" simplefsdp_chorus_graph_cross_graph_method: {simplefsdp_chorus_diag['graph_cross_graph_method']}"
            f" simplefsdp_chorus_graph_cross_graph_status: {simplefsdp_chorus_diag['graph_cross_graph_status']}"
            f" simplefsdp_chorus_graph_cross_graph_solve_time_s: {simplefsdp_chorus_diag['graph_cross_graph_solve_time_s']:.6f}"
            f" simplefsdp_chorus_runtime_attempted_puts: {simplefsdp_chorus_diag['runtime_attempted_puts']}"
            f" simplefsdp_chorus_runtime_admitted_puts: {simplefsdp_chorus_diag['runtime_admitted_puts']}"
            f" simplefsdp_chorus_runtime_admitted_bytes: {simplefsdp_chorus_diag['runtime_admitted_bytes']}"
            f" simplefsdp_chorus_runtime_retain_aliases: {simplefsdp_chorus_diag['runtime_retain_aliases']}"
            f" simplefsdp_chorus_runtime_retain_clones: {simplefsdp_chorus_diag['runtime_retain_clones']}"
            f" simplefsdp_chorus_runtime_clone_mode: {simplefsdp_chorus_diag['runtime_clone_mode']}"
            f" simplefsdp_chorus_runtime_skipped_disabled: {simplefsdp_chorus_diag['runtime_skipped_disabled']}"
            f" simplefsdp_chorus_runtime_skipped_live_budget: {simplefsdp_chorus_diag['runtime_skipped_live_budget']}"
            f" simplefsdp_chorus_runtime_skipped_memory_pressure: {simplefsdp_chorus_diag['runtime_skipped_memory_pressure']}"
            f" simplefsdp_chorus_runtime_skipped_oom: {simplefsdp_chorus_diag['runtime_skipped_oom']}"
            f" simplefsdp_chorus_runtime_hit_gets: {simplefsdp_chorus_diag['runtime_hit_gets']}"
            f" simplefsdp_chorus_runtime_fallback_gets: {simplefsdp_chorus_diag['runtime_fallback_gets']}"
            f" simplefsdp_chorus_runtime_peak_retained_bytes: {simplefsdp_chorus_diag['runtime_peak_retained_bytes']}"
            f" simplefsdp_chorus_runtime_peak_driver_used_bytes: {simplefsdp_chorus_diag['runtime_peak_driver_used_bytes']}"
            f" simplefsdp_chorus_runtime_peak_allocated_bytes: {simplefsdp_chorus_diag['runtime_peak_allocated_bytes']}"
            f" simplefsdp_chorus_runtime_peak_reserved_bytes: {simplefsdp_chorus_diag['runtime_peak_reserved_bytes']}"
        ) if simplefsdp_stats.get("enabled", False) else ""
        activation_checkpointing_report = bool(
            activation_checkpointing_enabled or simplefsdp_stats.get("selective_checkpoint_layers", 0) > 0
        )
        msg = (
            f"{args.model_name} ds={is_deepspeed} np={accelerator.num_processes}"
            f" batch_size={args.batch_size} seq={args.seq_length} zero_stage={args.zero_stage}"
            f" acc={args.gradient_accumulation_steps} ac={activation_checkpointing_report}"
            f" requested_ac={args.activation_checkpointing} compile={args.compile} backend={args.backend}"
            f" distributed_backend={args.distributed_backend} deepcompile={is_deepcompile}"
            f" passes={args.passes} compile_time={compile_time_sum:.4f}"
            f" compile_time_source={compile_time_source}"
            f" first_step_time={first_step_time:.4f}"
            f" iteration time: {steady_iteration_time:.4f}"
            f" alloc_mem_gb: {alloc_gb:.2f}"
            f" peak_mem_gb: {peak_gb:.2f}"
            f" peak_mem_global_gb: {peak_global_gb:.2f}"
            f"{fsdp2_stats}{simplefsdp_msg}{milp_stats}"
        )
        final_metrics_msg = (
            f"Final metrics: iteration_time={steady_iteration_time:.4f}s"
            f" compile_time={compile_time_sum:.4f}s"
            f" alloc_mem={alloc_global_gb:.2f}GB"
            f" peak_mem={peak_global_gb:.2f}GB"
        )
        print(msg)
        print(final_metrics_msg)


    if accelerator.is_main_process and args.profile_dir:
        from pathlib import Path
        filepath = Path(args.profile_dir) / f"result.txt"
        with open(filepath, "a") as f:
            f.write(f"{timestamp} {msg}" + "\n")
            f.write(f"{timestamp} {final_metrics_msg}" + "\n")

        if args.compile:
            filepath = Path(args.profile_dir) / f"compile_time.txt"
            with open(filepath, "a") as f:
                compile_msg =  f"{msg} compile_time_detail={compile_time}"
                f.write(f"{timestamp} {compile_msg}" + "\n")

    # Keep sequential sweep cases aligned across nodes and let rank 0 finish
    # writing metrics before every worker tears down its process group.
    accelerator.wait_for_everyone()

    # # Save the model
    # if accelerator.is_main_process:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained("fine_tuned_model", save_function=accelerator.save)
    #     tokenizer.save_pretrained("fine_tuned_model")

if __name__ == "__main__":
    torch._dynamo.config.accumulated_cache_size_limit = 256
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    main()
