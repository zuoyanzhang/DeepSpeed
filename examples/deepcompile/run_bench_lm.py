import os
import argparse
import subprocess
import threading
import time
from datetime import datetime
from contextlib import nullcontext
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


def query_nvidia_smi_memory_bytes(device_index: int | None = None) -> int:
    if not torch.cuda.is_available():
        return 0
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
        first_line = output.strip().splitlines()[0].strip()
        used_mib = int(first_line.split()[0])
        return used_mib * 1024 * 1024
    except Exception:
        return 0


class NvidiaSmiMemorySampler:
    def __init__(self, device_index: int | None = None, interval_s: float = 0.2) -> None:
        self.device_index = device_index
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_bytes = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self.max_bytes = max(self.max_bytes, query_nvidia_smi_memory_bytes(self.device_index))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self.max_bytes = max(self.max_bytes, query_nvidia_smi_memory_bytes(self.device_index))

    def stop(self) -> int:
        self.max_bytes = max(self.max_bytes, query_nvidia_smi_memory_bytes(self.device_index))
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.max_bytes = max(self.max_bytes, query_nvidia_smi_memory_bytes(self.device_index))
        return int(self.max_bytes)


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
    parser.add_argument("--dataset_name", type=str, default="timdettmers/openassistant-guanaco")
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
                        help="Enable fused AdamW for SimpleFSDP ablations. Default uses the same AdamW path as the other backends.")
    parser.add_argument("--offload_opt_states", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--profile_dir", type=str, default=None)
    parser.add_argument("--warmup_step", type=int, default=15)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--print_interval", type=int, default=1)
    parser.add_argument("--save_weights", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--model_path", type=str, default="/home/dev/")
    parser.add_argument("--disable_fsdp2_prefetch", "--disable-fsdp2-prefetch", action="store_true",
                        help="Disable explicit FSDP2 module prefetching.")
    parser.add_argument("--fsdp2_forward_prefetch_distance", "--fsdp2-forward-prefetch-distance",
                        type=int, default=0,
                        help="Number of following FSDP2 layers to prefetch in forward.")
    parser.add_argument("--fsdp2_backward_prefetch_distance", "--fsdp2-backward-prefetch-distance",
                        type=int, default=1,
                        help="Number of preceding FSDP2 layers to prefetch in backward.")

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
        return {"enabled": False, "layers": 0, "forward_distance": 0, "backward_distance": 0}

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
    }


def make_adamw_optimizer(params, lr: float, fused: bool = False):
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr=lr, fused=True), "fused"
        except TypeError:
            pass
    return torch.optim.AdamW(params, lr=lr), "default"



def configure_native_simplefsdp(
    model: torch.nn.Module,
    accelerator: Accelerator,
    replicate_small_param_numel: int = 0,
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
    data_parallel(
        model,
        mesh,
        mode="fully_shard",
        mp_policy=mp_policy,
        shard_dim=0,
        full_dtensor=False,
        replicate_numel_threshold=int(replicate_small_param_numel),
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
        if int(args.simplefsdp_coalesce_bucket_mb) > 0:
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

    # model_weight_path = f"{model_name.split('/')[1]}_cp_layer{args.num_layers}"
    model_weight_path = os.path.join(args.model_path, args.model_name)
    
    if accelerator.is_main_process:
        print(f"model_weight_path: {model_weight_path}")
    if args.load_weights:
        model = AutoModelForCausalLM.from_pretrained(model_weight_path, 
                                                     trust_remote_code=True)
    else:
        config_source = model_weight_path if os.path.exists(model_weight_path) else model_name
        model_config = AutoConfig.from_pretrained(config_source,
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

    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_weight_path, 
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
        
    # dataset = load_dataset('ag_news', split='train[:100%]', download_config=DownloadConfig(disable_tqdm=True))
    dataset = load_dataset('/home/dev/DeepSpeed/examples/deepcompile/datasets', split='train[:100%]', download_config=DownloadConfig(disable_tqdm=True))

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
        "replicate_small_param_numel": 0,
        "replicated_params": 0,
        "replicated_global_numel": 0,
        "sharded_params": 0,
        "dtensor_params": 0,
        "dtensor_local_numel": 0,
        "dtensor_global_numel": 0,
        "coalesce_bucket_mb": 0,
    }
    if is_simplefsdp:
        simplefsdp_stats = configure_native_simplefsdp(
            model,
            accelerator,
            replicate_small_param_numel=args.simplefsdp_replicate_small_param_numel,
        )
        optimizer, optimizer_name = make_adamw_optimizer(
            model.parameters(),
            lr=args.learning_rate,
            fused=args.simplefsdp_enable_fused_optimizer,
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
        simplefsdp_stats["coalesce_bucket_mb"] = (
            int(args.simplefsdp_coalesce_bucket_mb) if simplefsdp_inductor_comm_overlap else 0
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

    fsdp2_prefetch_stats = {"enabled": False, "layers": 0, "forward_distance": 0, "backward_distance": 0}
    if (not is_simplefsdp) and _is_accelerate_fsdp2(accelerator) and not args.disable_fsdp2_prefetch:
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
                f"backward_distance={fsdp2_prefetch_stats['backward_distance']}"
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
    nvidia_smi_sampler = NvidiaSmiMemorySampler(device_index=accelerator.local_process_index)

    # See https://github.com/microsoft/DeepSpeed/issues/6793
    acc_context = nullcontext if (is_deepspeed or is_simplefsdp) else accelerator.accumulate

    stop = False
    with prof_context as prof:
        step_compute_time = 0.0
        for epoch in range(args.num_epochs):
            for step, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

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
                            nvidia_smi_sampler.start()
                            memory_peak_reset_done = True
                    else:
                        step_compute_time += time.time() - micro_start

                if do_profile:
                    prof.step()

                # Only run through step 11 for faster experiments.
                stop = global_step >= 11 * args.gradient_accumulation_steps
                if stop:
                    break
            if stop:
                break

    # Report iteration time as the mean of true times from step 7..11 (5 steps).
    _report_steps = list(range(7, 12))
    report_times = [iter_times_by_step[s] for s in _report_steps if s in iter_times_by_step]
    if not report_times:
        report_times = iter_times

    local_nvidia_smi_peak_bytes = nvidia_smi_sampler.stop()
    nvidia_smi_peak_global_bytes = distributed_max_int(local_nvidia_smi_peak_bytes)

    alloc_bytes = int(torch.cuda.memory_allocated())
    peak_bytes = int(torch.cuda.max_memory_allocated())
    peak_global_bytes = distributed_max_int(peak_bytes)

    if accelerator.is_main_process:
        compile_time_sum = 0
        compile_time = 0
        if args.compile and hasattr(model, "get_compile_time"):
            compile_time = model.get_compile_time()
            compile_time_sum = sum(t for _, _, _, t in compile_time)

        is_deepcompile = is_deepspeed and model._config.compile_config.deepcompile
        alloc_gb = float(alloc_bytes) / (1024**3)
        peak_gb = float(peak_bytes) / (1024**3)
        peak_global_gb = float(peak_global_bytes) / (1024**3)

        predicted_peak = ""
        predicted_total_peak_bytes = 0
        predicted_total_peak_gb = 0.0
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
                predicted_total_peak_bytes = int(
                    meta.get("predicted_total_peak_mem_bytes", meta.get("predicted_peak_mem_bytes", 0)))
                predicted_total_peak_gb = float(
                    meta.get("predicted_total_peak_mem_gb", meta.get("predicted_peak_mem_gb", 0.0)))
                predicted_graph_peak_bytes = int(meta.get("predicted_graph_peak_mem_bytes", 0))
                predicted_graph_peak_gb = float(meta.get("predicted_graph_peak_mem_gb", 0.0))
                predicted_baseline_offset_bytes = int(meta.get("predicted_peak_baseline_offset_bytes", 0))
                predicted_baseline_offset_gb = float(meta.get("predicted_peak_baseline_offset_gb", 0.0))
                if predicted_total_peak_gb > 0.0 or predicted_total_peak_bytes > 0:
                    prediction_error_bytes = int(peak_global_bytes) - int(predicted_total_peak_bytes)
                    prediction_error_gb = float(prediction_error_bytes) / (1024**3)
                    prediction_error_pct = (
                        abs(float(prediction_error_bytes)) / float(peak_global_bytes) * 100.0
                        if int(peak_global_bytes) > 0 else 0.0
                    )
                    predicted_peak = (
                        f" predicted_total_peak_mem: {predicted_total_peak_gb:.2f} GB"
                        f" predicted_total_peak_mem_bytes: {predicted_total_peak_bytes}"
                        f" predicted_graph_peak_mem: {predicted_graph_peak_gb:.2f} GB"
                        f" predicted_graph_peak_mem_bytes: {predicted_graph_peak_bytes}"
                        f" predicted_peak_baseline_offset: {predicted_baseline_offset_gb:.2f} GB"
                        f" predicted_peak_baseline_offset_bytes: {predicted_baseline_offset_bytes}"
                        f" prediction_error: {prediction_error_gb:.2f} GB"
                        f" prediction_error_bytes: {prediction_error_bytes}"
                        f" prediction_abs_error_pct: {prediction_error_pct:.2f}"
                    )
        except Exception:
            predicted_peak = ""

        fsdp2_stats = (
            f" fsdp2_prefetch_enabled: {fsdp2_prefetch_stats['enabled']}"
            f" fsdp2_prefetch_layers: {fsdp2_prefetch_stats['layers']}"
            f" fsdp2_forward_prefetch_distance: {fsdp2_prefetch_stats['forward_distance']}"
            f" fsdp2_backward_prefetch_distance: {fsdp2_prefetch_stats['backward_distance']}"
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
        ) if simplefsdp_stats.get("enabled", False) else ""
        activation_checkpointing_report = bool(
            activation_checkpointing_enabled or simplefsdp_stats.get("selective_checkpoint_layers", 0) > 0
        )
        nvidia_smi_peak_gb = float(nvidia_smi_peak_global_bytes) / (1024**3)
        msg = f"Pred. {predicted_total_peak_gb:.2f} GB Meas. {nvidia_smi_peak_gb:.2f} GB"
        print(msg)

        if args.profile_dir:
            from pathlib import Path
            filepath = Path(args.profile_dir) / f"result.txt"
            with open(filepath, "a") as f:
                f.write(f"{timestamp} {msg}" + "\n")

            if args.compile:
                filepath = Path(args.profile_dir) / f"compile_time.txt"
                with open(filepath, "a") as f:
                    msg =  f"{msg} compile_time={compile_time_sum} {compile_time}"
                    f.write(f"{timestamp} {msg}" + "\n")

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
