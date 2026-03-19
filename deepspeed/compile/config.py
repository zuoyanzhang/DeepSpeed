# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigModel


class CompileConfig(DeepSpeedConfigModel):
    """ Configure compile settings """

    deepcompile: bool = False
    """ Turn on/off the DeepCompile mode """

    global_layer_scheduler: bool = False
    """ Enable DeepCompile global layer scheduler (keep-unshard + prefetch/fusion with rank0 broadcast) """

    global_layer_scheduler_layer_regexes: list = [
        r"\.layers\.(\d+)\.",
        r"\.model\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.transformer\.h\.(\d+)\.",
        r"\.decoder\.layers\.(\d+)\.",
    ]
    """ Regex patterns (first match wins) used to map parameter names to transformer layer indices """

    global_layer_scheduler_fuse_max_bytes: int = 256 * 1024 * 1024
    """ Upper bound for one fused prefetch_params_fused payload size """

    global_layer_scheduler_fuse_deadline_window_blocks: int = 1
    """ Only fuse tasks whose deadlines are within this many blocks of the earliest deadline in the fused group """

    global_layer_scheduler_fuse_factor: float = 0.8
    """ Fusion discount factor used in the scheduler comm-cost model (<=1.0) """

    global_layer_scheduler_inner_refine_max_iters: int = 512
    """ Max iterations for inner (start_k) refinement local search; set 0 to disable """

    global_layer_scheduler_inner_refine_min_gain_ms: float = 0.01
    """ Minimum predicted step-time improvement (ms) to accept a refinement move """

    global_layer_scheduler_dump_schedule: bool = False
    """ Dump the planned schedule JSON on rank0 (debug) """

    global_layer_scheduler_dump_dir: str = ""
    """ Directory to dump schedule JSON (defaults to CWD if empty) """

    selective_activation_recompute: bool = False
    """ Enable DeepCompile selective activation recompute planning """

    selective_activation_recompute_layer_regexes: list = [
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
    ]
    """ Regex patterns used to map parameter names to transformer layer indices for selective recompute """

    selective_activation_recompute_module_regexes: list = [
        r"(?:^|\.)layers\.(\d+)$",
        r"(?:^|\.)h\.(\d+)$",
    ]
    """ Regex patterns used to detect checkpoint-capable layer modules for selective recompute """

    selective_activation_recompute_dump_plan: bool = False
    """ Dump the selective recompute plan JSON on rank0 """

    selective_activation_recompute_dump_dir: str = ""
    """ Directory to dump selective recompute plan JSON (defaults to CWD if empty) """

    selective_activation_recompute_pressure_ratio: float = 0.9
    """ Safety ratio (0,1] for planning memory budget. Effective budget ~= total_mem * ratio. """

    global_layer_scheduler_milp_time_limit_s: float = 2.0
    """ Time limit (seconds) for the global layer scheduler MILP solver """

    global_layer_scheduler_milp_node_limit: int = 0
    """ Node limit for the MILP solver (0 means no limit) """

    global_layer_scheduler_milp_rel_gap: float = 0.01
    """ Relative MIP gap termination criterion for the MILP solver """

    global_layer_scheduler_milp_presolve: bool = True
    """ Enable/disable MILP presolve """

    free_activation: bool = False
    """ Turn on/off the free activation mode """

    free_activation_threshold: int = 10 * 1024 * 1024
    """ In free activation mode, activations no less than this threshold (in byte) are eagerly freed """

    offload_activation: bool = False
    """ Turn on/off the activation offloading """

    offload_opt_states: bool = False
    """ Turn on/off the optimizer states offloading """

    double_buffer: bool = True
    """ Turn on/off the double buffering """

    symmetric_memory: bool = False
    """ Turn on/off the symmetric memory """

    debug_log: bool = False
    """ Turn on/off the graph dumping """

    offload_parameters: bool = False
    """ Turn on/off the parameter offloading """

    sync_before_reduce: bool = False
    """ Turn on/off the sync before reduce """

    sync_after_reduce: bool = False
    """ Turn on/off the sync after reduce """

    sync_before_allgather: bool = False
    """ Turn on/off the sync before allgather """

    sync_after_allgather: bool = False
    """ Turn on/off the sync after allgather """

    separate_allgather_communicator: bool = True
    """ If True, use a dedicated NCCL communicator for allgather ops to improve overlap with reduce ops """

    keep_int_input_tensors: bool = True
    """ Keep real values for int tensors in InputStorage instead of using dummy values """

    keep_all_input_tensors: bool = False
    """ Keep real values for all input tensors in InputStorage instead of using dummy values """
