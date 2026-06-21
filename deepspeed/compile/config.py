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

    global_layer_scheduler_topology_policy: str = "auto"
    """ Chorus topology policy: auto, generic, low_comm, or nvlink. """

    global_layer_scheduler_low_comm_effective_bw_gib_per_s: float = 120.0
    """ Effective all-gather bandwidth threshold for auto low-communication backend detection. """

    global_layer_scheduler_low_comm_pressure_ratio: float = 0.35
    """ Param-communication/compute ratio threshold for auto low-communication backend detection. """

    global_layer_scheduler_low_comm_keep_opportunity_factor: float = 0.5
    """ Opportunity-cost weight for long-lived keep decisions on low-communication backends. """

    global_layer_scheduler_low_comm_keep_cap_fraction: float = 0.10
    """ Max keep-layer fraction when low-communication backend persistent selection is starved. """

    global_layer_scheduler_low_comm_persistent_budget_mode: str = "selective"
    """ Persistent budget mode for low-communication backends: conservative, selective, or max. """

    global_layer_scheduler_low_comm_persistent_usable_fraction: float = 0.90
    """ Usable memory fraction for selective-style persistent budgeting on low-communication backends. """

    global_layer_scheduler_low_comm_persistent_starvation_fraction: float = 0.75
    """ Rebalance keep layers when persistent budget covers less than this fraction of candidates. """

    global_layer_scheduler_low_comm_graph_rewrite_mode: str = "local_prefetch"
    """ Graph rewrite mode on low-communication backends: local_prefetch or global. """

    global_layer_scheduler_low_comm_prefetch_fuse_slack: float = 1.9
    """ Aggressive local-prefetch fusion slack for low-communication backends. """

    global_layer_scheduler_low_comm_prefetch_fuse_max_bytes: int = 1850000000
    """ Max fused local-prefetch payload size for low-communication backends. """

    global_layer_scheduler_low_comm_prefetch_buffer_max_bytes: int = 5300000000
    """ Max buffered local-prefetch bytes for low-communication backends. """

    global_layer_scheduler_low_comm_elide_persistent_releases: bool = True
    """ Remove redundant release_param ops for persistent parameters on low-communication backends. """

    global_layer_scheduler_low_comm_elide_persistent_waits: bool = True
    """ Remove redundant wait_allgather ops for persistent parameters on low-communication backends. """

    global_layer_scheduler_low_comm_elide_persistent_prefetches: bool = True
    """ Remove persistent ds_ids from fused prefetch ops on low-communication backends. """

    global_layer_scheduler_low_comm_persistent_value_mode: str = "event_density"
    """ Persistent objective on low-communication backends: comm_state or event_density. """

    global_layer_scheduler_low_comm_comm_value_weight: float = 0.25
    """ Weight applied to all-gather latency value in event-density persistent mode. """

    global_layer_scheduler_low_comm_state_op_ms: float = 0.02
    """ Estimated per-op persistent state overhead on low-communication backends. """

    global_layer_scheduler_low_comm_recompute_relief: bool = False
    """ Let Chorus use low-communication memory slack to disable selected activation checkpointing layers. """

    global_layer_scheduler_low_comm_recompute_pressure_ratio: float = 0.72
    """ Memory pressure ratio for Chorus recompute-relief planning on low-communication backends. """

    global_layer_scheduler_low_comm_post_step_refresh: bool = False
    """ Experimentally refresh Chorus persistent parameters after optimizer.step() on a side stream. """

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
