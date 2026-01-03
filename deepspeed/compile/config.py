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

    global_layer_scheduler_lookahead_blocks: int = 4
    """ Prefetch lookahead window in layer blocks """

    global_layer_scheduler_fuse_max_bytes: int = 256 * 1024 * 1024
    """ Upper bound for one fused prefetch_params_fused payload size """

    global_layer_scheduler_fuse_deadline_window_blocks: int = 1
    """ Only fuse tasks whose deadlines are within this many blocks of the earliest deadline in the fused group """

    global_layer_scheduler_fuse_factor: float = 0.8
    """ Fusion discount factor used in the scheduler comm-cost model (<=1.0) """

    global_layer_scheduler_include_unmapped_params: bool = True
    """ If True, assign ds_ids not matched by layer regex to the nearest layer by wait_allgather position """

    global_layer_scheduler_set_persistent: bool = False
    """ If True, call DeepCompile set_persistent(ds_id) for selected ds_ids to reuse gathered buffers across steps """

    global_layer_scheduler_inner_refine_max_iters: int = 128
    """ Max iterations for inner (start_k) refinement local search; set 0 to disable """

    global_layer_scheduler_inner_refine_min_gain_ms: float = 0.05
    """ Minimum predicted step-time improvement (ms) to accept a refinement move """

    global_layer_scheduler_mem_margin: float = 0.1
    """ Memory safety margin (fraction of total memory) when computing unsharded buffer budget """

    global_layer_scheduler_dump_schedule: bool = False
    """ Dump the planned schedule JSON on rank0 (debug) """

    global_layer_scheduler_dump_dir: str = ""
    """ Directory to dump schedule JSON (defaults to CWD if empty) """

    global_layer_scheduler_safety_margin_bytes: int = 512 * 1024 * 1024
    """ Extra safety margin (bytes) subtracted from per-block memory cap """

    global_layer_scheduler_outer_keep_max_iters: int = 32
    """ Max iterations for greedy keep-unshard selection """

    global_layer_scheduler_outer_keep_min_gain_ms: float = 0.5
    """ Minimum predicted step-time improvement (ms) to accept a keep layer """

    global_layer_scheduler_outer_keep_top_n: int = 3
    """ Try top-N keep candidates per iteration """

    global_layer_scheduler_max_tasks_per_block: int = 16
    """ Upper bound on how many layer-tasks can be launched at the same block """

    global_layer_scheduler_rewrite_comm_ops: bool = True
    """ Rewrite/move allgather/wait nodes to layer anchors (can reduce intra-layer overlap; default off) """

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

    keep_int_input_tensors: bool = True
    """ Keep real values for int tensors in InputStorage instead of using dummy values """

    keep_all_input_tensors: bool = False
    """ Keep real values for all input tensors in InputStorage instead of using dummy values """
