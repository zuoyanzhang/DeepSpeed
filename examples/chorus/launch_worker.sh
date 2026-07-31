#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

die() {
    echo "Error: $*" >&2
    exit 2
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^(0|[1-9][0-9]*)$ ]]
}

NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
is_positive_integer "$NUM_NODES" || die "NUM_NODES must be a positive integer"
is_positive_integer "$NGPUS_PER_NODE" || die "NGPUS_PER_NODE must be a positive integer"
NUM_PROCESSES=$((NUM_NODES * NGPUS_PER_NODE))
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN=$PYTHON
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=python3
fi

BACKEND=deepspeed
MODEL=meta-llama/Meta-Llama-3-8B
ZERO_STAGE=3
COMPILE=0
PASSES=none
EAGER=0
DEEPCOMPILE=0
GRADIENT_ACCUMULATION_STEPS=1
ACTIVATION_CHECKPOINTING=0
BATCH_SIZE=1
SEQ_LENGTH=512
DEBUG_LOG=0
SYNC_BEFORE_REDUCE=0
SYNC_AFTER_REDUCE=0
SYNC_BEFORE_ALLGATHER=0
SYNC_AFTER_ALLGATHER=0
DS_OFFLOAD=0
FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-true}
ENV_MACHINE_RANK=${MACHINE_RANK:-}
EXPLICIT_MACHINE_RANK=""
BENCHMARK_ARGS=()

while (($#)); do
    case "$1" in
        --host-ip)
            HOST_IP=$2
            shift 2
            ;;
        --host-port)
            HOST_PORT=$2
            shift 2
            ;;
        --machine-rank)
            EXPLICIT_MACHINE_RANK=$2
            shift 2
            ;;
        --backend)
            BACKEND=$2
            shift 2
            ;;
        --zero-stage)
            ZERO_STAGE=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            BENCHMARK_ARGS+=(--batch_size "$2")
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH=$2
            BENCHMARK_ARGS+=(--seq_length "$2")
            shift 2
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS=$2
            shift 2
            ;;
        --activation-checkpointing)
            ACTIVATION_CHECKPOINTING=1
            BENCHMARK_ARGS+=(--activation_checkpointing)
            shift
            ;;
        --compile)
            COMPILE=1
            BENCHMARK_ARGS+=(--compile)
            shift
            ;;
        --eager)
            EAGER=1
            BENCHMARK_ARGS+=(--backend eager)
            shift
            ;;
        --deepcompile)
            DEEPCOMPILE=1
            shift
            ;;
        --passes)
            PASSES=$2
            BENCHMARK_ARGS+=(--passes "$2")
            shift 2
            ;;
        --fsdp2-fast|--fsdp2-no-reshard-after-forward)
            FSDP2_RESHARD_AFTER_FORWARD=false
            shift
            ;;
        --fsdp2-reshard-after-forward)
            FSDP2_RESHARD_AFTER_FORWARD=$2
            shift 2
            ;;
        --profile)
            BENCHMARK_ARGS+=(--profile)
            shift
            ;;
        --profile-dir)
            BENCHMARK_ARGS+=(--profile_dir "$2")
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --num-layers)
            BENCHMARK_ARGS+=(--num_layers "$2")
            shift 2
            ;;
        --attn-impl)
            BENCHMARK_ARGS+=(--attn_impl "$2")
            shift 2
            ;;
        --eval)
            BENCHMARK_ARGS+=(--eval)
            shift
            ;;
        --debug-log)
            DEBUG_LOG=1
            shift
            ;;
        --sync-before-reduce)
            SYNC_BEFORE_REDUCE=1
            shift
            ;;
        --sync-after-reduce)
            SYNC_AFTER_REDUCE=1
            shift
            ;;
        --sync-before-allgather)
            SYNC_BEFORE_ALLGATHER=1
            shift
            ;;
        --sync-after-allgather)
            SYNC_AFTER_ALLGATHER=1
            shift
            ;;
        --ds-offload)
            DS_OFFLOAD=1
            shift
            ;;
        *)
            BENCHMARK_ARGS+=("$1")
            shift
            ;;
    esac
done

HOST_IP=${HOST_IP:-${MASTER_ADDR:-127.0.0.1}}
HOST_PORT=${HOST_PORT:-${MASTER_PORT:-29500}}
is_positive_integer "$HOST_PORT" || die "rendezvous port must be an integer from 1 to 65535"
((HOST_PORT <= 65535)) || die "rendezvous port must be an integer from 1 to 65535"
if ((NUM_NODES > 1)); then
    case "$HOST_IP" in
        127.*|localhost|localhost.*|::1|0.0.0.0)
            die "multi-node runs require the reachable hostname or IP of machine rank 0"
            ;;
    esac
fi

if [[ -n "$EXPLICIT_MACHINE_RANK" ]]; then
    MACHINE_RANK=$EXPLICIT_MACHINE_RANK
elif [[ -n "${SLURM_NODEID:-}" ]]; then
    MACHINE_RANK=$SLURM_NODEID
elif [[ -n "${SLURM_PROCID:-}" ]]; then
    MACHINE_RANK=$SLURM_PROCID
elif [[ -n "$ENV_MACHINE_RANK" ]]; then
    MACHINE_RANK=$ENV_MACHINE_RANK
else
    MACHINE_RANK=0
fi
is_nonnegative_integer "$MACHINE_RANK" || die "machine rank must be a non-negative integer"
((MACHINE_RANK < NUM_NODES)) || die "machine rank must be smaller than NUM_NODES"

case "$BACKEND" in
    deepspeed|fsdp|fsdp2|simplefsdp|ddp|singlegpu) ;;
    *)
        echo "Error: invalid backend: $BACKEND" >&2
        exit 2
        ;;
esac

"$PYTHON_BIN" - "$BACKEND" <<'PY'
import sys
import torch

backend = sys.argv[1]
try:
    version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
except ValueError as exc:
    raise SystemExit(f"Cannot parse PyTorch version: {torch.__version__}") from exc
supported = version == (2, 7) if backend == "simplefsdp" else (2, 6) <= version <= (2, 7)
if not supported:
    expected = "2.7.x" if backend == "simplefsdp" else "2.6.x or 2.7.x"
    raise SystemExit(f"Chorus backend {backend!r} requires PyTorch {expected}; found {torch.__version__}")
PY

if [[ "$BACKEND" != deepspeed ]]; then
    ZERO_STAGE=3
fi

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

RUN_ID=${CHORUS_RUN_ID:-${SLURM_JOB_ID:-local-$$}}
SAFE_RUN_ID=${RUN_ID//[^a-zA-Z0-9_.-]/_}
SAFE_RUN_ID=${SAFE_RUN_ID:0:64}
[[ -n "$SAFE_RUN_ID" ]] || SAFE_RUN_ID=run
if [[ -z "${DS_DEEPCOMPILE_USE_GLOBAL_CACHE:-}" ]]; then
    CACHE_BASE=${CHORUS_RUNTIME_DIR:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}
    CACHE_DIR=${CACHE_BASE%/}/chorus-cache-${UID:-user}-${SAFE_RUN_ID}-rank-${MACHINE_RANK}
    export TRITON_CACHE_DIR=$CACHE_DIR/triton
    export TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR/torchinductor
    mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
fi

RUNTIME_BASE=${CHORUS_RUNTIME_DIR:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/chorus-runtime-${UID:-user}-$SAFE_RUN_ID-$HOST_PORT}
RUNTIME_CONFIG_DIR=$RUNTIME_BASE/rank-$MACHINE_RANK
mkdir -p "$RUNTIME_CONFIG_DIR"
ACCELERATE_CONFIG=$RUNTIME_CONFIG_DIR/accelerate.yaml
DEEPSPEED_CONFIG=$RUNTIME_CONFIG_DIR/deepspeed.json

CONFIG_TEMPLATE=$SCRIPT_DIR/configs/ds_config.yaml.template
case "$BACKEND" in
    fsdp) CONFIG_TEMPLATE=$SCRIPT_DIR/configs/fsdp_config.yaml.template ;;
    fsdp2) CONFIG_TEMPLATE=$SCRIPT_DIR/configs/fsdp2_config.yaml.template ;;
    simplefsdp) CONFIG_TEMPLATE=$SCRIPT_DIR/configs/simplefsdp_config.yaml.template ;;
    ddp) CONFIG_TEMPLATE=$SCRIPT_DIR/configs/ddp_config.yaml.template ;;
    singlegpu) CONFIG_TEMPLATE=$SCRIPT_DIR/configs/singlegpu_config.yaml.template ;;
esac

GENERATOR_ARGS=(
    --machine_rank "$MACHINE_RANK"
    --num_machines "$NUM_NODES"
    --num_processes "$NUM_PROCESSES"
    --zero_stage "$ZERO_STAGE"
    --model "$MODEL"
    --fsdp2_reshard_after_forward "$FSDP2_RESHARD_AFTER_FORWARD"
    --deepspeed_config_file "$DEEPSPEED_CONFIG"
    --template_file "$CONFIG_TEMPLATE"
    --output_file "$ACCELERATE_CONFIG"
)
"$PYTHON_BIN" "$SCRIPT_DIR/generate_config.py" "${GENERATOR_ARGS[@]}"

if [[ "$BACKEND" == deepspeed ]]; then
    DEEPSPEED_GENERATOR_ARGS=(
        --machine_rank "$MACHINE_RANK"
        --num_machines "$NUM_NODES"
        --num_processes "$NUM_PROCESSES"
        --zero_stage "$ZERO_STAGE"
        --model "$MODEL"
        --fsdp2_reshard_after_forward "$FSDP2_RESHARD_AFTER_FORWARD"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --deepspeed_config_file "$DEEPSPEED_CONFIG"
        --template_file "$SCRIPT_DIR/configs/ds_config.json.template"
        --output_file "$DEEPSPEED_CONFIG"
    )
    ((DEEPCOMPILE)) && DEEPSPEED_GENERATOR_ARGS+=(--deepcompile)
    ((DEBUG_LOG)) && DEEPSPEED_GENERATOR_ARGS+=(--debug_log)
    ((SYNC_BEFORE_REDUCE)) && DEEPSPEED_GENERATOR_ARGS+=(--sync_before_reduce)
    ((SYNC_AFTER_REDUCE)) && DEEPSPEED_GENERATOR_ARGS+=(--sync_after_reduce)
    ((SYNC_BEFORE_ALLGATHER)) && DEEPSPEED_GENERATOR_ARGS+=(--sync_before_allgather)
    ((SYNC_AFTER_ALLGATHER)) && DEEPSPEED_GENERATOR_ARGS+=(--sync_after_allgather)
    ((DS_OFFLOAD)) && DEEPSPEED_GENERATOR_ARGS+=(--ds_offload)
    "$PYTHON_BIN" "$SCRIPT_DIR/generate_config.py" "${DEEPSPEED_GENERATOR_ARGS[@]}"
fi

SAFE_PASSES=${PASSES//,/_}
SAFE_MODEL=${MODEL##*/}
LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs}
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/chorus_${SAFE_RUN_ID}_n${MACHINE_RANK}_${SAFE_MODEL}_${BACKEND}_np${NUM_PROCESSES}z${ZERO_STAGE}c${COMPILE}dc${DEEPCOMPILE}e${EAGER}o${DS_OFFLOAD}b${BATCH_SIZE}seq${SEQ_LENGTH}g${GRADIENT_ACCUMULATION_STEPS}a${ACTIVATION_CHECKPOINTING}p${SAFE_PASSES}.log

echo "Chorus backend: $BACKEND"
echo "Machine rank: $MACHINE_RANK/$NUM_NODES"
echo "Processes: $NUM_PROCESSES ($NGPUS_PER_NODE GPUs per node)"
echo "Model: $MODEL"
echo "Runtime config: $RUNTIME_CONFIG_DIR"
echo "Log: $LOG_FILE"
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

LAUNCH_CMD=(
    "$PYTHON_BIN" -m accelerate.commands.launch
    --main_process_ip "$HOST_IP"
    --main_process_port "$HOST_PORT"
    --num_machines "$NUM_NODES"
    --num_processes "$NUM_PROCESSES"
    --machine_rank "$MACHINE_RANK"
    --config_file "$ACCELERATE_CONFIG"
    "$SCRIPT_DIR/benchmark.py"
    --model_name "$MODEL"
    --zero_stage "$ZERO_STAGE"
    --distributed_backend "$BACKEND"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
)
if ((${#BENCHMARK_ARGS[@]})); then
    LAUNCH_CMD+=("${BENCHMARK_ARGS[@]}")
fi

"${LAUNCH_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
