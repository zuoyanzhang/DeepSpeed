#!/usr/bin/env bash

set -Eeuo pipefail

INVOCATION_DIR=$PWD
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Run Chorus on one or more multi-GPU servers without Slurm.

Usage:
  run_server.sh [options] [-- extra benchmark arguments]

Options:
  --backend NAME          deepspeed or simplefsdp (default: deepspeed)
  --model NAME            Hugging Face model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)
  --model-path DIR        Exact local model directory; overrides model download
  --dataset-path PATH     Local JSON/JSONL dataset file or directory
  --gpus N                GPUs per server (default: all visible GPUs on one server;
                          required for multi-node runs)
  --nodes N               Number of servers participating in the run (default: 1)
  --node-rank R           Zero-based rank of this server (default: 0 on one node;
                          required for multi-node runs)
  --master-addr HOST      Reachable hostname or IP of node rank 0
  --batch-size N          Per-process batch size (default: 2)
  --seq-length N          Sequence length (default: 1024)
  --gradient-accumulation-steps N
                          Gradient accumulation steps (default: 1)
  --master-port PORT      Rendezvous port (default: 29500)
  --profile               Enable the PyTorch profiler
  --profile-dir DIR       Profiler output directory (default: profiles)
  --no-activation-checkpointing
                          Disable activation checkpointing
  --random-init           Initialize from model config instead of loading weights
  --dry-run               Print resolved distributed settings and the per-node worker command
  -h, --help              Show this help

DeepSpeed expands to the DeepCompile global_layer_scheduler implementation.
SimpleFSDP expands to the compiled-autograd SimpleFSDP Chorus implementation.

For a multi-node run, execute this script once on every server with identical
arguments except for --node-rank. All servers must use the same --master-addr,
--master-port, --nodes, and --gpus values.
EOF
}

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

BACKEND=${BACKEND:-deepspeed}
MODEL_NAME_EXPLICIT=0
[[ -n "${MODEL_NAME:-}" ]] && MODEL_NAME_EXPLICIT=1
MODEL_NAME=${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}
MODEL_DIR=${MODEL_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
GPU_COUNT=${NGPUS_PER_NODE:-}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-${MACHINE_RANK:-}}
MASTER_ADDR=${MASTER_ADDR:-}
BATCH_SIZE=${BATCH_SIZE:-2}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
PROFILE_DIR=${PROFILE_DIR:-$SCRIPT_DIR/profiles}
PROFILE=0
ACTIVATION_CHECKPOINTING=1
LOAD_WEIGHTS=1
DRY_RUN=0
EXTRA_ARGS=()

while (($#)); do
    case "$1" in
        --backend)
            (($# >= 2)) || die "--backend requires a value"
            BACKEND=$2
            shift 2
            ;;
        --model)
            (($# >= 2)) || die "--model requires a value"
            MODEL_NAME=$2
            MODEL_NAME_EXPLICIT=1
            shift 2
            ;;
        --model-path|--model-dir)
            (($# >= 2)) || die "$1 requires a value"
            MODEL_DIR=$2
            shift 2
            ;;
        --dataset-path)
            (($# >= 2)) || die "--dataset-path requires a value"
            DATASET_PATH=$2
            shift 2
            ;;
        --gpus|--gpus-per-node)
            (($# >= 2)) || die "--gpus requires a value"
            GPU_COUNT=$2
            shift 2
            ;;
        --nodes|--num-nodes)
            (($# >= 2)) || die "$1 requires a value"
            NUM_NODES=$2
            shift 2
            ;;
        --node-rank|--machine-rank)
            (($# >= 2)) || die "$1 requires a value"
            NODE_RANK=$2
            shift 2
            ;;
        --master-addr)
            (($# >= 2)) || die "--master-addr requires a value"
            MASTER_ADDR=$2
            shift 2
            ;;
        --batch-size)
            (($# >= 2)) || die "--batch-size requires a value"
            BATCH_SIZE=$2
            shift 2
            ;;
        --seq-length)
            (($# >= 2)) || die "--seq-length requires a value"
            SEQ_LENGTH=$2
            shift 2
            ;;
        --gradient-accumulation-steps)
            (($# >= 2)) || die "--gradient-accumulation-steps requires a value"
            GRADIENT_ACCUMULATION_STEPS=$2
            shift 2
            ;;
        --master-port)
            (($# >= 2)) || die "--master-port requires a value"
            MASTER_PORT=$2
            shift 2
            ;;
        --profile)
            PROFILE=1
            shift
            ;;
        --profile-dir)
            (($# >= 2)) || die "--profile-dir requires a value"
            PROFILE=1
            PROFILE_DIR=$2
            shift 2
            ;;
        --no-activation-checkpointing)
            ACTIVATION_CHECKPOINTING=0
            shift
            ;;
        --random-init)
            LOAD_WEIGHTS=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            die "unknown option: $1 (pass benchmark-only arguments after --)"
            ;;
    esac
done

[[ "$PROFILE_DIR" == /* ]] || PROFILE_DIR="$INVOCATION_DIR/$PROFILE_DIR"
if [[ -n "${LOG_DIR:-}" && "$LOG_DIR" != /* ]]; then
    LOG_DIR="$INVOCATION_DIR/$LOG_DIR"
    export LOG_DIR
fi

case "$BACKEND" in
    deepspeed|simplefsdp) ;;
    *) die "--backend must be deepspeed or simplefsdp" ;;
esac

is_positive_integer "$BATCH_SIZE" || die "--batch-size must be a positive integer"
is_positive_integer "$SEQ_LENGTH" || die "--seq-length must be a positive integer"
is_positive_integer "$GRADIENT_ACCUMULATION_STEPS" || die "--gradient-accumulation-steps must be a positive integer"
is_positive_integer "$NUM_NODES" || die "--nodes must be a positive integer"
if [[ -z "$NODE_RANK" ]]; then
    if ((NUM_NODES == 1)); then
        NODE_RANK=0
    else
        die "--node-rank is required when --nodes is greater than 1"
    fi
fi
is_nonnegative_integer "$NODE_RANK" || die "--node-rank must be a non-negative integer"
((NODE_RANK < NUM_NODES)) || die "--node-rank must be smaller than --nodes"
is_positive_integer "$MASTER_PORT" || die "--master-port must be an integer from 1 to 65535"
((MASTER_PORT <= 65535)) || die "--master-port must be an integer from 1 to 65535"

if [[ -z "$MASTER_ADDR" ]]; then
    if ((NUM_NODES == 1)); then
        MASTER_ADDR=127.0.0.1
    else
        die "--master-addr is required when --nodes is greater than 1"
    fi
fi
if ((NUM_NODES > 1)); then
    [[ -n "$GPU_COUNT" ]] || die "--gpus is required when --nodes is greater than 1"
    case "$MASTER_ADDR" in
        127.*|localhost|localhost.*|::1|0.0.0.0)
            die "--master-addr must be the reachable hostname or IP of node rank 0 for a multi-node run"
            ;;
    esac
fi

if [[ "$BACKEND" == simplefsdp && "$GRADIENT_ACCUMULATION_STEPS" != 1 ]]; then
    die "the public SimpleFSDP Chorus launcher currently requires --gradient-accumulation-steps 1"
fi

if ((${#EXTRA_ARGS[@]})); then
    for arg in "${EXTRA_ARGS[@]}"; do
        option_name=${arg%%=*}
        case "$option_name" in
            --backend|--distributed-backend|--distributed_backend|--compile|--deepcompile|--passes|--eager|\
            --simplefsdp-enable-chorus|--simplefsdp_enable_chorus|\
            --simplefsdp-enable-compiled-autograd|--simplefsdp_enable_compiled_autograd|\
            --model|--model-name|--model_name|--model-dir|--model_dir|--model-path|--model_path|\
            --dataset-path|--dataset_path|--batch-size|--batch_size|--seq-length|--seq_length|\
            --gradient-accumulation-steps|--gradient_accumulation_steps|\
            --activation-checkpointing|--activation_checkpointing|--load-weights|--load_weights|\
            --profile|--profile-dir|--profile_dir|--zero-stage|--zero_stage|\
            --gpus|--gpus-per-node|--nodes|--num-nodes|--node-rank|--machine-rank|\
            --master-addr|--master-port|--host-ip|--host-port|\
            --no-activation-checkpointing|--random-init|--dry-run)
                die "reserved Chorus option cannot be overridden after --: $arg"
                ;;
        esac
    done
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN=$PYTHON
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=python3
fi
if [[ -z "$GPU_COUNT" ]]; then
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        GPU_COUNT=$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)
    fi
    if ! is_positive_integer "${GPU_COUNT:-}"; then
        if ((DRY_RUN)); then
            GPU_COUNT=1
        else
            die "no CUDA GPU is visible to PyTorch; set CUDA_VISIBLE_DEVICES or --gpus"
        fi
    fi
fi
is_positive_integer "$GPU_COUNT" || die "--gpus must be a positive integer"

if [[ -n "$MODEL_DIR" ]]; then
    [[ -f "$MODEL_DIR/config.json" ]] || die "--model-path must be a model directory containing config.json: $MODEL_DIR"
    MODEL_DIR=$(cd -- "$MODEL_DIR" && pwd -P)
    if ((MODEL_NAME_EXPLICIT == 0)); then
        MODEL_NAME=${MODEL_DIR##*/}
        [[ -n "$MODEL_NAME" ]] || MODEL_NAME=local-model
    fi
fi
if [[ -n "$DATASET_PATH" ]]; then
    [[ -e "$DATASET_PATH" ]] || die "--dataset-path does not exist: $DATASET_PATH"
    if [[ -d "$DATASET_PATH" ]]; then
        DATASET_PATH=$(cd -- "$DATASET_PATH" && pwd -P)
    else
        DATASET_DIR=$(cd -- "$(dirname -- "$DATASET_PATH")" && pwd -P)
        DATASET_PATH="$DATASET_DIR/$(basename -- "$DATASET_PATH")"
    fi
fi

ARGS=(
    --machine-rank "$NODE_RANK"
    --backend "$BACKEND"
    --model "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --seq-length "$SEQ_LENGTH"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --compile
)

if [[ "$BACKEND" == deepspeed ]]; then
    ARGS+=(--deepcompile --passes global_layer_scheduler)
else
    ARGS+=(--simplefsdp-enable-compiled-autograd --simplefsdp-enable-chorus)
fi

((ACTIVATION_CHECKPOINTING)) && ARGS+=(--activation-checkpointing)
((LOAD_WEIGHTS)) && ARGS+=(--load_weights)
if [[ -n "$MODEL_DIR" ]]; then
    ARGS+=(--model-dir "$MODEL_DIR")
fi
if [[ -n "$DATASET_PATH" ]]; then
    ARGS+=(--dataset-path "$DATASET_PATH")
fi
if ((PROFILE)); then
    ARGS+=(--profile --profile-dir "$PROFILE_DIR")
fi
if ((${#EXTRA_ARGS[@]})); then
    ARGS+=("${EXTRA_ARGS[@]}")
fi

COMMAND=(bash "$SCRIPT_DIR/launch_worker.sh" "${ARGS[@]}")

if ((DRY_RUN)); then
    WORLD_SIZE=$((NUM_NODES * GPU_COUNT))
    printf 'NUM_NODES=%q NGPUS_PER_NODE=%q WORLD_SIZE=%q MACHINE_RANK=%q MASTER_ADDR=%q MASTER_PORT=%q ' \
        "$NUM_NODES" "$GPU_COUNT" "$WORLD_SIZE" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT"
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

"$PYTHON_BIN" - "$BACKEND" <<'PY'
import importlib
import sys
import torch

for module in ("accelerate", "datasets", "jinja2", "scipy", "transformers"):
    importlib.import_module(module)
backend = sys.argv[1]
version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
supported = version == (2, 7) if backend == "simplefsdp" else (2, 6) <= version <= (2, 7)
if not supported:
    expected = "2.7.x" if backend == "simplefsdp" else "2.6.x or 2.7.x"
    raise SystemExit(f"Chorus backend {backend!r} requires PyTorch {expected}; found {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the current Chorus configurations require a GPU with bfloat16 support")
PY

VISIBLE_GPUS=$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')
((GPU_COUNT <= VISIBLE_GPUS)) || die "requested $GPU_COUNT GPUs, but PyTorch sees only $VISIBLE_GPUS"

export NUM_NODES
export NGPUS_PER_NODE=$GPU_COUNT
export MASTER_ADDR
export MASTER_PORT
export MACHINE_RANK=$NODE_RANK
if [[ -z "${CHORUS_RUN_ID:-}" ]]; then
    if ((NUM_NODES > 1)); then
        RENDEZVOUS_KEY=$(printf '%s' "$MASTER_ADDR:$MASTER_PORT" | cksum)
        RENDEZVOUS_KEY=${RENDEZVOUS_KEY%% *}
        CHORUS_RUN_ID="multi-${RENDEZVOUS_KEY}-${MASTER_PORT}"
    else
        CHORUS_RUN_ID="local-$(date '+%Y%m%d-%H%M%S')-$$"
    fi
fi
export CHORUS_RUN_ID

CHORUS_RUNTIME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/chorus-runtime-${UID:-user}.XXXXXX")
CHORUS_RUNTIME_DIR=$(cd -- "$CHORUS_RUNTIME_DIR" && pwd -P)
readonly CHORUS_RUNTIME_DIR
export CHORUS_RUNTIME_DIR
trap 'rm -rf -- "${CHORUS_RUNTIME_DIR:?}"' EXIT

"${COMMAND[@]}"
