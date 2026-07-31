#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Run Chorus on a single multi-GPU server without Slurm.

Usage:
  run_server.sh [options] [-- extra benchmark arguments]

Options:
  --backend NAME          deepspeed or simplefsdp (default: deepspeed)
  --model NAME            Hugging Face model identifier (default: mistralai/Mistral-7B-Instruct-v0.3)
  --model-path DIR        Exact local model directory; overrides model download
  --dataset-path PATH     Local JSON/JSONL dataset file or directory
  --gpus N                Number of visible GPUs to use (default: all visible GPUs)
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
  --dry-run               Print the exact launch command without writing files or starting training
  -h, --help              Show this help

DeepSpeed expands to the DeepCompile global_layer_scheduler implementation.
SimpleFSDP expands to the compiled-autograd SimpleFSDP Chorus implementation.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 2
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

BACKEND=${BACKEND:-deepspeed}
MODEL_NAME=${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}
MODEL_DIR=${MODEL_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
GPU_COUNT=${NGPUS_PER_NODE:-}
BATCH_SIZE=${BATCH_SIZE:-2}
SEQ_LENGTH=${SEQ_LENGTH:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MASTER_PORT=${MASTER_PORT:-29500}
PROFILE_DIR=${PROFILE_DIR:-profiles}
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
        --gpus)
            (($# >= 2)) || die "--gpus requires a value"
            GPU_COUNT=$2
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

case "$BACKEND" in
    deepspeed|simplefsdp) ;;
    *) die "--backend must be deepspeed or simplefsdp" ;;
esac

is_positive_integer "$BATCH_SIZE" || die "--batch-size must be a positive integer"
is_positive_integer "$SEQ_LENGTH" || die "--seq-length must be a positive integer"
is_positive_integer "$GRADIENT_ACCUMULATION_STEPS" || die "--gradient-accumulation-steps must be a positive integer"
is_positive_integer "$MASTER_PORT" || die "--master-port must be an integer from 1 to 65535"
((MASTER_PORT <= 65535)) || die "--master-port must be an integer from 1 to 65535"

if [[ "$BACKEND" == simplefsdp && "$GRADIENT_ACCUMULATION_STEPS" != 1 ]]; then
    die "the public SimpleFSDP Chorus launcher currently requires --gradient-accumulation-steps 1"
fi

if ((${#EXTRA_ARGS[@]})); then
    for arg in "${EXTRA_ARGS[@]}"; do
        option_name=${arg%%=*}
        case "$option_name" in
            --backend|--distributed-backend|--distributed_backend|--compile|--deepcompile|--passes|\
            --simplefsdp-enable-chorus|--simplefsdp_enable_chorus|\
            --simplefsdp-enable-compiled-autograd|--simplefsdp_enable_compiled_autograd|\
            --model-name|--model_name|--model-dir|--model_dir|--model-path|--model_path|\
            --dataset-path|--dataset_path|--batch-size|--batch_size|--seq-length|--seq_length|\
            --gradient-accumulation-steps|--gradient_accumulation_steps|\
            --activation-checkpointing|--activation_checkpointing|--load-weights|--load_weights|\
            --profile|--profile-dir|--profile_dir|--zero-stage|--zero_stage)
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

if [[ -n "$MODEL_DIR" && ! -f "$MODEL_DIR/config.json" ]]; then
    die "--model-path must be a model directory containing config.json: $MODEL_DIR"
fi
if [[ -n "$DATASET_PATH" && ! -e "$DATASET_PATH" ]]; then
    die "--dataset-path does not exist: $DATASET_PATH"
fi

ARGS=(
    --machine-rank 0
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
    printf 'NUM_NODES=1 NGPUS_PER_NODE=%q MASTER_ADDR=127.0.0.1 MASTER_PORT=%q ' "$GPU_COUNT" "$MASTER_PORT"
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

export NUM_NODES=1
export NGPUS_PER_NODE=$GPU_COUNT
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
export MACHINE_RANK=0

CHORUS_RUNTIME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/chorus-runtime-${UID:-user}.XXXXXX")
readonly CHORUS_RUNTIME_DIR
export CHORUS_RUNTIME_DIR
trap 'rm -rf -- "${CHORUS_RUNTIME_DIR:?}"' EXIT

"${COMMAND[@]}"
