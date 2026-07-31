#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
INVOCATION_DIR=$PWD
cd "$SCRIPT_DIR"

BACKEND=${BACKEND:-deepspeed}
MODEL_NAME_EXPLICIT=0
[[ -n "${MODEL_NAME:-}" ]] && MODEL_NAME_EXPLICIT=1
MODEL_NAME=${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}
MODEL_PATH=${MODEL_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
PROFILE_DIR=${PROFILE_DIR:-$SCRIPT_DIR/profiles}
BASE_MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZES=${BATCH_SIZES:-2}
SEQ_LENGTHS=${SEQ_LENGTHS:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}
LOAD_WEIGHTS=${LOAD_WEIGHTS:-1}
PROFILE=${PROFILE:-0}
EXTRA_ARGS=("$@")

[[ "$PROFILE_DIR" == /* ]] || PROFILE_DIR="$INVOCATION_DIR/$PROFILE_DIR"
if [[ -n "${LOG_DIR:-}" && "$LOG_DIR" != /* ]]; then
    LOG_DIR="$INVOCATION_DIR/$LOG_DIR"
    export LOG_DIR
fi
if [[ -n "${SRUN_LOG_FILE:-}" && "$SRUN_LOG_FILE" != /* ]]; then
    SRUN_LOG_FILE="$INVOCATION_DIR/$SRUN_LOG_FILE"
    export SRUN_LOG_FILE
fi

case "$BACKEND" in
    deepspeed|simplefsdp) ;;
    *)
        echo "Error: BACKEND must be deepspeed or simplefsdp" >&2
        exit 2
        ;;
esac

if ! [[ "$BASE_MASTER_PORT" =~ ^[1-9][0-9]*$ ]] || ((BASE_MASTER_PORT > 65535)); then
    echo "Error: MASTER_PORT must be an integer from 1 to 65535" >&2
    exit 2
fi

if [[ -n "$MODEL_PATH" ]]; then
    [[ "$MODEL_PATH" == /* ]] || MODEL_PATH="$INVOCATION_DIR/$MODEL_PATH"
    [[ -f "$MODEL_PATH/config.json" ]] || {
        echo "Error: MODEL_PATH must be a model directory containing config.json: $MODEL_PATH" >&2
        exit 2
    }
    MODEL_PATH=$(cd -- "$MODEL_PATH" && pwd -P)
    if ((MODEL_NAME_EXPLICIT == 0)); then
        MODEL_NAME=${MODEL_PATH##*/}
        [[ -n "$MODEL_NAME" ]] || MODEL_NAME=local-model
    fi
fi
if [[ -n "$DATASET_PATH" ]]; then
    [[ "$DATASET_PATH" == /* ]] || DATASET_PATH="$INVOCATION_DIR/$DATASET_PATH"
    [[ -e "$DATASET_PATH" ]] || {
        echo "Error: DATASET_PATH does not exist: $DATASET_PATH" >&2
        exit 2
    }
    if [[ -d "$DATASET_PATH" ]]; then
        DATASET_PATH=$(cd -- "$DATASET_PATH" && pwd -P)
    else
        DATASET_DIR=$(cd -- "$(dirname -- "$DATASET_PATH")" && pwd -P)
        DATASET_PATH="$DATASET_DIR/$(basename -- "$DATASET_PATH")"
    fi
fi

if [[ "$BACKEND" == simplefsdp && "$GRADIENT_ACCUMULATION_STEPS" != 1 ]]; then
    echo "Error: the public SimpleFSDP Chorus benchmark currently requires GRADIENT_ACCUMULATION_STEPS=1" >&2
    exit 2
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
            --machine-rank|--host-ip|--host-port)
                echo "Error: managed Chorus option cannot be overridden: $arg" >&2
                exit 2
                ;;
        esac
    done
fi

read -r -a BATCH_SIZE_VALUES <<<"$BATCH_SIZES"
read -r -a SEQ_LENGTH_VALUES <<<"$SEQ_LENGTHS"
TOTAL_RUNS=$((${#BATCH_SIZE_VALUES[@]} * ${#SEQ_LENGTH_VALUES[@]}))
((TOTAL_RUNS > 0)) || {
    echo "Error: BATCH_SIZES and SEQ_LENGTHS must each contain at least one value" >&2
    exit 2
}
((BASE_MASTER_PORT + TOTAL_RUNS - 1 <= 65535)) || {
    echo "Error: MASTER_PORT is too high for a sweep with $TOTAL_RUNS configurations" >&2
    exit 2
}

SWEEP_INDEX=0
for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
    for SEQ_LENGTH in "${SEQ_LENGTH_VALUES[@]}"; do
        RUN_MASTER_PORT=$((BASE_MASTER_PORT + SWEEP_INDEX))
        ARGS=(
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

        [[ "$ACTIVATION_CHECKPOINTING" == 1 ]] && ARGS+=(--activation-checkpointing)
        [[ "$LOAD_WEIGHTS" == 1 ]] && ARGS+=(--load_weights)
        [[ -n "$MODEL_PATH" ]] && ARGS+=(--model-dir "$MODEL_PATH")
        [[ -n "$DATASET_PATH" ]] && ARGS+=(--dataset-path "$DATASET_PATH")
        [[ "$PROFILE" == 1 ]] && ARGS+=(--profile --profile-dir "$PROFILE_DIR")
        if ((${#EXTRA_ARGS[@]})); then
            ARGS+=("${EXTRA_ARGS[@]}")
        fi

        MASTER_PORT=$RUN_MASTER_PORT bash "$SCRIPT_DIR/launch.sh" "${ARGS[@]}"
        SWEEP_INDEX=$((SWEEP_INDEX + 1))
    done
done
