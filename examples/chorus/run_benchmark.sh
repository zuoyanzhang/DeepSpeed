#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

BACKEND=${BACKEND:-deepspeed}
MODEL_NAME=${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.3}
MODEL_PATH=${MODEL_PATH:-}
DATASET_PATH=${DATASET_PATH:-}
PROFILE_DIR=${PROFILE_DIR:-profiles}
BATCH_SIZES=${BATCH_SIZES:-2}
SEQ_LENGTHS=${SEQ_LENGTHS:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-1}
LOAD_WEIGHTS=${LOAD_WEIGHTS:-1}
PROFILE=${PROFILE:-0}
EXTRA_ARGS=("$@")

case "$BACKEND" in
    deepspeed|simplefsdp) ;;
    *)
        echo "Error: BACKEND must be deepspeed or simplefsdp" >&2
        exit 2
        ;;
esac

if [[ "$BACKEND" == simplefsdp && "$GRADIENT_ACCUMULATION_STEPS" != 1 ]]; then
    echo "Error: the public SimpleFSDP Chorus benchmark currently requires GRADIENT_ACCUMULATION_STEPS=1" >&2
    exit 2
fi

if ((${#EXTRA_ARGS[@]})); then
    for arg in "${EXTRA_ARGS[@]}"; do
        option_name=${arg%%=*}
        case "$option_name" in
            --backend|--distributed-backend|--distributed_backend|--compile|--deepcompile|--passes|\
            --simplefsdp-enable-chorus|--simplefsdp_enable_chorus|\
            --simplefsdp-enable-compiled-autograd|--simplefsdp_enable_compiled_autograd|\
            --model|--model-name|--model_name|--model-dir|--model_dir|--model-path|--model_path|\
            --dataset-path|--dataset_path|--batch-size|--batch_size|--seq-length|--seq_length|\
            --gradient-accumulation-steps|--gradient_accumulation_steps|\
            --activation-checkpointing|--activation_checkpointing|--load-weights|--load_weights|\
            --profile|--profile-dir|--profile_dir)
                echo "Error: managed Chorus option cannot be overridden: $arg" >&2
                exit 2
                ;;
        esac
    done
fi

read -r -a BATCH_SIZE_VALUES <<<"$BATCH_SIZES"
read -r -a SEQ_LENGTH_VALUES <<<"$SEQ_LENGTHS"

for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
    for SEQ_LENGTH in "${SEQ_LENGTH_VALUES[@]}"; do
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

        bash "$SCRIPT_DIR/launch.sh" "${ARGS[@]}"
    done
done
