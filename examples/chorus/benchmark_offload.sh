#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

PROFILE_DIR=${PROFILE_DIR:-profile_offload}
mkdir -p "$PROFILE_DIR"

mkdir -p logs

export LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs_offload}
mkdir -p "$LOG_DIR"

MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH_OPTS=(1024)
for BATCH_SIZE in "${BATCH_SIZE_OPTS[@]}"; do
    for SEQ_LENGTH in "${SEQ_LENGTH_OPTS[@]}"; do
        ARGS=(--model "$MODEL" --batch-size "$BATCH_SIZE" --seq-length "$SEQ_LENGTH"
              --gradient-accumulation-steps 1 --activation-checkpointing
              --profile --profile-dir "$PROFILE_DIR" --zero-stage 3)
        bash "$SCRIPT_DIR/launch.sh" --backend deepspeed "${ARGS[@]}"
        bash "$SCRIPT_DIR/launch.sh" --backend deepspeed "${ARGS[@]}" --ds-offload
        bash "$SCRIPT_DIR/launch.sh" --backend deepspeed "${ARGS[@]}" --compile --deepcompile --eager --passes offload_adam_states
        bash "$SCRIPT_DIR/launch.sh" --backend deepspeed "${ARGS[@]}" --compile --deepcompile --eager --passes offload_adam_states_sync
    done
done
