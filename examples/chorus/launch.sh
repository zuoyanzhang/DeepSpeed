#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUM_NODES=${NUM_NODES:-1}
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN=$PYTHON
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=python3
fi

if [[ -z "${NGPUS_PER_NODE:-}" ]]; then
    NGPUS_PER_NODE=$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')
fi

if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_NODES must be a positive integer" >&2
    exit 2
fi
if ! [[ "$NGPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NGPUS_PER_NODE must be a positive integer" >&2
    exit 2
fi

if [[ "$NUM_NODES" == 1 ]]; then
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
    export MACHINE_RANK=0
    exec bash "$SCRIPT_DIR/launch_worker.sh" --machine-rank 0 "$@"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: multi-node launch currently requires a Slurm allocation" >&2
    exit 2
fi

MAIN_HOST=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
    MAIN_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | sed -n '1p')
fi
export MASTER_ADDR=${MASTER_ADDR:-${MAIN_HOST:-$(hostname)}}
export MASTER_PORT=${MASTER_PORT:-$((15000 + RANDOM % 10000))}

LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs}
mkdir -p "$LOG_DIR"
SRUN_LOG_FILE=${SRUN_LOG_FILE:-$LOG_DIR/${SLURM_JOB_NAME:-chorus_bench}.out.log}

srun \
    --nodes="$NUM_NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node="$NGPUS_PER_NODE" \
    --label \
    --export=ALL,NUM_NODES="$NUM_NODES",NGPUS_PER_NODE="$NGPUS_PER_NODE",MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/launch_worker.sh" "$@" \
    2>&1 | tee -a "$SRUN_LOG_FILE"
