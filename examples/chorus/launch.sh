#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
NUM_NODES=${NUM_NODES:-1}

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

is_positive_integer "$NUM_NODES" || die "NUM_NODES must be a positive integer"
is_positive_integer "$NGPUS_PER_NODE" || die "NGPUS_PER_NODE must be a positive integer"
export NUM_NODES NGPUS_PER_NODE

if [[ "$NUM_NODES" == 1 ]]; then
    if [[ -n "${MACHINE_RANK:-}" && "$MACHINE_RANK" != 0 ]]; then
        die "MACHINE_RANK must be 0 when NUM_NODES=1"
    fi
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
    is_positive_integer "$MASTER_PORT" || die "MASTER_PORT must be an integer from 1 to 65535"
    ((MASTER_PORT <= 65535)) || die "MASTER_PORT must be an integer from 1 to 65535"
    export MACHINE_RANK=0
    exec bash "$SCRIPT_DIR/launch_worker.sh" --machine-rank 0 "$@"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    [[ -n "${MASTER_ADDR:-}" ]] || die "MASTER_ADDR is required for a non-Slurm multi-node run"
    case "$MASTER_ADDR" in
        127.*|localhost|localhost.*|::1|0.0.0.0)
            die "MASTER_ADDR must be the reachable hostname or IP of node rank 0 for a multi-node run"
            ;;
    esac
    [[ -n "${MACHINE_RANK:-}" ]] || die "MACHINE_RANK is required for a non-Slurm multi-node run"
    is_nonnegative_integer "$MACHINE_RANK" || die "MACHINE_RANK must be a non-negative integer"
    ((MACHINE_RANK < NUM_NODES)) || die "MACHINE_RANK must be smaller than NUM_NODES"
    export MASTER_PORT=${MASTER_PORT:-29500}
    is_positive_integer "$MASTER_PORT" || die "MASTER_PORT must be an integer from 1 to 65535"
    ((MASTER_PORT <= 65535)) || die "MASTER_PORT must be an integer from 1 to 65535"
    exec bash "$SCRIPT_DIR/launch_worker.sh" --machine-rank "$MACHINE_RANK" "$@"
fi

MAIN_HOST=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
    MAIN_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | sed -n '1p')
fi
export MASTER_ADDR=${MASTER_ADDR:-${MAIN_HOST:-$(hostname)}}
export MASTER_PORT=${MASTER_PORT:-$((15000 + RANDOM % 10000))}
is_positive_integer "$MASTER_PORT" || die "MASTER_PORT must be an integer from 1 to 65535"
((MASTER_PORT <= 65535)) || die "MASTER_PORT must be an integer from 1 to 65535"

LOG_DIR=${LOG_DIR:-$SCRIPT_DIR/logs}
mkdir -p "$LOG_DIR"
SRUN_LOG_FILE=${SRUN_LOG_FILE:-$LOG_DIR/${SLURM_JOB_NAME:-chorus_bench}-${SLURM_JOB_ID:-local}.out.log}

srun \
    --nodes="$NUM_NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node="$NGPUS_PER_NODE" \
    --label \
    --export=ALL,NUM_NODES="$NUM_NODES",NGPUS_PER_NODE="$NGPUS_PER_NODE",MASTER_ADDR="$MASTER_ADDR",MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/launch_worker.sh" "$@" \
    2>&1 | tee -a "$SRUN_LOG_FILE"
