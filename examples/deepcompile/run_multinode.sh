#!/bin/bash

echo $*

SCRIPT_DIR=$(dirname $(realpath $0))
NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# verify that NUM_NODES is a positive integer
if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_NODES must be a positive integer"
    exit 1
fi

# check if NUM_NODES ==1 or hostfile_n${NUM_NODES} exists (only for non-Slurm runs)
if [ -z "${SLURM_JOB_ID:-}" ] && [ ! -f hostfile_n${NUM_NODES} ] && [ "${NUM_NODES}" != "1" ]; then
    echo "Error: hostfile_n${NUM_NODES} does not exist"
    exit 1
fi

if [ "${NUM_NODES}" == "1" ]; then
    # avoid dependency on pdsh when possible
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
    export MASTER_PORT="${MASTER_PORT:-12345}"
    cd ${SCRIPT_DIR}; bash ./run.sh --host-ip "${MASTER_ADDR}" --host-port "${MASTER_PORT}" $*
else
    if [ -n "${SLURM_JOB_ID}" ]; then
        MAIN_HOST=""
        if [ -n "${SLURM_NODELIST:-}" ]; then
            MAIN_HOST=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
        fi
        export MASTER_ADDR="${MASTER_ADDR:-${MAIN_HOST}}"
        export MASTER_ADDR="${MASTER_ADDR:-$(hostname)}"
        export MASTER_PORT="${MASTER_PORT:-$((15000 + RANDOM % 10000))}"
        LOG_DIR="${SCRIPT_DIR}/logs"
        mkdir -p "${LOG_DIR}"
        SRUN_LOG_FILE="${SRUN_LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-deepcompile_bench}.out.log}"
        set -o pipefail
        srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --gpus-per-node="${NGPUS_PER_NODE}" \
            --label \
            --export=ALL,NUM_NODES="${NUM_NODES}",NGPUS_PER_NODE="${NGPUS_PER_NODE}",MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
            bash "${SCRIPT_DIR}/run.sh" --host-ip "${MASTER_ADDR}" --host-port "${MASTER_PORT}" $* \
            2>&1 | tee -a "${SRUN_LOG_FILE}" >/dev/null
    else
        echo "Error: multi-node run requires Slurm; SLURM_JOB_ID is not set."
        exit 1
    fi
fi
