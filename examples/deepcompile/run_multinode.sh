#!/bin/bash

echo $*

SCRIPT_DIR=$(dirname $(realpath $0))
HOST_IP=$(hostname -i)
NUM_NODES=${NUM_NODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# verify that NUM_NODES is a positive integer
if ! [[ "$NUM_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: NUM_NODES must be a positive integer"
    exit 1
fi

# check if NUM_NODES ==1 or hostfile_n${NUM_NODES} exists
if [ ! -f hostfile_n${NUM_NODES} ] && [ "${NUM_NODES}" != "1" ]; then
    echo "Error: hostfile_n${NUM_NODES} does not exist"
    exit 1
fi

if [ "${NUM_NODES}" == "1" ]; then
    # avoid dependency on pdsh when possible
    cd ${SCRIPT_DIR}; bash ./run.sh --host-ip ${HOST_IP} $*
else
    if [ -n "${SLURM_JOB_ID}" ]; then
        if [ -n "${SLURM_NODELIST}" ]; then
            MAIN_HOST=$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)
            HOST_IP=$(getent hosts "${MAIN_HOST}" | awk '{print $1}')
        fi
        if [ -z "${HOST_IP}" ]; then
            HOST_IP=$(hostname -i)
        fi
        LOG_DIR="${SCRIPT_DIR}/logs"
        mkdir -p "${LOG_DIR}"
        srun --nodes="${NUM_NODES}" --ntasks-per-node=1 --gpus-per-node="${NGPUS_PER_NODE}" \
            --output="${LOG_DIR}/srun-%t.out.log" \
            --export=ALL,NUM_NODES="${NUM_NODES}",NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
            bash "${SCRIPT_DIR}/run.sh" --host-ip "${HOST_IP}" $*
    else
        echo "Error: multi-node run requires Slurm; SLURM_JOB_ID is not set."
        exit 1
    fi
fi
