#!/bin/bash
#
#SBATCH --job-name=deepcompile_bench
#SBATCH --nodes=1
#SBATCH --qos=gpugpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out.log

module load miniforge3/24.1 compilers/gcc/11.3.0  compilers/cuda/12.4  cudnn/8.9.5.29_cuda12.x nccl/2.26.5-1_cuda12.4
source /home/bingxing2/apps/miniforge3/24.1.2/etc/profile.d/conda.sh
conda activate py310

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(dirname $(realpath $0))}
cd ${SCRIPT_DIR}

# 优先使用 SLURM 分配的 GPU 数目
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export NGPUS_PER_NODE="${SLURM_GPUS_ON_NODE}"
elif [ -n "${SLURM_GPUS_PER_NODE:-}" ]; then
    if [[ "${SLURM_GPUS_PER_NODE}" =~ ^[0-9]+$ ]]; then
        export NGPUS_PER_NODE="${SLURM_GPUS_PER_NODE}"
    else
        export NGPUS_PER_NODE=$(echo "${SLURM_GPUS_PER_NODE}" | sed -E 's/.*:([0-9]+)/\1/')
    fi
elif [ -n "${SLURM_JOB_GPUS:-}" ]; then
    export NGPUS_PER_NODE=$(echo "${SLURM_JOB_GPUS}" | awk -F',' '{print NF}')
else
    export NGPUS_PER_NODE=1
fi
export NUM_NODES=${SLURM_NNODES:-1}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-$((15000 + RANDOM % 10000))}

# 运行基准脚本（多节点启动由 run_multinode.sh 内部的 srun 负责）
bash ./run_bench.sh
