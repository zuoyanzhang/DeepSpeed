#!/usr/bin/env bash
#SBATCH --job-name=chorus_bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --output=chorus-%j.out.log
#SBATCH --error=chorus-%j.out.log
#SBATCH --open-mode=append

set -Eeuo pipefail

if [[ -n "${CHORUS_REPO_ROOT:-}" ]]; then
    SCRIPT_DIR=$(cd -- "$CHORUS_REPO_ROOT/examples/chorus" && pwd)
elif [[ -f "${SLURM_SUBMIT_DIR:-}/examples/chorus/run_benchmark.sh" ]]; then
    SCRIPT_DIR=$(cd -- "$SLURM_SUBMIT_DIR/examples/chorus" && pwd)
else
    echo "Error: submit from the Chorus repository root or set CHORUS_REPO_ROOT" >&2
    exit 2
fi
cd "$SCRIPT_DIR"
mkdir -p logs

# Optional site-specific setup. The referenced file may load modules and
# activate a Python environment, but is intentionally kept outside the repo.
if [[ -n "${CHORUS_ENV_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$CHORUS_ENV_SETUP"
fi

if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "$SLURM_GPUS_ON_NODE" =~ ^[0-9]+$ ]]; then
    export NGPUS_PER_NODE=$SLURM_GPUS_ON_NODE
elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    if [[ "$SLURM_GPUS_PER_NODE" =~ ^[0-9]+$ ]]; then
        export NGPUS_PER_NODE=$SLURM_GPUS_PER_NODE
    else
        export NGPUS_PER_NODE=$(sed -E 's/.*:([0-9]+)/\1/' <<<"$SLURM_GPUS_PER_NODE")
    fi
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    export NGPUS_PER_NODE=$(awk -F',' '{print NF}' <<<"$SLURM_JOB_GPUS")
else
    export NGPUS_PER_NODE=1
fi

export NUM_NODES=${SLURM_NNODES:-1}
if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_NODELIST:-}" ]]; then
        export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | sed -n '1p')
    else
        export MASTER_ADDR=127.0.0.1
    fi
fi
export MASTER_PORT=${MASTER_PORT:-$((15000 + RANDOM % 10000))}

echo "===== BEGIN Chorus job_id=${SLURM_JOB_ID:-unknown} start=$(date '+%F %T') ====="
if bash "$SCRIPT_DIR/run_benchmark.sh"; then
    status=0
else
    status=$?
fi
echo "===== END Chorus job_id=${SLURM_JOB_ID:-unknown} status=$status end=$(date '+%F %T') ====="
exit "$status"
