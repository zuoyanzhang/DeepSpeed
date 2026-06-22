#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "${SCRIPT_DIR}"

# Local server defaults. Slurm used to inject these in sbatch_run_bench.sbatch.
export NUM_NODES=1
export NGPUS_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

PROFILE_DIR=${PROFILE_DIR:-"profiles"}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"
LOAD_OPTS="--load_weights"
FSDP2_PREFETCH_OPTS=${FSDP2_PREFETCH_OPTS:-"--fsdp2_forward_prefetch_distance 0 --fsdp2_backward_prefetch_distance 1"}
SIMPLEFSDP_PREFETCH_OPTS=${SIMPLEFSDP_PREFETCH_OPTS:-""}
SIMPLEFSDP_FAST_OPTS=${SIMPLEFSDP_FAST_OPTS:-"--simplefsdp_enable_compiled_autograd"}
if [ -z "${FSDP2_FAST_OPTS+x}" ]; then
        FSDP2_FAST_OPTS="--fsdp2-fast"
fi
# MODEL_PATH=${MODEL_PATH:-"/home/dev/.cache/modelscope/hub/models/LLM-Research/"}
MODEL_PATH=${MODEL_PATH:-"/home/dev/.cache/modelscope/hub/models/baichuan-inc/"}
# MODEL_PATH=${MODEL_PATH:-"/home/dev/.cache/modelscope/hub/models/Qwen/"}


# MODEL=${MODEL_NAME:-"Mistral-7B-Instruct-v0.3"}
# MODEL=${MODEL_NAME:-"Llama-3.2-1B-Instruct"}
# MODEL=${MODEL_NAME:-"Llama-3.2-3B-Instruct"}
# MODEL=${MODEL_NAME:-"llama-2-7b"}
# MODEL=${MODEL_NAME:-"Meta-Llama-3-8B"}
# MODEL=${MODEL_NAME:-"Qwen2.5-7B-Instruct"}
# MODEL=${MODEL_NAME:-"Qwen3-8B"}
# MODEL=${MODEL_NAME:-"Qwen3-4B"}
# MODEL=${MODEL_NAME:-"Qwen3-14B"}
MODEL=${MODEL_NAME:-"Baichuan2-13B-Base"}
BATCH_SIZE_OPTS=(1)
SEQ_LENGTH_OPTS=(2048)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # 如果要预加载权重, 加上${LOAD_OPTS}， 固定种子--deterministic
        ARGS="--model ${MODEL} ${LOAD_OPTS} --model_path ${MODEL_PATH} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS}"
        # bash ./run_multinode.sh --backend deepspeed ${ARGS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        # bash ./run_multinode.sh --backend fsdp2 ${ARGS} ${FSDP2_PREFETCH_OPTS} ${FSDP2_FAST_OPTS}
        # bash ./run_multinode.sh --backend fsdp2 ${ARGS} ${COMPILE_OPTS} ${FSDP2_PREFETCH_OPTS} ${FSDP2_FAST_OPTS}
        # bash ./run_multinode.sh --backend simplefsdp ${ARGS} ${COMPILE_OPTS} ${SIMPLEFSDP_FAST_OPTS} ${SIMPLEFSDP_PREFETCH_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch,selective_gather
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes global_layer_scheduler
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_activation_recompute
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done
