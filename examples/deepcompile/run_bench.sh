PROFILE_DIR=${PROFILE_DIR:-"profiles"}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"
LOAD_OPTS="--load_weights"
MODEL_PATH=${MODEL_PATH:-"/home/bingxing2/home/scx7euw/.cache/modelscope/hub/models/LLM-Research/"}

export NUM_NODES=${NUM_NODES:-4}

# MODEL=${MODEL_NAME:-"Mistral-7B-Instruct-v0.3"}
# MODEL=${MODEL_NAME:-"Llama-3.2-1B-Instruct"}
MODEL=${MODEL_NAME:-"Llama-3.2-3B-Instruct"}
# MODEL=${MODEL_NAME:-"llama-2-7b"}
# MODEL=${MODEL_NAME:-"Meta-Llama-3-8B"}
# MODEL=${MODEL_NAME:-"Qwen2.5-7B-Instruct"}
# MODEL=${MODEL_NAME:-"Qwen3-8B"}
# MODEL=${MODEL_NAME:-"Qwen3-4B"}
BATCH_SIZE_OPTS=(2)
SEQ_LENGTH_OPTS=(1024)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        # 如果要预加载权重, 加上${LOAD_OPTS}， 固定种子--deterministic
        ARGS="--model ${MODEL} --model_path ${MODEL_PATH} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS}"
        # bash ./run_multinode.sh --backend deepspeed ${ARGS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch,selective_gather
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes global_layer_scheduler
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_activation_recompute
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done
