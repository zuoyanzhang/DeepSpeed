PROFILE_DIR=${PROFILE_DIR:-"profiles"}
mkdir -p ${PROFILE_DIR}
PROFILE_OPTS="--profile --profile-dir ${PROFILE_DIR}"
COMPILE_OPTS="--compile"
DC_OPTS="--compile --deepcompile"
ACC_OPTS="--gradient-accumulation-steps 1"
AC_OPTS="--activation-checkpointing"
MODEL_PATH=${MODEL_PATH:-"/data/home/scvj326/run/.cache/modelscope/hub/models/LLM-Research/"}

export NUM_NODES=${NUM_NODES:-4}

MODEL=${MODEL_NAME:-"Llama-3.2-1B-Instruct"}
BATCH_SIZE_OPTS=(2)
SEQ_LENGTH_OPTS=(1024)
for BATCH_SIZE in ${BATCH_SIZE_OPTS[@]}; do
    for SEQ_LENGTH in ${SEQ_LENGTH_OPTS[@]}; do
        ARGS="--model ${MODEL} --model_path ${MODEL_PATH} --batch-size ${BATCH_SIZE} --seq-length ${SEQ_LENGTH} ${ACC_OPTS} ${AC_OPTS}"
        # bash ./run_multinode.sh --backend deepspeed ${ARGS}
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${COMPILE_OPTS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS}
        # bash ./run_multinode.sh --backend fsdp ${ARGS} ${COMPILE_OPTS}
        bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch,selective_gather
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes prefetch
        # bash ./run_multinode.sh --backend deepspeed ${ARGS} ${DC_OPTS} --passes selective_gather

        cp -r logs ${PROFILE_DIR}/
    done
done

