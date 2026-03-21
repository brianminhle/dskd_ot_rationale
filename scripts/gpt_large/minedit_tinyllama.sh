#! /bin/bash
GPUS=(0 1 2 3)

export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale
CKPT_TYPE="gpt2" # ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"] "qwen" "qwen257B_Instruct"
CKPT_NAME="gpt2-xl"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_TYPE}/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"  # gpt2, qwen, mistral, llama2
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_TYPE}/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale/outputs/qwen/Qwen2.5-7B-Instruct/sft/raw/self_instruct/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=10__bsz=16x1x4=64__lr=0.001/epoch8_step1304_loss1.4630_rougel46.8340"
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"
PROJECTOR_LR=0.001
# data
DATASET_NAME="self_instruct" # 
DATA_DIR="${BASE_PATH}/data/${DATASET_NAME}/"
DATASET='raw' # raw , rationale
# task
TASK="min_edit"
# hp
BATCH_SIZE=1
LR=0.001
GRAD_ACC=1
EVAL_BATCH_SIZE=16
EPOCH=10
KD_RATE=0.5
KD_TEMP=2.0
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1
# length
MAX_LENGTH=512
# runtime
PRECISION="bf16"
CRITERION="min_edit_dis_kld"
TEA2STU_ID_MAP="${BASE_PATH}/data/vocab_alignment/qwen_to_gpt2/tea2stu_id_mapping.json"
CONFIG="lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
SETTING=criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_TYPE}__kd^rate=${KD_RATE}__kd^temp=${KD_TEMP}__tea^temp=${TEA_TEMP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x${GPUS_PER_NODE}=$((BATCH_SIZE * GRAD_ACC * GPUS_PER_NODE * NNODES))__lr=${LR}
SAVE_PATH="${BASE_PATH}/outputs/${CKPT_TYPE}/${CKPT_NAME}/${TASK}/${DATASET}/${DATASET_NAME}/${SETTING}" 
SAVE_BEST_N_CKPTS=1
# seed
SEED=10

mkdir -p ${SAVE_PATH}

OPTS=""
OPTS+=" --dataset ${DATASET}"

# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-type ${TEACHER_MODEL_TYPE}"
OPTS+=" --teacher-model-path ${TEACHER_MODEL_PATH}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# task
OPTS+=" --task ${TASK}"
OPTS+=" --teacher-to-student-id-mapping ${TEA2STU_ID_MAP}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --num-epochs ${EPOCH}"
OPTS+=" --kd-rate ${KD_RATE}"
OPTS+=" --kd-temperature ${KD_TEMP}"
OPTS+=" --peft lora"
OPTS+=" --peft-lora-r ${LORA_RANK}"
OPTS+=" --peft-lora-alpha ${LORA_ALPHA}"
OPTS+=" --peft-lora-dropout ${LORA_DROPOUT}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval 1"
OPTS+=" --eval-interval 1"
OPTS+=" --log-interval 50"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --keep-best-n-checkpoints ${SAVE_BEST_N_CKPTS}"
OPTS+=" --criterion ${CRITERION}"
# distiller
OPTS+=" --projector-lr ${PROJECTOR_LR}"
OPTS+=" --projector-config-path ${PROJECTOR_CONFIG_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
if [[ $PRECISION == "bf16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
elif [[ $PRECISION == "fp16" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
elif [[ $PRECISION == "fp32" ]]; then
    OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_fp32.json"
fi
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/distillation.py ${OPTS}"
echo ${CMD}
${CMD} \
>> ${SAVE_PATH}/train_min_edit_ot_01.log 2>&1
