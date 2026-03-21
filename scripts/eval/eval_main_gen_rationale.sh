#! /bin/bash
set -e
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MASTER_ADDR=localhost
MASTER_PORT=29500
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
# GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH='/home/aac/DSKD'
CKPT_PATH='/home/aac/DSKD/outputs/qwen257B_Instruct/Qwen2.5-7B-Instruct/sft/raw/dolly/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=1__bsz=16x1x8=128__lr=0.001/epoch1_step90_loss1.5837_rougel26.5747'
MODEL_TYPE="qwen257B_Instruct"    # gpt2, qwen, mistral, llama2, qwen257B_Instruct


# task
TASK="eval_main"
# data
DATA_NAME=dolly

DATA_DIR="${BASE_PATH}/data/${DATA_NAME}"
DATA_NUM=-1      # "-1" means evaluation on all data

# hp
EVAL_BATCH_SIZE=16     # depends on your GPU memory
SEED=1    

# runtime
SAVE_PATH=$(dirname ${CKPT_PATH})

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT_PATH}"

OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type ${MODEL_TYPE}"
# task
OPTS+=" --task ${TASK}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAME}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num ${DATA_NUM}"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save-dir ${SAVE_PATH}"
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="/opt/conda/envs/py_3.10/bin/torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/code/evaluate.py ${OPTS}"
echo ${CMD}

${CMD}