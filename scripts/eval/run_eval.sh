#!/bin/bash
# GPUS=(4 5 6 7)
GPUS=(0 1 2 3)
WORK_DIR=/home/aac/DSKD
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")

CKPT_PATH=${1}
BATCH_SIZE=${2-32}
DATASET=alpaca

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} ${DATASET} ${BATCH_SIZE} $seed
done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
# done
# bash scripts/eval/run_eval.sh /home/aac/DSKD/outputs/gpt2/gpt2-xlarge/sft/raw/alpaca/criterion=cross_entropy__default-fp16__epoch=10__bsz=4x2x4=32__lr=0.00002/epoch10_step3250_loss2.1139_rougel30.3165 8 >> alpaca_sft_gpt2_large.log 
