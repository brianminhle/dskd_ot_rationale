#!/bin/bash
GPUS=(0 1)
# GPUS=(4 5 6 7)

WORK_DIR=/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")
# echo ${DEVICE}
DATASET=dialogsum
MODEL_PATH=/mnt/bn/magellan-product-llm-data/tu.vu/matrix_one/dskd_ot_rationale/model_hub/gpt2/gpt2-xl
CKPT_PATH=${1}
BATCH_SIZE=${2-16}

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} ${DATASET} ${BATCH_SIZE} $seed
done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
# done
# for seed in 10 20 30 40 50
# do
#     bash ${WORK_DIR}/scripts/eval/eval_main_lora.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${MODEL_PATH} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
# done

# bash scripts/eval/run_eval_lora.sh /home/aac/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/dual_space_kd_with_cma/criterion=dual_space_kd_with_cma__adaptive_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=mistral__kd^rate=0.5__kd^temp=2.0__tea^temp=__epoch=10__bsz=4x2x4=32__lr=0.001/epoch2_step856_loss2.6540_rougel18.7531 8 >> xxxxxxxxxxxxxxxxxxxxxxx.log 2>&1 &
# bash scripts/eval/run_eval_lora.sh /home/aac/dskd_ot_rationale/outputs/tinyllama/tinyllama-1.1b-3T/various_divergence_ot_rationale/rationale/dolly/criterion=various_divergence_ot_rationale__forward_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=llama2__kd^rate=0.5__kd^temp=2.0__epoch=10__bsz=4x2x4=32__lr=0.001/epoch10_step4280_loss2.1956_rougel29.5673 8 >> tiny_llama_same_vocab_ot_rationale.log 2>&1 &

# bash scripts/eval/run_eval_lora.sh /home/aac/DSKD/outputs/qwen257B_Instruct/Qwen2.5-7B-Instruct/sft/raw/dolly/criterion=cross_entropy__lora-rank=256-alpha=8-dropout=0.1-bf16__epoch=1__bsz=16x1x8=128__lr=0.001/epoch1_step90_loss1.5837_rougel26.5747 8 >> testttttttttttttttt.log 

