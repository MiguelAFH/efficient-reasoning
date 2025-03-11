#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="../logs/dpo_train_$DATE.log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

set -x

export CUDA_VISIBLE_DEVICES=0

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ../checkpoints/deepseek_qwen_1.5B_$DATE \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 255 \
   --micro_train_batch_size 1 \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --bf16 \
   --max_epochs 3 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset ../data/open-r1/OpenR1-Math-220k_filtered/ \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
