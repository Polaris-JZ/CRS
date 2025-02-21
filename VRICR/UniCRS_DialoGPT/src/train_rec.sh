#!/bin/bash
#SBATCH -J uni_re_rec
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train_rec.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
accelerate launch train_rec.py \
    --dataset redial_gen \
    --tokenizer microsoft/DialoGPT-small \
    --model microsoft/DialoGPT-small \
    --text_tokenizer roberta-base \
    --text_encoder roberta-base \
    --n_prefix_rec 10 \
    --prompt_encoder /projects/prjs1158/KG/redail/UniCRS_update/output/best \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 530 \
    --context_max_length 200 \
    --prompt_max_length 200 \
    --entity_max_length 32 \
    --learning_rate 1e-4 \
    --output_dir /projects/prjs1158/KG/redail/UniCRS_update/output_rec \