#!/bin/bash
#SBATCH -J um-n-red
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=test.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python test.py --do_eval \
                --beam_size=1 \
                --warmup_steps=400 \
                --max_seq_length=512 \
                --max_target_length=100 \
                --gpu=0 \
                --overwrite_output_dir \
                --per_gpu_train_batch_size=8 \
                --per_gpu_eval_batch_size=8 \
                --model_name_or_path=/projects/prjs1158/KG/tgredail/UniMIND/unimind-n/bart-base-chinese \
                --data_name=redail



