#!/bin/bash
#SBATCH -J mese-redail
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python -u main.py  \
    --gpu \
    --dataset TG \
    --task recommend \
    --ckpt data/ckpt/TG/recommend/{task_ID_for_pretrain}/{last_ckpt_path_for_pretrain}