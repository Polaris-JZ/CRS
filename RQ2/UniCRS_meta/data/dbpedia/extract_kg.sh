#!/bin/bash
#SBATCH -J extract_kg
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=0-01:00:00
#SBATCH --output=extract.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python extract_kg.py