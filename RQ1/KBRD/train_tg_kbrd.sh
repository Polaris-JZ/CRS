#!/bin/bash
#SBATCH -J kbrd_tg
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train_tg_kbrd.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python run_crslab.py --config config/crs/kbrd/tgredial.yaml --gpu 0