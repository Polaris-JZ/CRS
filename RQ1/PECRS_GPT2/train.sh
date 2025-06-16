#!/bin/bash
#SBATCH -J pecrs_redail
#SBATCH -p gpu_h100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python main.py --train_bs 128 --eval_bs 128 \
    --alpha 0.2 \
    --beta 0.8 \
    --gamma 1.0