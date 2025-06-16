#!/bin/bash
#SBATCH -J vricr-redail
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=train.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1


srun --gres=gpu:1 \
    python -u main.py  \
    --gpu \
    --dataset Redial \
    --task recommend \
    --pretrain \
    --efficient_train_batch_size 256