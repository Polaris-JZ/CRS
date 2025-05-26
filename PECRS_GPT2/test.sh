#!/bin/bash
#SBATCH -J um-n-redail
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python main.py \
    --mode=eval \
    --load_model_path=./Outputs/REDIAL/temp/CRS_Train_best_model.pt \
                              