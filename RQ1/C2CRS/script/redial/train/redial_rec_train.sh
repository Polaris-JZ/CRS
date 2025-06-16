#!/bin/bash
#SBATCH -J cecrs_redial_rec    
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train_rec.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python run_crslab.py \
    -c config/crs/c2crs/redial.yaml \
    -g 0 \
    -ss \
    -ct 256 \
    -it 100 \
    --scale 1.0 \
    -pbs 256 \
    -rbs 256  \
    -cbs 256 \
    --info_truncate 40  \
    --coarse_loss_lambda 0.2 \
    --fine_loss_lambda 1.0 \
    --coarse_pretrain_epoch 12 \
    --pretrain_epoch 25 \
    --rec_epoch 50 \
    --conv_epoch 0  \