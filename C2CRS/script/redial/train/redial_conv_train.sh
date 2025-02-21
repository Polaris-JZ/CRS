#!/bin/bash
#SBATCH -J cecrs_redial_conv   
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=5-00:00:00
#SBATCH --output=train_conv.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
    python run_crslab.py \
    -c config/crs/c2crs/redial.yaml \
    -g 0 \
    -ss  \
    -ct 256 \
    -it 100 \
    --scale 1.0 \
    -pbs 256 \
    -rbs 256 \
    -cbs 256 \
    --info_truncate 40  \
    --coarse_loss_lambda 0.2 \
    --fine_loss_lambda 1.0 \
    --coarse_pretrain_epoch 0 \
    --pretrain_epoch 0 \
    --rec_epoch 0 \
    --conv_epoch 23 \
    -rs \
    --restore_path /projects/prjs1158/KG/redail/C2CRS/save/ReDial_C2CRS_Model2025-02-01-09-26-42 \
    --model_file_for_restore C2CRS_Model_0.pth \
    --freeze_parameters_name k_c  \
    --freeze_parameters \
    --logit_type hs_copy2 \
    --is_coarse_weight_loss \
    --token_freq_th 1500 \
    --coarse_weight_th 0.02
