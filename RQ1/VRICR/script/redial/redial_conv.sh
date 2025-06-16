#!/bin/bash
#SBATCH -J vricr-re_conv
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=train_conv.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1


srun --gres=gpu:1 \
    python 	-u main.py \
  		--gpu \
   		--train_batch_size 48 \
    	--dataset Redial  \
      	--task   generation \
       	--ckpt data/ckpt/ckpt_redial_rec.model.ckpt \
       	--data_processed