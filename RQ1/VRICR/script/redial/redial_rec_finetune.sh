#!/bin/bash
#SBATCH -J vricr-re_rec
#SBATCH -p gpu_a100
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=train_rec.txt  # 标准输出文件

# 激活 conda 环境
source ~/.bashrc
conda activate torch2.1.1

srun --gres=gpu:1 \
	python -u main.py \
		--gpu \
		--dataset Redial \
		--task recommend  \
		--ckpt /projects/prjs1158/KG/redail/VRICR_update/data/ckpt/Redial/recommend/120c455a/13650.model.ckpt   \
		--data_processed \
		--alpha 0.9 \
		--beta 0.8 \
		--gamma 10 \
		--lambda_3 0.0055