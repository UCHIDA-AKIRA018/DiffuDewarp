#!/bin/sh
# カレントディレクトリでジョブを実行する場合に指定
#$ -cwd

#$ -l gpu_1=1
# 実行時間を指定
#$ -l h_rt=24:00:00
#$ -N DiffuDewarp_20
#$ -p -3
# Moduleコマンドの初期化
. /etc/profile.d/modules.sh

# CUDA環境の読込
module load cuda/12.3.2  cudnn/9.0.0 
module load ffmpeg/6.1.1 

source VENV_DiffusionAD/bin/activate

python train.py 20
