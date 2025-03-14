#!/bin/bash

# 80から94までの数値をループで処理
# for i in {2037376..2037397}; 
# do
#     qdel $i
# done

# items=(
#     "9"    
#     "10"    
#     "11"    
#     "12"    
#     "13"    
#     "14"    
# )
# for i in "${items[@]}"; 
for i in {9..14}; 
do
    echo "args$iを実行"
    cat <<EOF > S_DiffuDewarp_$i.sh
#!/bin/sh
#$ -cwd

#$ -l gpu_1=1
#$ -l h_rt=24:00:00

. /etc/profile.d/modules.sh

module load cuda/12.3.2  cudnn/9.0.0 
module load ffmpeg/6.1.1 

source VENV_DiffusionAD/bin/activate

python train.py $i
EOF

    qsub -g tga-RLA S_DiffuDewarp_$i.sh
done