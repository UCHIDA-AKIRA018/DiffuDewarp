#!/bin/bash
# for i in {2126177..2126184}; 
# do
#     qdel $i
# done

items=(
    "101"    
    "103"    
    "105"    
)
for i in "${items[@]}"; 
# for i in {9..14}; 
do
    echo "args$iを実行"
    cat <<EOF > S_DiffuDewarp_test_$i.sh
#!/bin/sh
#$ -cwd

#$ -l gpu_1=1
#$ -l h_rt=01:00:00
#$ -p -3

. /etc/profile.d/modules.sh

module load cuda/12.3.2  cudnn/9.0.0 
module load ffmpeg/6.1.1 

source VENV_DiffusionAD/bin/activate

python eval.py $i
EOF

    qsub -g tga-RLA S_DiffuDewarp_test_$i.sh
    rm S_DiffuDewarp_test_$i.sh
done