#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=ExampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

#module purge
#module load 2021
#module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
#cd $HOME/code/proto2proto-main/
# Activate your environment
source activate myenv

# Run your code
RANDOMN=$$

runName=student_Resnet18
newRun=False
serviceType=recognition
ablationType=Resnet50_18_birds

# print working directory
# echo "$PWD"

echo "Baseline Student Resnet 18"
 
mkdir Experiments/$ablationType/$runName/$RANDOMN

expOut=Experiments/$ablationType/$runName/$RANDOMN.out
errorOut=Experiments/$ablationType/$runName/$RANDOMN/error.out

echo "Seed Search"
# for i in {10..11}
# do
#   echo "Seed $i"
cp Experiments/$ablationType/$runName/args.yaml Experiments/$ablationType/$runName/$RANDOMN/args.yaml
#   sed -i -e 's/seed: [0-9]*/seed: $i/g' Experiments/$ablationType/$runName/$RANDOMN/args.yaml
  
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main_search.py $runName $newRun $serviceType $RANDOMN $ablationType > $expOut 2>$errorOut
# done

