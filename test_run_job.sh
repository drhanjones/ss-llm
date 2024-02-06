#!/bin/bash

#SBATCH --job-name=ss-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH -t 1:30:00

echo $SLURM_JOB_ID

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/repo/ss-llm/mt1/bin/activate

cd $HOME/repo/ss-llm/nanoGPT

python train.py config/simple_wikipedia_char.py