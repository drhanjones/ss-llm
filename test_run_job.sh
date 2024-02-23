#!/bin/bash

#SBATCH --job-name=ss-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH -t 2:30:00

echo $SLURM_JOB_ID

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/repo/mt1/bin/activate

cd $HOME/repo/ss-llm/nanoGPT

#python train.py config/train_simplewiki_char.py

torchrun --standalone --nproc_per_node=4 train.py config/train_wikipedia_gpt.py


#This training script can be run both on a single gpu in debug mode,
#and also in a larger training run with distributed data parallel (ddp).
#
#To run on a single GPU, example:
#$ python train.py --batch_size=32 --compile=False
#
#To run with DDP on 4 gpus on 1 node, example:
#$ torchrun --standalone --nproc_per_node=4 train.py
#
#To run with DDP on 4 gpus across 2 nodes, example:
#- Run on the first (master) node with example IP 123.456.123.456:
#$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
#- Run on the worker node:
#$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
#(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
#"""