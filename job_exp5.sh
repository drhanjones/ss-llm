#!/bin/bash

#SBATCH --job-name=ss-llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.thamma@student.vu.nl
#SBATCH -t 3:30:00

# srun  --nodes=1 --partition=gpu --gpus-per-node=4 -t 0:5:00 --pty /bin/bash

# sbatch --export=config_file_name=wikipedia_bpe/train_wikipedia_gpt_exp2.py test_run_job.sh


echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

echo $SLURM_JOB_ID

#module load 2023
module load 2021
#module load Python/3.11.3-GCCcore-12.3.0
module load Python/3.9.5-GCCcore-10.3.0

source $HOME/repo/mt1_p39/bin/activate

cd $HOME/repo/ss-llm/nanoGPT

data_root=$HOME/repo/ss-llm/nanoGPT/data
output_root=$HOME/repo/ss-llm/nanoGPT/output_dump

#python train.py config/train_simplewiki_char.py
#python train.py config/$config_file_name

#torchrun --standalone --nproc_per_node=4 train.py config/train_wikipedia_gpt.py
mkfifo $HOME/blimp_out_dir
mkfifo $HOME/blimp_data_dir

#torchrun --standalone --nproc_per_node=4 train.py config/$config_file_name/babylm_full_bpe/ngram_models/adjusted_batch/train_blmf_8k_6x6_2gramgpt_nomask.py
torchrun --standalone --nproc_per_node=4 train.py config/$config_file_name
#wait

#assuming output directory is sourced into the environment by config file


#cd $HOME/repo/ss-llm/nanoGPT/eval/evaluation_pipeline/

#if [ -z "$blimp_out_dir" ]
#then
#read -r blimp_out_dir < $HOME/blimp_out_dir
#read -r blimp_data_dir < $HOME/blimp_data_dir

#output_dir=$output_root/$blimp_out_dir
#data_dir=$data_root/$blimp_data_dir

#python babylm_eval.py --model_type=nanogpt --output_dir $output_dir --data_dir $data_dir --tasks all

#rm $HOME/blimp_out_dir
#rm $HOME/blimp_data_dir

#else
#    echo "No BLIMP evaluation possible."3
#fi


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


