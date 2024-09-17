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
#SBATCH -t 1:00:00

# srun  --nodes=1 --partition=gpu --gpus-per-node=4 -t 0:5:00 --pty /bin/bash

#python babylm_eval.py --model_type=nanogpt --output_dir /home/athamma/repo/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-6x6-nomask-5768066-2gramgpt --data_dir /home/athamma/repo/ss-llm/nanoGPT/data/babylm_full_bpe_8k --tasks all
# sbatch --export=config_file_name=wikipedia_bpe/train_wikipedia_gpt_exp2.py test_run_job.sh

echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

echo $SLURM_JOB_ID

module load 2021
#module load Python/3.11.3-GCCcore-12.3.0

module load Python/3.9.5-GCCcore-10.3.0

source $HOME/repo/mt1_p39/bin/activate

cd $HOME/repo/ss-llm/nanoGPT/data/babylm_full_bpe_100M_8k


python prepare.py


