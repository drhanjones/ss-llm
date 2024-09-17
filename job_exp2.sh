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
#SBATCH -t 30:00

# srun  --nodes=1 --partition=gpu --gpus-per-node=4 -t 0:5:00 --pty /bin/bash

srun  --nodes=1 --partition=gpu --gpus-per-node=4 -t 0:45:00 --pty /bin/bash
python babylm_eval.py --model_type=nanogpt --output_dir /home/athamma/repo/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-6x6-nomask-5768066-2gramgpt --data_dir /home/athamma/repo/ss-llm/nanoGPT/data/babylm_full_bpe_8k --tasks all
# sbatch --export=config_file_name=wikipedia_bpe/train_wikipedia_gpt_exp2.py test_run_job.sh

echo "Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

echo $SLURM_JOB_ID

module load 2021
#module load Python/3.11.3-GCCcore-12.3.0

module load Python/3.9.5-GCCcore-10.3.0

source $HOME/repo/mt1_p39/bin/activate

cd $HOME/repo/ss-llm/nanoGPT/eval/evaluation_pipeline/

output_root=$HOME/repo/ss-llm/nanoGPT/output_dump
data_root=$HOME/repo/ss-llm/nanoGPT/data

#output_dir=$output_root/out-babylm_full_bpe-4x4-nomask-5444724
#data_dir=$data_root/babylm_full_bpe

case $SLURM_ARRAY_TASK_ID in
    1)
        output_dir=$output_root/out-babylm_full_bpe_8k-6x6-mask_e002-5734464_s1337
        data_dir=$data_root/babylm_full_bpe_8k
        ;;
    2)
        output_dir=$output_root/out-babylm_full_bpe_8k-6x6-mask_e100-5734467_s1337
        data_dir=$data_root/babylm_full_bpe_8k
        ;;
    3)
        output_dir=$output_root/out-babylm_full_bpe_8k-6x6-mask_e250-5734550
        data_dir=$data_root/babylm_full_bpe_8k
        ;;
    4)
        output_dir=$output_root/out-babylm_full_bpe_8k-6x6-mask_e010-5734465_s1337
        data_dir=$data_root/babylm_full_bpe_8k
        ;;

esac



python babylm_eval.py --model_type=nanogpt --output_dir $output_dir --data_dir $data_dir --tasks all


#sbatch -a 1-4 job_exp2.sh



#case $SLURM_ARRAY_TASK_ID in
#    1)
#        output_dir=$output_root/out-babylm_full_bpe-4x4-nomask-1709506109
#        data_dir=$data_root/babylm_full_bpe
#        ;;
#    2)
#        output_dir=$output_root/out-babylm_full_bpe-2x2-nomask-1709512418
#        data_dir=$data_root/babylm_full_bpe
#        ;;
#    3)
#        output_dir=$output_root/out-babylm_full_bpe_8k-4x4-nomask-1709513018
#        data_dir=$data_root/babylm_full_bpe_8k
#        ;;
#    4)
#        output_dir=$output_root/out-babylm_full_bpe_8k-2x2-nomask-1709513019
#        data_dir=$data_root/babylm_full_bpe_8k
#        ;;
#    5)
#        output_dir=$output_root/out-babylm_full_bpe-4x4-nomask-5444724
#        data_dir=$data_root/babylm_full_bpe
#        ;;
#    6)
#        output_dir=$output_root/out-babylm_full_bpe-8x8-nomask-5492054
#        data_dir=$data_root/babylm_full_bpe
#        ;;
#    7)
#        output_dir=$output_root/out-babylm_full_bpe-6x6-nomask-5492134
#        data_dir=$data_root/babylm_full_bpe
#        ;;
#    8)
#        output_dir=$output_root/out-babylm_full_bpe_8k-6x6-nomask-5496427
#        data_dir=$data_root/babylm_full_bpe_8k
#        ;;
#    9)
#        output_dir=$output_root/out-babylm_full_bpe_8k-8x8-nomask-5496426
#        data_dir=$data_root/babylm_full_bpe_8k
#        ;;
#
#esac

#python babylm_eval.py --model_type=nanogpt --output_dir /home/athamma/repo/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-2x2-nomask-1709513019 --data_dir /home/athamma/repo/ss-llm/nanoGPT/data/babylm_full_bpe --tasks all
#python babylm_eval.py --model_type=nanogpt --output_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-2x2-nomask-1709513019 --data_dir /home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data/babylm_full_bpe --tasks  blimp_test



python babylm_eval.py --model_type=nanogpt --output_dir /home/athamma/repo/ss-llm/nanoGPT/output_dump/out-babylm_full_bpe_8k-6x6-mask_e010-5734465_s1337 --data_dir /home/athamma/repo/ss-llm/nanoGPT/data/babylm_full_bpe_8k --tasks all
