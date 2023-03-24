#!/bin/bash
#SBATCH --job-name="LegalLM"
#SBATCH --mail-user=jniklaus@stanford.edu
#SBATCH --mail-type=end,fail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH -C GPU_MEM:80GB
#SBATCH --gpus=4
#SBATCH --partition=deho

cd /home/groups/deho/jniklaus/LegalLM
conda activate llama

export HF_DATASETS_CACHE=$SCRATCH/legallm/cache/datasets TRANSFORMERS_CACHE=$SCRATCH/legallm/cache/models
export WANDB_PROJECT=LegalLM

bash run_training.sh


# IMPORTANT:
# Run with                  sbatch run_hpc_job.sh
# check with                squeue --user=jniklaus --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --partition=deho --gpus=1 -C GPU_SKU:A100_SXM4 --mem=128G --time=02:00:00 --pty /bin/bash
# run interactive job with  srun --partition=gpu --gpus=1 -C GPU_SKU:V100_SXM2 --mem=64G --time=02:00:00 --pty /bin/bash
