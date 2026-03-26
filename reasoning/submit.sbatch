#!/bin/bash
#SBATCH -J lcb
#SBATCH -p csail-shared-h200
#SBATCH -t 24:00:00
#SBATCH --account=csail
#SBATCH --qos=shared-if-available
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH -o logfile/%j.out
#SBATCH -e logfile/%j.err

set -euo pipefail
mkdir -p logfile

# module purge  # 可选

source /data/scratch/rjie/init.bash
conda activate lcb

# srun --account=csail --partition=csail-shared-h200 --qos=shared-if-available --time=24:00:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=128G --pty /bin/bash

echo "python: $(which python)"

TASK_NAME="hmmt"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUT_PATH="./results/deepseek_new_${TASK_NAME}_5.jsonl"

MY_CMD="HF_HOME=/data/scratch/rjie/.cache/huggingface ./launch.sh --model-dir ${MODEL_NAME} --task ${TASK_NAME} --n 64 --dp-size 4 --out ${OUT_PATH} --port 8004"

echo $MY_CMD

date >>./history_job.log
echo ${SLURM_JOB_ID}>>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log

$MY_CMD
