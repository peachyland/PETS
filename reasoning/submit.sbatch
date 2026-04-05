#!/bin/bash
#SBATCH -J lcb
#SBATCH -p mit_normal_gpu
#SBATCH -t 48:00:00
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH -o logfile/%j.out
#SBATCH -e logfile/%j.err

set -euo pipefail

# module purge  # 可选

# source /data/scratch/rjie/init2.bash
source /home/rjie/.bashrc
export HF_HOME=/home/rjie/orcd/scratch/.cache

conda activate lcb

# srun --account=csail --partition=csail-shared-h200 --qos=shared-if-available --time=24:00:00 --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=128G --pty /bin/bash

echo "python: $(which python)"

TASK_NAME="aime25" # aime24 aime25 brumo hmmt
# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME="mistralai/Ministral-3-8B-Reasoning-2512"
OUT_PATH="./results/qwen_new_${TASK_NAME}_300.jsonl"

MY_CMD="./launch.sh --model-dir ${MODEL_NAME} --task ${TASK_NAME} --n 64 --dp-size 2 --out ${OUT_PATH} --port 8008"

echo $MY_CMD

date >>./history_job.log
echo ${SLURM_JOB_ID}>>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log

$MY_CMD
