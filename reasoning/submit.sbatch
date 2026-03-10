#!/bin/bash
#SBATCH -J lcb
#SBATCH -p mit_normal
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o logfile/%j.out
#SBATCH -e logfile/%j.err

set -euo pipefail
mkdir -p logfile

module purge  # 可选

source /home/rjie/miniconda3/etc/profile.d/conda.sh
conda activate lcb

echo "python: $(which python)"
python -V

MODEL_NAME="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DATA_PATH="HuggingFaceH4/aime_2024"
OUT_PATH="./results/aime24_results.jsonl"
N=64
MAX_NEW_TOKENS=30000
GROUP_SIZE=10
GROUP_ID=0
TEMPERATURE=1.0
TOP_P=0.9

MY_CMD="python aime24.py --model_name ${MODEL_NAME} --data_path ${DATA_PATH} --out ${OUT_PATH} --n ${N} --max_new_tokens ${MAX_NEW_TOKENS} --group_size ${GROUP_SIZE} --group_id ${GROUP_ID} --temperature ${TEMPERATURE} --top_p ${TOP_P}"

echo $MY_CMD

date >>./history_job.log
echo ${SLURM_JOB_ID}>>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log

$MY_CMD
