#!/usr/bin/env bash


MY_CMD="python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 4"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log

CUDA_VISIBLE_DEVICES='0' $MY_CMD
