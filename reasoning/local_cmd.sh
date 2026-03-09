#!/usr/bin/env bash


MY_CMD="python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 64 --max_new_tokens 300 --temperature 0.7 --top_p 0.85 --group_size 10 --group_id 0"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log

CUDA_VISIBLE_DEVICES='0' $MY_CMD
