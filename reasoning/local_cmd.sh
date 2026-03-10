#!/usr/bin/env bash


MY_CMD="python hmmt.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --n 4 --max_new_tokens 300 --temperature 1.0 --top_p 0.9 --group_size 10 --group_id 0 --batch_size 2 --seed 42"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log

CUDA_VISIBLE_DEVICES='0' $MY_CMD
