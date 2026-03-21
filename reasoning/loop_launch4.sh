#!/bin/bash

TASK=gpqa

for ((i=1; i<=8; i++))
do
  echo $i
  ./launch.sh --model-dir deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --task ${TASK} --n 64 --dp-size 5 --out ./results/deepseek_${TASK}_64_${i}.jsonl --seed $i

done