#!/bin/bash

TASK=aime25

for ((i=8; i<=8; i++))
do
  echo $i
  ./launch.sh --model-dir Qwen/Qwen3-8B --task ${TASK} --n 64 --dp-size 4 --out ./results/qwen3_${TASK}_64_${i}.jsonl --seed $i

done