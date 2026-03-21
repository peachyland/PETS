#!/bin/bash

TASK=hmmt

for ((i=1; i<=8; i++))
do
  echo $i
  CUDA_VISIBLE_DEVICES='3,4,5' ./launch.sh --model-dir mistralai/Ministral-3-8B-Reasoning-2512 --task ${TASK} --n 64 --dp-size 3 --out ./results/mistral_${TASK}_64_${i}.jsonl --seed $i

done