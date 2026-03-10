cd /data/scratch/rjie/tts/PETS/reasoning
MY_CMD="python aime25.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime25_results_1.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 1 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 64 "
CUDA_VISIBLE_DEVICES='3' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:7 \n Python_command: \n python aime25.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime25_results_1.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 1 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 64 \n " | mail -s "[Done] grandriver 494504" renjie2179@outlook.com
else
echo -e "grandriver JobID:7 \n Python_command: \n python aime25.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime25_results_1.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 1 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 64 \n " | mail -s "[Fail] grandriver 494504" renjie2179@outlook.com
fi
