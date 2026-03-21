cd /data/scratch/rjie/tts/PETS/reasoning
MY_CMD="python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime24_results_2_44.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 2 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 16 "
CUDA_VISIBLE_DEVICES='2' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:44 \n Python_command: \n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime24_results_2_44.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 2 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 16 \n " | mail -s "[Done] grandriver 574111" renjie2179@outlook.com
else
echo -e "grandriver JobID:44 \n Python_command: \n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/aime24_results_2_44.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 2 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 16 \n " | mail -s "[Fail] grandriver 574111" renjie2179@outlook.com
fi
