cd /home/rjie/projects/controlability/PETS/reasoning
MY_CMD="python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 0 --temperature 1.0 --top_p 0.9 "
CUDA_VISIBLE_DEVICES='0' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:1 \\n Python_command: \\n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 0 --temperature 1.0 --top_p 0.9 \\n " | mail -s "[Done] grandriver 10313364" renjie2179@outlook.com
else
echo -e "grandriver JobID:1 \\n Python_command: \\n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 --max_new_tokens 30000 --group_size 10 --group_id 0 --temperature 1.0 --top_p 0.9 \\n " | mail -s "[Fail] grandriver 10313364" renjie2179@outlook.com
fi
