cd /home/rjie/projects/controlability/PETS/reasoning
MY_CMD="python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 "
CUDA_VISIBLE_DEVICES='0' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID: \\n Python_command: \\n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 \\n " | mail -s "[Done] grandriver 10313364" renjie2179@outlook.com
else
echo -e "grandriver JobID: \\n Python_command: \\n python aime24.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --data_path HuggingFaceH4/aime_2024 --out ./results/aime24_results.jsonl --n 128 \\n " | mail -s "[Fail] grandriver 10313364" renjie2179@outlook.com
fi
