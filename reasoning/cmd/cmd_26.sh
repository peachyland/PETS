cd /data/scratch/rjie/tts/PETS/reasoning
MY_CMD="python hmmt.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/hmmt_results_0.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 0 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 32 "
CUDA_VISIBLE_DEVICES='6' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:26 \n Python_command: \n python hmmt.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/hmmt_results_0.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 0 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 32 \n " | mail -s "[Done] grandriver 494504" renjie2179@outlook.com
else
echo -e "grandriver JobID:26 \n Python_command: \n python hmmt.py --model_name deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --out ./results/hmmt_results_0.jsonl --n 128 --max_new_tokens 30000 --group_size 15 --group_id 0 --temperature 1.0 --top_p 0.9 --seed 42 --batch_size 32 \n " | mail -s "[Fail] grandriver 494504" renjie2179@outlook.com
fi
