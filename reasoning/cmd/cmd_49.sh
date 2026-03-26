cd /data/scratch/rjie/tts/PETS/reasoning
MY_CMD="./launch.sh --model-dir deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --task brumo --n 64 --dp-size 2 --out ./results/deepseek_supp_brumo.jsonl "
CUDA_VISIBLE_DEVICES='0,1' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:49 \n Python_command: \n ./launch.sh --model-dir deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --task brumo --n 64 --dp-size 2 --out ./results/deepseek_supp_brumo.jsonl \n " | mail -s "[Done] grandriver 574683" renjie2179@outlook.com
else
echo -e "grandriver JobID:49 \n Python_command: \n ./launch.sh --model-dir deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --task brumo --n 64 --dp-size 2 --out ./results/deepseek_supp_brumo.jsonl \n " | mail -s "[Fail] grandriver 574683" renjie2179@outlook.com
fi
