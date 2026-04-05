cd /home/rjie/projects/controlability/PETS/reasoning
MY_CMD="./launch.sh --model-dir Qwen/Qwen3-8B --task brumo --n 64 --dp-size 2 --out ./results/qwen_new_brumo.jsonl "
CUDA_VISIBLE_DEVICES='0,1' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:50 \\n Python_command: \\n ./launch.sh --model-dir Qwen/Qwen3-8B --task brumo --n 64 --dp-size 2 --out ./results/qwen_new_brumo.jsonl \\n " | mail -s "[Done] grandriver 11031334" renjie2179@outlook.com
else
echo -e "grandriver JobID:50 \\n Python_command: \\n ./launch.sh --model-dir Qwen/Qwen3-8B --task brumo --n 64 --dp-size 2 --out ./results/qwen_new_brumo.jsonl \\n " | mail -s "[Fail] grandriver 11031334" renjie2179@outlook.com
fi
