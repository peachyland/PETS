cd /home/rjie/projects/controlability/PETS/reasoning
MY_CMD="./launch.sh --model-dir Qwen/Qwen3-8B --task aime25 --n 64 --dp-size 2 --out ./results/qwen_new_aime25_supp.jsonl "
CUDA_VISIBLE_DEVICES='0,1' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:51 \\n Python_command: \\n ./launch.sh --model-dir Qwen/Qwen3-8B --task aime25 --n 64 --dp-size 2 --out ./results/qwen_new_aime25_supp.jsonl \\n " | mail -s "[Done] grandriver 11180678" renjie2179@outlook.com
else
echo -e "grandriver JobID:51 \\n Python_command: \\n ./launch.sh --model-dir Qwen/Qwen3-8B --task aime25 --n 64 --dp-size 2 --out ./results/qwen_new_aime25_supp.jsonl \\n " | mail -s "[Fail] grandriver 11180678" renjie2179@outlook.com
fi
