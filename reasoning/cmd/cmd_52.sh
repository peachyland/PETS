cd /home/rjie/projects/controlability/PETS/reasoning
MY_CMD="./launch.sh --model-dir mistralai/Ministral-3-8B-Reasoning-2512 --task hmmt --n 64 --dp-size 2 --out ./results/mistral_new_hmmt_300.jsonl --port 8003 "
CUDA_VISIBLE_DEVICES='0,1' ${MY_CMD}
if [ $? -eq 0 ];then
echo -e "grandriver JobID:52 \\n Python_command: \\n ./launch.sh --model-dir mistralai/Ministral-3-8B-Reasoning-2512 --task hmmt --n 64 --dp-size 2 --out ./results/mistral_new_hmmt_300.jsonl --port 8003 \\n " | mail -s "[Done] grandriver 11256355" renjie2179@outlook.com
else
echo -e "grandriver JobID:52 \\n Python_command: \\n ./launch.sh --model-dir mistralai/Ministral-3-8B-Reasoning-2512 --task hmmt --n 64 --dp-size 2 --out ./results/mistral_new_hmmt_300.jsonl --port 8003 \\n " | mail -s "[Fail] grandriver 11256355" renjie2179@outlook.com
fi
