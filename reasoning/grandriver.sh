JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log

GPU_ID="2"

GROUP_ID=2
DATA_SET="aime24" # aime24 aime25 brumo hmmt
MODEL_NAME="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# DATA_PATH="HuggingFaceH4/aime_2024"
OUT_PATH="./results/${DATA_SET}_results_${GROUP_ID}_${JOB_ID}.jsonl"
N=128
MAX_NEW_TOKENS=30000
GROUP_SIZE=10
TEMPERATURE=1.0
TOP_P=0.9
SEED=42
BATCH_SIZE=16

MY_CMD="python ${DATA_SET}.py --model_name ${MODEL_NAME} --out ${OUT_PATH} --n ${N} --max_new_tokens ${MAX_NEW_TOKENS} --group_size ${GROUP_SIZE} --group_id ${GROUP_ID} --temperature ${TEMPERATURE} --top_p ${TOP_P} --seed ${SEED} --batch_size ${BATCH_SIZE}"

MY_ROOT_PATH=`pwd`

echo "cd ${MY_ROOT_PATH}" > ./cmd/cmd_${JOB_ID}.sh
echo "MY_CMD=\"${MY_CMD} \"" >> ./cmd/cmd_${JOB_ID}.sh
echo "CUDA_VISIBLE_DEVICES='${GPU_ID}' \${MY_CMD}" >> ./cmd/cmd_${JOB_ID}.sh
echo "if [ \$? -eq 0 ];then" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Done] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "else" >> ./cmd/cmd_${JOB_ID}.sh
echo "echo -e \"grandriver JobID:${JOB_ID} \\\n Python_command: \\\n ${MY_CMD} \\\n \" | mail -s \"[Fail] grandriver ${SLURM_JOB_ID}\" renjie2179@outlook.com" >> ./cmd/cmd_${JOB_ID}.sh
echo "fi" >> ./cmd/cmd_${JOB_ID}.sh

nohup sh ./cmd/cmd_${JOB_ID}.sh >./logfile/${JOB_ID}.log 2>./logfile/${JOB_ID}.err &

echo "GPU_ID=${GPU_ID}"
echo $MY_CMD

date >>./history_job.log
echo ${JOB_ID}>>./history_job.log
echo "GPU_ID=${GPU_ID}">>./history_job.log
echo ${MY_CMD}>>./history_job.log
echo "---------------------------------------------------------------" >>./history_job.log