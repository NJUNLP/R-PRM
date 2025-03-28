#!/bin/bash

select_free_gpus() {
  # 设置显存占用阈值（MiB）
  local threshold=500
  local gpu_info
  local free_gpus

  # 查询 GPU 的 index 和已用显存（以 MiB 为单位）
  gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  # 从 GPU_INFO 中筛选出显存使用低于阈值的 GPU，并按显存使用量排序
  free_gpus=$(echo "$gpu_info" | \
    awk -F "," -v thresh="$threshold" '{ gsub(/ /, "", $2); if ($2 < thresh) print $1, $2 }' | \
    sort -k2n | \
    awk '{print $1}' | \
    tr '\n' ',' | sed 's/,$//')

  # 如果没有空闲 GPU，则默认使用所有 GPU
  if [ -z "$free_gpus" ]; then
    echo "未找到空闲 GPU！默认使用所有 GPU。"
    free_gpus=$(echo "$gpu_info" | awk -F "," '{ print $1 }' | tr '\n' ',' | sed 's/,$//')
  fi

  echo "$free_gpus"
}

FREE_GPUS=$(select_free_gpus)
echo "自动检测到空闲 GPU ID: $FREE_GPUS"

# 将空闲 GPU id 设置到 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$FREE_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# ----------------------------
# 后续原本的逻辑
# ----------------------------
source "/data1/software/anaconda3/etc/profile.d/conda.sh"
conda activate old-vllm

NOW_DIR=$(cd $(dirname $0); pwd)
cd $NOW_DIR

echo "当前目录为：$NOW_DIR"
echo "启动PRM计算脚本..."
# 设置环境变量
# Testing CoT Script
# Testing CoT Script

export DATASET_NAME="math50"
export TEMPERATURE=0.7

# Test case 1: Model path 3:7
export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406"
export OUTPUT_DIR="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct"
export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/MATH50/GenPRM-90k-train-4:6-decontamination-10-0.7-temperature/response.jsonl"
export DATA_PATH="/home/adminad/MCTS-RM/GenRM/datasets/MATH/Math-OAI-step.jsonl"
export PROMPT_PATH="/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new2-llama3.txt"

# Run evaluation script
bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES"
mv /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/checkpoint-1406-GSM8K-PART-PRM-Rewards-${TEMPERATURE} /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/GenPRM-90k-train-3:7-decontamination-checkpoint-1406-GSM8K-PART-PRM-Rewards${TEMPERATURE}

# Test case 2: Model path 4:6
# export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406"
# export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/GSM8K-PART/meta-llama-3.1-8b-instruct/GenPRM-90k-train-4:6-decontamination-10-0.9-temperature/response.jsonl"
export TEMPERATURE=0.5
# export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406"
export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/MATH50/GenPRM-90k-train-4:6-decontamination-10-0.5-temperature/response.jsonl"
# Run evaluation script
bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES"
mv /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/checkpoint-1406-MATH50-PRM-Rewards-0.5 /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/GenPRM-90k-train-4:6-decontamination-checkpoint-1406-MATH50-PRM-Rewards-0.5

# Test case 3: Model path 5:5
# export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-5:5-decontamination/checkpoint-1406"
# export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/GSM8K-PART/meta-llama-3.1-8b-instruct/GenPRM-90k-train-5:5-decontamination-10-0.9-temperature/response.jsonl"
export TEMPERATURE=0.3
# export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-5:5-decontamination/checkpoint-1406"
export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/MATH50/GenPRM-90k-train-4:6-decontamination-10-0.3-temperature/response.jsonl"
# Run evaluation script
bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES"
mv /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/checkpoint-1406-GSM8K-PART-PRM-Rewards-0.3 /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/GenPRM-90k-train-5:5-decontamination-checkpoint-1406-MATH50-PRM-Rewards-0.3


bash /home/adminad/MCTS-RM/GenRM/scripts/examples/generate-PRM-CoT-Test.sh
# Test case 4: Model path 7:3
# export MODEL_PATH="/data2/ckpts/GenRM/meta-llama-3.1-8b-instruct/full/GenPRM-90k-train-7:3-decontamination/checkpoint-1406"
# export RESPONSE_PATH="/home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/GSM8K-PART/meta-llama-3.1-8b-instruct/GenPRM-90k-train-7:3-decontamination-10-0.9-temperature/response.jsonl"

# # Run evaluation script
# bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES"
# mv /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/checkpoint-1406-GSM8K-PART-PRM-Rewards-0.9 /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/meta-llama-3.1-8b-instruct/GenPRM-90k-train-7:3-decontamination-checkpoint-1406-GSM8K-PART-PRM-Rewards-0.9

# echo "PRM计算脚本执行完成！"


# function generate_qwen {
#   export DATASET_NAME=math-part
#   export TEMPERATURE=0.9
#   export N=10
#   export MODEL_PATH=$1
#   export PROMPT_PATH=/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new2-qwen.txt
#   export DATA_PATH=/home/adminad/MCTS-RM/GenRM/datasets/MATH/Math-OAI-50-step.jsonl
#   export OUTPUT_DIR=$2

#   bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/generate-PRM-CoT.sh $FREE_GPUS 2
#   python /home/adminad/MCTS-RM/GenRM/utils/merge-file.py ${OUTPUT_DIR}
# }

# generate_qwen /data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406 /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/MATH50/qwen-2.5-math-instruct/GenPRM-90k-train-4:6-decontamination-10-0.9-temperature


# function evaluate_qwen {
#   export DATASET_NAME=math50
#   export TEMPERATURE=0.9
#   export MODEL_PATH=$1
#   export OUTPUT_DIR=$2
#   export RESPONSE_PATH=$3
#   export DATA_PATH=/home/adminad/MCTS-RM/GenRM/datasets/MATH/Math-OAI-50-step.jsonl
#   export PROMPT_PATH=/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new2-qwen.txt

#   bash /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES"
#   # 获取basename，模型路径是/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406这里应该是GenPRM-90k-train-4:6-decontamination而不是checkpoint-1406
#   BASENAME=$(basename $(dirname $MODEL_PATH))
#   mv ${OUTPUT_DIR}/checkpoint-1406-MATH50-PRM-Rewards-0.9 ${OUTPUT_DIR}/${BASENAME}-MATH50-PRM-Rewards-0.9
# }

# evaluate_qwen /data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-90k-train-4:6-decontamination/checkpoint-1406 /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/CoT/qwen-2.5-math-instruct /home/adminad/MCTS-RM/GenRM/evaluate/GenPRM/output/result/MATH50/qwen-2.5-math-instruct/GenPRM-90k-train-4:6-decontamination-10-0.9-temperature/response.jsonl
