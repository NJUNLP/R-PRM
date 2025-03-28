#!/bin/bash

# 获取当前脚本目录
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
echo "当前目录: $(pwd)"

# 自动检测空闲 GPU
select_free_gpus() {
  local threshold=500
  local gpu_info free_gpus
  gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  free_gpus=$(echo "$gpu_info" | awk -F "," -v thresh="$threshold" '{ gsub(/ /, "", $2); if ($2 < thresh) print $1 }' | tr '\n' ',' | sed 's/,$//')

  if [ -z "$free_gpus" ]; then
    echo "未找到空闲 GPU！默认使用所有 GPU。"
    free_gpus=$(echo "$gpu_info" | awk -F "," '{print $1}' | tr '\n' ',' | sed 's/,$//')
  fi
  echo "$free_gpus"
}

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=$(select_free_gpus)
echo "自动检测到空闲 GPU ID: $CUDA_VISIBLE_DEVICES"

# 全局参数
MODEL_TYPE="qwen-2.5-math-instruct"
TEMPERATURE=0.95
NUMBER=10
# DATA_PATH=$(realpath "../../datasets/MATH/Math-OAI-qwen-math-instruct-step.jsonl") # 数据集上传到了huggingface: https://huggingface.co/datasets/masterLan/MATH500-qwen-math-instruct-step
DATA_PATH=$1
# export PROMPT_PATH=$(realpath "../../prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt")
export PROMPT_PATH=$4

# 模型路径和数据集配置
# MODEL_PATH="/data2/model/GenRM_Qwen25_MATH_7B_v3All_DPO"
MODEL_PATH=$2
DATASET_NAME=$3

# 通用生成和评估函数
generate_and_evaluate() {
  local model_path=$1 dataset_name=$2

  # 设置环境变量
  export DATASET_NAME="$dataset_name"
  export TEMPERATURE="$TEMPERATURE"
  export N="$NUMBER"
  export MODEL_PATH="$model_path"
  export PROMPT_PATH="$PROMPT_PATH"
  export DATA_PATH="$DATA_PATH"

  # 生成输出路径
  # local basename=$(basename "$(dirname "$model_path")")
  local basename=$(basename "$model_path")
  echo "basename_origin: $basename_origin"

  local dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')
  local output_name="${basename}-${dataset_name_upper}-${NUMBER}-${TEMPERATURE}"
  mkdir -p "../../evaluate/GenPRM/output/result/${dataset_name_upper}/${MODEL_TYPE}/${output_name}"
  export OUTPUT_DIR=$(realpath "../../evaluate/GenPRM/output/result/${dataset_name_upper}/${MODEL_TYPE}/${output_name}")
  # 生成数据
  echo "开始生成数据：$dataset_name，输出目录：$OUTPUT_DIR"
  bash ../../evaluate/GenPRM/scripts/generate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES" 2
  python ../../utils/merge-file.py "$OUTPUT_DIR"

  # 评估数据
  export RESPONSE_PATH="${OUTPUT_DIR}/response.jsonl"
  mkdir -p "../../evaluate/GenPRM/output/CoT/${MODEL_TYPE}/${dataset_name_upper}"
  export OUTPUT_DIR=$(realpath "../../evaluate/GenPRM/output/CoT/${MODEL_TYPE}/${dataset_name_upper}")
  echo "开始评估数据：$dataset_name，输出目录：$OUTPUT_DIR"
  bash ../../evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES" 1

}

# 执行生成和评估
generate_and_evaluate "$MODEL_PATH" "$DATASET_NAME"
