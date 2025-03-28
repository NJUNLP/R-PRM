#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$0")
# 进入脚本所在的目录
cd "$SCRIPT_DIR"
# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

model_name=$1
model_path=""
chat_template="../chat-template/qwen2.5-instruct.jinja"
# curl http://localhost:1234/v1/models  # 直接测试接口是否响应
CUDA_VISIBLE_DEVICES=0 vllm serve $model_path \
    --chat-template $chat_template \
    --port 8997 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --swap-space 0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --disable-log-stats \
    --port 1234 &
CUDA_VISIBLE_DEVICES=4 vllm serve $model_path \
    --chat-template $chat_template \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --swap-space 0 \
    --disable-log-stats \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 1235 
