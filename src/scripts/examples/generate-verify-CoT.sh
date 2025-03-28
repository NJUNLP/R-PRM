#!/bin/bash
# source环境
# CONDA_BASE=$(conda info --base)
# export CUDA_VISIBLE_DEVICES=4,5,6,7

source "/data1/software/anaconda3/etc/profile.d/conda.sh"

conda activate old-vllm

SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"

export DATASET_NAME=math
export MODEL_PATH=/data1/ckpts/GenRM/meta-llama-3.1-8b-instruct/merge-model/GenRM-CoT-76K-MATH-checkpoint3564
export N=10
export OUTPUT_DIR=/home/adminad/MCTS-RM/GenRM/evaluate/GenRM-CoT/output/result/MATH-New
export TEMPERATURE=0.9
export GPU_MEMORY_UTILIZATION=1

bash /home/adminad/MCTS-RM/GenRM/evaluate/GenRM-CoT/scripts/generate-verify-CoT-MATH.sh 4,5,6,7 2

bash /home/adminad/MCTS-RM/GenRM/scripts/examples/evaluate-verify-CoT.sh

