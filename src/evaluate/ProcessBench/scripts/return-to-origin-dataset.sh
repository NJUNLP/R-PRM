#!/bin/bash
# source环境
# CONDA_BASE=$(conda info --base)
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# source "/data1/software/anaconda3/etc/profile.d/conda.sh"

# conda activate old-vllm

SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"

DATASET_PATH1_DIR=$1
DATASET_PATH2=$2

python ../../../utils/merge-file.py "$DATASET_PATH1_DIR"
python ../70B-return-to-origin-file.py "$DATASET_PATH2" "$DATASET_PATH1_DIR/response.jsonl"
