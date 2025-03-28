#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"  # 进入脚本所在目录

prompt_path=$(realpath "../../../prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt")
model_path=$1
dataset_path="/home/nfs05/liujx/MCTS-RM-Newest/prmbench_preview_steps-10.jsonl"
output_dir=$2
threshold=$3

bash ./util.sh \
    $dataset_path \
    $model_path \
    prmbench \
    $prompt_path \
    $output_dir \
    $threshold
