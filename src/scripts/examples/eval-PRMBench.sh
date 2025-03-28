#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"  # 进入脚本所在目录

prompt_path=$(realpath "../../prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt")

bash ./evaluate-GenPRM.sh \
    /opt/tiger/MAPO-Ultra/MCTS/GenRM/datasets/prmbench_preview_steps-test.jsonl \
    /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_dpo_v5 \
    prmbench \
    $prompt_path