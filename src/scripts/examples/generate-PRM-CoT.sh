#!/bin/bash
# source环境

source "/data1/software/anaconda3/etc/profile.d/conda.sh"

conda activate old-vllm

SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"
MODEL_PATH="/data2/model/DeepSeek-R1-Distill-Llama-70B"
DATA_PATH="/home/adminad/GithubRepos/prm800k/prm800k/data/phase_all_flat_sample_760k-error-set-except-149k.jsonl"
# DATA_LENGTH=764607
# 每4卡启动一个程序
# DATA_PIECE=？ # 这个地方就是每个程序使用多少的数据，通过调整data_begin和data_end来实现


# data_begin默认为0
# data_end默认为数据集大小 CUDA_VISIBLE_DEVICES=0,1,2,3  
python ../../utils/inference.py \
--temperature 0.4 \
--gpu_memory_utilization 0.95 \
--tensor_number 8 \
--max_token 8192 \
--n 3 \
--data_begin 0 \
--data_end 1 \
--dtype bfloat16 \
--model_path $MODEL_PATH \
--prompt_path ../../prompts/GenPRM/GenPRM-generate-verify-CoT-new3.json \
--data_path $DATA_PATH    \
--write_path ../../output/Gen-Place/DeepSeek-R1-Distill-Llama-70B/test \
--prompt_type PRM-CoT
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ../../utils/inference.py \
# --temperature 0.4 \
# --gpu_memory_utilization 0.95 \
# --tensor_number 4 \
# --n 3 \
# --data_begin 25000 \
# --data_end 50000 \
# --dtype bfloat16 \
# --model_path $MODEL_PATH \
# --prompt_path ../../prompts/GenPRM/GenPRM-generate-verify-CoT-new3.json \
# --data_path $DATA_PATH    \
# --write_path ../../output/Gen-Place/Llama-3.3-70B-Instruct/phase_all_flat_sample_760k-error-set-except-149k_GenPRM-generate-verify-CoT-new3 \
# --prompt_type PRM-CoT

wait
