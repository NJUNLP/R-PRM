# # #!/bin/bash
# # # source环境
# # # CONDA_BASE=$(conda info --base)
# # # export CUDA_VISIBLE_DEVICES=4,5,6,7

# # source "/data1/software/anaconda3/etc/profile.d/conda.sh"



# SCRIPT_DIR=$(dirname "$0")

# # 进入脚本所在的目录
# cd "$SCRIPT_DIR"

# # # 现在你可以在这个目录下执行其他操作
# # echo "当前目录: $(pwd)"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 64 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/need-regen-aime24.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64 \
# --prompt_type generation_answer 

# CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 64 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-3.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-5 \
# --prompt_type R1-Mistake

# CUDA_VISIBLE_DEVICES=3,5,6,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 64 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-3.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-6 \
# --prompt_type generation_answer 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 64 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake-qwen.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-2.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-GenPRM-24 \
# --prompt_type R1-Mistake-Qwen
# CUDA_VISIBLE_DEVICES=0,2 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 2 \
# --n 1 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake-qwen.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-2.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-GenPRM-24-6 \
# --prompt_type R1-Mistake-Qwen

# CUDA_VISIBLE_DEVICES=0,2 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 2 \
# --n 1 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-2.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-GenPRM-24-7 \
# --prompt_type R1-Mistake

# CUDA_VISIBLE_DEVICES=0,2 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 2 \
# --n 1 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake-new2.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-2.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-GenPRM-24-11 \
# --prompt_type R1-Mistake

# CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 64 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1-continue.txt \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Reward-Guide-Search-Greedy/GenRM_Qwen25_MATH_7B_v3All_DPO/AMC23-DeepSeek-R1-Distill-Qwen-7B-new/result-0-40.json \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AMC23-R1 \
# --prompt_type continue-generate

CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
--temperature 0.6 \
--gpu_memory_utilization 0.9 \
--tensor_number 4 \
--n 4 \
--dtype bfloat16 \
--model_path /home/nfs05/model/Qwen2.5-7B-Instruct \
--max_token 2048 \
--prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-zero-shot.json \
--data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/datasets/MATH/Math-OAI.jsonl \
--write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/Qwen2.5-7B-Instruct/MATH500-0.6-16-new-prompt \
--prompt_type generation_answer 
# CUDA_VISIBLE_DEVICES=1,2,3,4 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.9 \
# --tensor_number 4 \
# --n 4 \
# --dtype bfloat16 \
# --model_path /home/nfs05/liujx/ckpts/merge/GenPRM-train \
# --max_token 4096 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1-new-llamafactory.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/datasets/MATH/Math-OAI.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/GenPRM-train/MATH500-0.6-16-new-prompt \
# --prompt_type generation_answer

# CUDA_VISIBLE_DEVICES=1,2,3,4 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 16 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 12000 \
# --data_begin 0 \
# --data_end 15 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1-new.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/datasets/AIME25/AIME25_train.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME25-0.6-64-new-prompt \
# --prompt_type generation_answer

# CUDA_VISIBLE_DEVICES=1,2,3,4 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 16 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 12000 \
# --data_begin 0 \
# --data_end 15 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/math-r1-new.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/datasets/AIME25/AIME25_train.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME25-0.6-64-new-prompt \
# --prompt_type generation_answer
# conda activate lm-evaluation
# bash /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/scripts/examples/evaluate-math-script.sh /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME25-0.6-64-new-prompt/DeepSeek-R1-Distill-Qwen-7B-response-0-15.jsonl 64 aime24 /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/datasets/AIME25/AIME25_train.jsonl



# CUDA_VISIBLE_DEVICES=5,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.6 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 2 \
# --n 1 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/DeepSeek-R1-Distill-Qwen-7B \
# --max_token 10000 \
# --data_begin 0 \
# --data_end 24 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/MATH/deepseek-r1-mistake.json \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/mistake-aime24-2.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64-Mistake-GenPRM-24-5 \
# --prompt_type R1-Mistake
# CUDA_VISIBLE_DEVICES=4,5,6,7 python /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/utils/inference.py \
# --temperature 0.95 \
# --gpu_memory_utilization 0.85 \
# --tensor_number 4 \
# --n 15 \
# --dtype bfloat16 \
# --model_path /home/nfs05/model/GenRM_Qwen25_MATH_7B_v3All_DPO \
# --max_token 8192 \
# --prompt_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt \
# --data_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/first_40_steps-correct.jsonl \
# --write_path /home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/ \
# --prompt_type GenPRM-CoT-Generate

# PYTHON_SCRIPT="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/evaluate/GenPRM/evaluate-PRM-CoT.py"
# DATA_NAME="r1-first-40-correct"
# MODEL_PATH="/home/nfs05/model/GenRM_Qwen25_MATH_7B_v3All_DPO"
# OUTPUT_PATH="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/evaluate-prm-correct"
# PROMPT_PATH="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt"
# RESPONSE_PATH="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/GenRM_Qwen25_MATH_7B_v3All_DPO-response-0-560.jsonl"
# DATA_PATH="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/output/Gen-Place/DeepSeek-R1-Distill-Qwen-7B/AIME24-0.6-64/first_40_steps-correct.jsonl"
# TEMPERATURE=0.95

# # Log directory and files
# # LOG_DIR="/home/nfs05/liujx/MCTS-RM-Newest/MCTS-RM/GenRM/logs"
# # mkdir -p "$LOG_DIR"

# # Command 1: GPU 4, 0-70 (original command)
# CUDA_VISIBLE_DEVICES=4 python "$PYTHON_SCRIPT" \
#     --data_name "$DATA_NAME" \
#     --begin 0 \
#     --end 140 \
#     --model_path "$MODEL_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --prompt_path "$PROMPT_PATH" \
#     --response_path "$RESPONSE_PATH" \
#     --data_path "$DATA_PATH" \
#     --temperature "$TEMPERATURE" &

# echo "Started evaluation on GPU 4 for range 0-70 (PID: $!)"

# # Command 2: GPU 5, 70-140
# CUDA_VISIBLE_DEVICES=5 python "$PYTHON_SCRIPT" \
#     --data_name "$DATA_NAME" \
#     --begin 140 \
#     --end 280 \
#     --model_path "$MODEL_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --prompt_path "$PROMPT_PATH" \
#     --response_path "$RESPONSE_PATH" \
#     --data_path "$DATA_PATH" \
#     --temperature "$TEMPERATURE" &

# echo "Started evaluation on GPU 5 for range 70-140 (PID: $!)"

# # Command 3: GPU 4, 140-210
# CUDA_VISIBLE_DEVICES=6 python "$PYTHON_SCRIPT" \
#     --data_name "$DATA_NAME" \
#     --begin 280 \
#     --end 420 \
#     --model_path "$MODEL_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --prompt_path "$PROMPT_PATH" \
#     --response_path "$RESPONSE_PATH" \
#     --data_path "$DATA_PATH" \
#     --temperature "$TEMPERATURE" &

# echo "Started evaluation on GPU 4 for range 140-210 (PID: $!)"

# # Command 4: GPU 5, 210-280
# CUDA_VISIBLE_DEVICES=7 python "$PYTHON_SCRIPT" \
#     --data_name "$DATA_NAME" \
#     --begin 420 \
#     --end 560 \
#     --model_path "$MODEL_PATH" \
#     --output_path "$OUTPUT_PATH" \
#     --prompt_path "$PROMPT_PATH" \
#     --response_path "$RESPONSE_PATH" \
#     --data_path "$DATA_PATH" \
#     --temperature "$TEMPERATURE" &

# echo "Started evaluation on GPU 5 for range 210-280 (PID: $!)"

# # Wait for all background processes to complete
# wait

# echo "All evaluations completed for 0-280 data points."
# # CUDA_VISIBLE_DEVICES=7 python "$PYTHON_SCRIPT" \
# #     --data_name "$DATA_NAME" \
# #     --begin 600 \
# #     --end 640 \
# #     --model_path "$MODEL_PATH" \
# #     --output_path "$OUTPUT_PATH" \
# #     --prompt_path "$PROMPT_PATH" \
# #     --response_path "$RESPONSE_PATH" \
# #     --data_path "$DATA_PATH" \
# #     --temperature "$TEMPERATURE"