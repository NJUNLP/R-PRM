SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"  # 进入脚本所在目录

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"  # 进入脚本所在目录

# 参数解释

# 检查参数数量
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <model_path> <base_model_name> <prompt_path> [N] [temperature]"
  exit 1
fi

# export PROMPT_PATH=$(realpath )
#export N=10
# export TEMPERATURE=0.95
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v0  GenRM_Qwen25_MATH_7B_v5_2Mix_v0 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v1  GenRM_Qwen25_MATH_7B_v5_2Mix_v1 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v2  GenRM_Qwen25_MATH_7B_v5_2Mix_v2 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v0  GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v0 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v1  GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v1 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt
# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v2  GenRM_Qwen25_MATH_7B_v5_2Mix_v2_dpo_v2 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwen.txt

# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_v5_2Mix_v0  GenRM_Qwen25_7B_v5_2Mix_v0 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt


# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3  GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt

# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_v5_2Mix_v3  GenRM_Qwen25_7B_v5_2Mix_v3 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt

# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov0  GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov0 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt

# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov1  GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov1 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt

# bash eval-ProcessBench.sh /opt/tiger/Trained/GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov1-checkpoint200  GenRM_Qwen25_7B_MATHBase_v5_2Mix_v3_dpov1-checkpoint200 /opt/tiger/MAPO-Ultra/MCTS/GenRM/prompts/GenPRM/GenPRM-evaluate-new3-qwensys.txt



# 需要传递的三个必需参数
export MODEL_PATH=$1
MODEL_BASENAME=$(basename "$MODEL_PATH")
BASE_MODEL=$2
export PROMPT_PATH=$3
threshold=$4

# 设置默认参数
DEFAULT_N=10
DEFAULT_TEMPERATURE=0.95

# 使用传递的参数或默认值
export N=${5:-$DEFAULT_N}
export TEMPERATURE=${6:-$DEFAULT_TEMPERATURE}
mkdir -p "../../output/ProcessBench/GenPRM/${BASE_MODEL}"
OUTPUT_DIR_GLOBAL=$(realpath "../../output/ProcessBench/GenPRM/${BASE_MODEL}")

UTILS_PATH=$(realpath "../../utils")
OPERATE_PATH=$(realpath "../../evaluate/ProcessBench")

declare -A DATASETS=(
  ["gsm8k"]="../../datasets/ProcessBench/gsm8k_filtered.json"
  ["math"]="../../datasets/ProcessBench/math_filtered.json"
  ["omnimath"]="../../datasets/ProcessBench/omnimath_filtered.json"
  ["olympiadbench"]="../../datasets/ProcessBench/olympiadbench_filtered.json"
)

declare -A DATASETS_ORIGIN=(
  ["gsm8k"]="../../datasets/ProcessBench/gsm8k.json"
  ["math"]="../../datasets/ProcessBench/math.json"
  ["omnimath"]="../../datasets/ProcessBench/omnimath.json"
  ["olympiadbench"]="../../datasets/ProcessBench/olympiadbench.json"
)
# declare -A DATASETS=(
#   ["gsm8k-part"]="../../datasets/ProcessBench/gsm8k-part_filtered.json"
#   ["math-part"]="../../datasets/ProcessBench/math-part_filtered.json"
#   ["omnimath-part"]="../../datasets/ProcessBench/omnimath-part_filtered.json"
#   ["olympiadbench-part"]="../../datasets/ProcessBench/olympiadbench-part_filtered.json"
# )

select_free_gpus() {
  # 设置显存占用阈值（MiB）
  local threshold=500
  local gpu_info
  local free_gpus

  # 查询 GPU 的 index 和已用显存（以 MiB 为单位）
  gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  # 从 GPU_INFO 中筛选出显存使用低于阈值的 GPU
  free_gpus=$(echo "$gpu_info" | \
    awk -F "," -v thresh="$threshold" '{ gsub(/ /, "", $2); if ($2+0 < thresh) print $1 }' | \
    tr '\n' ',' | sed 's/,$//')

  # 如果没有空闲 GPU，则默认使用所有 GPU
  if [ -z "$free_gpus" ]; then
    free_gpus=$(echo "$gpu_info" | awk -F "," '{ print $1 }' | tr '\n' ',' | sed 's/,$//')
  fi

  # 仅输出空闲 GPU 索引，不带调试信息
  echo "$free_gpus"
}

# 自动检测GPU并设置 CUDA_VISIBLE_DEVICES 环境变量
export CUDA_VISIBLE_DEVICES=$(select_free_gpus)

# 输出使用的 GPU
echo "----------------使用的 GPU: $CUDA_VISIBLE_DEVICES"



# 评分函数
get_score() {
    local dir_path=$1
    local data_name=$2

    echo "Scoring results for $data_name..."
    rm -f "$dir_path/response.jsonl" "$dir_path/final_result.json"
    python "$UTILS_PATH/merge-file.py" "$dir_path"
    python "$OPERATE_PATH/return-to-origin.py" --data_name "$data_name" --response_path "$dir_path/response.jsonl"
    python "$OPERATE_PATH/get-GenPRM-Score.py" --input_path "$dir_path/final_result.json"
    origin_dataset_path=$(realpath "${DATASETS_ORIGIN[$data_name]}")
    python "$OPERATE_PATH/get-error-accuracy-single.py" $origin_dataset_path "$dir_path/final_result.json"
    python "$OPERATE_PATH/plt-all-score.py" --input_path "$dir_path/final_result.json" --output_path "$dir_path/output/{$data_name}-different-threshold.png" --threshold $threshold
}

# 数据处理主函数
process_dataset() {
  local dataset_name=$1
  local data_path=$2

  local dataset_name_upper=$(echo "$dataset_name" | tr '[:lower:]' '[:upper:]')
  local output_dir_cot="${OUTPUT_DIR_GLOBAL}/CoT/${MODEL_BASENAME}-${dataset_name_upper}-${TEMPERATURE}"
  mkdir -p "$output_dir_cot"

  echo "Processing dataset: $dataset_name"

  # 设置环境变量
  export DATASET_NAME="$dataset_name-process-bench"
  export MODEL_PATH="$MODEL_PATH"
  local basename=$(basename "$MODEL_PATH")
  export OUTPUT_DIR="$output_dir_cot"
  export DATA_PATH="$data_path"
  export RESPONSE_PATH="${output_dir_cot}/response.jsonl"

  echo "Generating responses..."
  bash ../../evaluate/GenPRM/scripts/generate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES" 2
  python $UTILS_PATH/merge-file.py "$output_dir_cot"
  echo "Evaluating responses..."
  bash ../../evaluate/GenPRM/scripts/evaluate-PRM-CoT.sh "$CUDA_VISIBLE_DEVICES" 1


  get_score "${output_dir_cot}/${MODEL_BASENAME}-${dataset_name_upper}-PROCESS-BENCH-PRM-Rewards-${TEMPERATURE}" "$dataset_name"
}

declare -A DATASETS=(
  ["gsm8k"]="../../datasets/ProcessBench/gsm8k_filtered.json"
  ["math"]="../../datasets/ProcessBench/math_filtered.json"
  ["omnimath"]="../../datasets/ProcessBench/omnimath_filtered.json"
  ["olympiadbench"]="../../datasets/ProcessBench/olympiadbench_filtered.json"
)

# 定义明确的遍历顺序
ORDERED_DATASETS=("gsm8k" "math" "omnimath" "olympiadbench")

for dataset_name in "${ORDERED_DATASETS[@]}"; do
  data_path=$(realpath "${DATASETS[$dataset_name]}")
  process_dataset "$dataset_name" "$data_path"
done

echo "All datasets processed and evaluated."

