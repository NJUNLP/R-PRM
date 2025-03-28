#!/bin/bash
SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"

DATASET_PATH1_DIR=$1
DATASET_PATH2=$2
threshold=$3

python ../../../utils/merge-file.py "$DATASET_PATH1_DIR"
python ../code/return-to-origin-dataset-PRMBench.py \
--response_path "$DATASET_PATH1_DIR/response.jsonl" \
--dataset_path "$DATASET_PATH2" \
--output_path "$DATASET_PATH1_DIR/merged_dataset.jsonl"

prm_path=$(realpath "../../../datasets/PRMBench_Preview/prmbench_preview.jsonl")
bash ./get_result.sh \
"$DATASET_PATH1_DIR/merged_dataset.jsonl" \
"$prm_path" \
"$threshold" 