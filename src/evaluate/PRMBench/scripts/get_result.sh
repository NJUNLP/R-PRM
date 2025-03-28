# Your Data File
#!/bin/bash

# 进入当前目录
NOW_DIR=$(cd `dirname $0`; pwd)
cd $NOW_DIR
echo "当前代码运行所在的路径为： $NOW_DIR"
SCORE_PATH=$1
PRMBENCH_DATASET_PATH=$2
threshold=$3
MODEL_NAME="GenPRM"

python ../code/collate.py \
    --score_path $SCORE_PATH \
    --prm_dataset_path $PRMBENCH_DATASET_PATH \
    --threshold $threshold

python ../code/evaluate.py \
    --model_name=$MODEL_NAME

python ../code/get_result.py \
    --model_name=$MODEL_NAME \