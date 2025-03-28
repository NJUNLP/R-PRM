#!/bin/bash

# 获取当前目录
NOW_DIR=$(cd `dirname $0`; pwd)
cd $NOW_DIR

# 输出当前目录
echo "当前目录为：$NOW_DIR"

# 检查是否提供了CUDA_VISIBLE_DEVICES参数
if [ -z "$1" ]; then
    echo "请通过参数提供CUDA_VISIBLE_DEVICES的值，例如：4,5,6,7"
    exit 1
fi

# 检查是否提供了GPU_PER_PROCESS参数
if [ -z "$2" ]; then
    echo "请通过参数提供每个程序使用的GPU数量，例如：2"
    exit 1
fi

# 获取参数并转换为数组
CUDA_VISIBLE_DEVICES_STR=$1
IFS=',' read -r -a CUDA_VISIBLE_DEVICES_ARRAY <<< "$CUDA_VISIBLE_DEVICES_STR"
GPU_PER_PROCESS=$2

# 获取总的GPU数量
TOTAL_GPU_COUNT=${#CUDA_VISIBLE_DEVICES_ARRAY[@]}
echo "检测到 GPU 数量: $TOTAL_GPU_COUNT"

# 计算可运行的并行程序数量
if (( TOTAL_GPU_COUNT % GPU_PER_PROCESS != 0 )); then
    echo "错误：总 GPU 数量 $TOTAL_GPU_COUNT 不能被每个程序使用的 GPU 数量 $GPU_PER_PROCESS 整除"
    exit 1
fi

PROCESS_COUNT=$((TOTAL_GPU_COUNT / GPU_PER_PROCESS))
echo "可以运行的并行程序数量: $PROCESS_COUNT"

# 检查必需的环境变量
REQUIRED_ENV_VARS=("DATASET_NAME" "MODEL_PATH" "OUTPUT_DIR" "RESPONSE_PATH" "DATA_PATH")
for VAR in "${REQUIRED_ENV_VARS[@]}"; do
    if [ -z "${!VAR}" ]; then
        echo "请通过环境变量 $VAR 提供值"
        exit 1
    fi
done

DATASET_SIZE=$(wc -l < "$DATA_PATH")
echo "数据集名称: $DATASET_NAME，数据集大小: $DATASET_SIZE"

# 计算每个程序处理的数据量
CHUNK_SIZE=$((DATASET_SIZE / PROCESS_COUNT))

# 检查是否能整除
if [ $((DATASET_SIZE % PROCESS_COUNT)) -ne 0 ]; then
    echo "警告：数据集大小 $DATASET_SIZE 不能被程序数量 $PROCESS_COUNT 整除，最后一个程序会处理剩余的数据。"
fi

# 并行运行任务
for ((i = 0; i < PROCESS_COUNT; i++)); do
    BEGIN=$((i * CHUNK_SIZE))
    END=$((BEGIN + CHUNK_SIZE))

    # 如果是最后一个程序，处理剩余的数据
    if [ $i -eq $((PROCESS_COUNT - 1)) ]; then
        END=$DATASET_SIZE
    fi

    # 提取该程序使用的GPU
    START_GPU_INDEX=$((i * GPU_PER_PROCESS))
    END_GPU_INDEX=$((START_GPU_INDEX + GPU_PER_PROCESS))
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CUDA_VISIBLE_DEVICES_ARRAY[*]:$START_GPU_INDEX:$GPU_PER_PROCESS}")

    echo "启动程序 $i，使用 GPU: $CUDA_VISIBLE_DEVICES，数据范围: [$BEGIN, $END)"
    # echo $(IFS=','; echo "$CUDA_VISIBLE_DEVICES") 

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ../../GenPRM/evaluate-PRM-CoT.py \
    --data_name $DATASET_NAME \
    --begin $BEGIN \
    --end $END \
    --model_path $MODEL_PATH \
    --output_path $OUTPUT_DIR \
    --prompt_path $PROMPT_PATH \
    --response_path $RESPONSE_PATH \
    --data_path $DATA_PATH \
    --temperature $TEMPERATURE &
done

wait

echo "所有任务完成！"
