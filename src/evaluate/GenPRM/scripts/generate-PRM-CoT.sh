#!/bin/bash

# 进入当前目录
NOW_DIR=$(cd `dirname $0`; pwd)
cd $NOW_DIR
echo "当前代码运行所在的路径为： $NOW_DIR"

# 检查是否提供了CUDA_VISIBLE_DEVICES参数
if [ -z "$1" ]; then
    echo "请通过参数提供CUDA_VISIBLE_DEVICES的值，例如：4,5,6,7"
    exit 1
fi

# 检查是否提供了每个程序使用的GPU数量参数
if [ -z "$2" ]; then
    echo "请通过第二个参数提供每个程序使用的GPU数量，例如：2"
    exit 1
fi

# 获取参数并转换为数组
CUDA_VISIBLE_DEVICES_STR=$1
IFS=',' read -r -a CUDA_VISIBLE_DEVICES_ARRAY <<< "$CUDA_VISIBLE_DEVICES_STR"

# 获取GPU数量
GPU_COUNT=${#CUDA_VISIBLE_DEVICES_ARRAY[@]}
echo "检测到 GPU 数量: $GPU_COUNT"

# 获取每个程序使用的GPU数量
GPUS_PER_PROCESS=$2

# 检查GPU数量是否足够
if [ $((GPU_COUNT % GPUS_PER_PROCESS)) -ne 0 ]; then
    echo "错误：GPU 数量 $GPU_COUNT 不能被每个程序使用的 GPU 数量 $GPUS_PER_PROCESS 整除。"
    exit 1
fi

# 计算需要启动的程序数量
PROCESS_COUNT=$((GPU_COUNT / GPUS_PER_PROCESS))

# 检查是否提供了数据集名称
if [ -z "$DATASET_NAME" ]; then
    echo "请通过环境变量 DATASET_NAME 提供数据集名称，例如：math 或 gsm8k"
    exit 1
fi

# 根据数据集名称设置数据集大小
case "$DATASET_NAME" in
    math50)
        DATASET_SIZE=14419
        ;;
    math500-llama)
       DATASET_SIZE=154612
        ;;
    math500-qwen)
       DATASET_SIZE=89744
        ;;
    prmbench)
       DATASET_SIZE=83456
        ;;
    prmbench-temp)
       DATASET_SIZE=17
        ;;
    prmbench-test)
       DATASET_SIZE=8
        ;;
    gsm8k-part)
        DATASET_SIZE=20083
        ;;
    gsm8k-process-bench)
        DATASET_SIZE=1568
        ;;
    processbench)
        DATASET_SIZE=16550
        ;;
    gsm8k-part-process-bench)
        DATASET_SIZE=410
        ;;
    math-process-bench)
        DATASET_SIZE=4368
        ;;
    math-part-process-bench)
        DATASET_SIZE=905
        ;;
    omnimath-process-bench)
        DATASET_SIZE=4792
        ;;
    omnimath-part-process-bench)
        DATASET_SIZE=936
        ;;
    olympiadbench-process-bench)
        DATASET_SIZE=5822
        ;;
    olympiadbench-part-process-bench)
        DATASET_SIZE=1230
        ;;
    *)
        echo "错误：不支持的数据集名称 '$DATASET_NAME'，请使用 'math' 或 'gsm8k'"
        exit 1
        ;;
esac

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

    # 计算当前程序使用的GPU
    GPU_START=$((i * GPUS_PER_PROCESS))
    GPU_END=$((GPU_START + GPUS_PER_PROCESS - 1))
    CUDA_VISIBLE_DEVICES_SUBSET=""
    for ((j = GPU_START; j <= GPU_END; j++)); do
        CUDA_VISIBLE_DEVICES_SUBSET+="${CUDA_VISIBLE_DEVICES_ARRAY[$j]},"
    done
    CUDA_VISIBLE_DEVICES_SUBSET=${CUDA_VISIBLE_DEVICES_SUBSET%,}

    # 设置环境变量 CUDA_VISIBLE_DEVICES
    # export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_SUBSET

    # 设置 tensor_number 为每个程序使用的 GPU 数量
    TENSOR_NUMBER=$GPUS_PER_PROCESS

    echo "启动程序 $i，使用的 GPU: $CUDA_VISIBLE_DEVICES_SUBSET，处理数据范围: $BEGIN - $END"

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_SUBSET python ../../../utils/inference.py \
    --temperature $TEMPERATURE \
    --gpu_memory_utilization 0.95 \
    --tensor_number $TENSOR_NUMBER \
    --n $N \
    --dtype bfloat16 \
    --data_begin $BEGIN \
    --data_end $END \
    --model_path $MODEL_PATH \
    --prompt_path $PROMPT_PATH \
    --data_path $DATA_PATH  \
    --write_path $OUTPUT_DIR \
    --max_token 4096 \
    --prompt_type GenPRM-CoT-Generate &
done
echo "Output directory: $OUTPUT_DIR"
wait