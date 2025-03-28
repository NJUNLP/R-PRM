#!/bin/bash
# source环境
# CONDA_BASE=$(conda info --base)
# export CUDA_VISIBLE_DEVICES=4,5,6,7
source "/data1/software/anaconda3/etc/profile.d/conda.sh"
export HF_ENDPOINT=https://hf-mirror.com

# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"

train () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/finetune
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/MCTS-RM/GenRM/configs/LF-config/train.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_all () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/finetune
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/MCTS-RM/GenRM/configs/LF-config/train_all.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_4GPU () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/finetune
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/MCTS-RM/GenRM/configs/LF-config/train-4GPU.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_8GPU () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/finetune
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/MCTS-RM/GenRM/configs/LF-config/train-8GPU.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_4GPU_DPO () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/DPO/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/DPO
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/liujx/scripts/LF-config/DPO.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_full () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"
mkdir -p /home/adminad/GenRM/logs/finetune
cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/MCTS-RM/GenRM/configs/LF-config/full-train.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}

train_discriminative () {

echo $OUTPUT_DIR
export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/finetune/$(date +%s).log"

cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$OUTPUT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/train-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/GenRM/GenRM/LF-config/discriminative.yaml > "$tmp_cfg"
llamafactory-cli train "$tmp_cfg" > "$log_file"

}


merge () {

echo $OUTPUT_DIR

export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/merge/$(date +%s).log"

cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$EXPORT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/merge-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/GenRM/GenRM/LF-config/merge.yaml > "$tmp_cfg"
llamafactory-cli export "$tmp_cfg" > "$log_file"
}

merge_discriminative () {

echo $OUTPUT_DIR

export TZ='Asia/Shanghai'

log_file="/home/adminad/GenRM/logs/merge/$(date +%s).log"

cd /home/adminad/GithubRepos/LLaMA-Factory || exit

tmp_cfg_dir="$EXPORT_DIR/tmp_configs"
mkdir -p "$tmp_cfg_dir"  # 确保目录存在
tmp_cfg="$tmp_cfg_dir/merge-$(date +%Y%m%d%H%M%S).yaml"

envsubst < /home/adminad/GenRM/GenRM/LF-config/merge-discriminative.yaml > "$tmp_cfg"
llamafactory-cli export "$tmp_cfg" > "$log_file"
}

conda activate ljx-LF

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export MODEL_PATH=/data1/model/Qwen2.5-Math-7B-Instruct
# export DATASET=GenPRM-67k-train-5:5-prompt4
# export TEMPLATE=qwen
# export OUTPUT_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-67k-train-5:5-prompt4
# export LEARNING_RATE=0.000005
# export WARMUP_RATIO=0.03
# export EPOCHS=1
# export LOGGING_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-67k-train-5:5-prompt4/runs
# export EVAL_DATASET=GenPRM-2k-prompt4
# export RESUME_FROM_CHECKPOINT=null

# train_8GPU


# bash /home/adminad/MCTS-RM/GenRM/scripts/examples/evaluate-GenPRM.sh
# conda activate old-vllm
# bash /home/adminad/MCTS-RM/GenRM/scripts/examples/eval-ProcessBench.sh "/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-67k-train-5:5-prompt4" "/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new4-qwen.txt"

conda activate ljx-LF
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/data1/model/Qwen2.5-Math-7B-Instruct
export DATASET=GenPRM-65k-train-5:5-prompt6
export TEMPLATE=qwen
export OUTPUT_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-65k-train-5:5-prompt6
export LEARNING_RATE=0.000005
export WARMUP_RATIO=0.03
export EPOCHS=1
export LOGGING_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/fullGenPRM-65k-train-5:5-prompt6/runs
export EVAL_DATASET=GenPRM-2k-test-prompt6
export RESUME_FROM_CHECKPOINT=null

# train_8GPU
conda activate old-vllm
# bash /home/adminad/MCTS-RM/GenRM/scripts/examples/eval-ProcessBench.sh /data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-65k-train-5:5-prompt6 "qwen-2.5-math-instruct"  "/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new6-qwen.txt"
conda activate ljx-LF
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MODEL_PATH=/data1/model/Qwen2.5-Math-7B-Instruct
export DATASET=GenPRM-77k-train-3:4-prompt6
export TEMPLATE=qwen
export OUTPUT_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-77k-train-3:4-prompt6
export LEARNING_RATE=0.000005
export WARMUP_RATIO=0.03
export EPOCHS=1
export LOGGING_DIR=/data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-77k-train-3:4-prompt6/runs
export EVAL_DATASET=GenPRM-2k-test-prompt6
export RESUME_FROM_CHECKPOINT=null

# train_4GPU
conda activate old-vllm
bash /home/adminad/MCTS-RM/GenRM/scripts/examples/eval-ProcessBench.sh /data2/ckpts/GenRM/qwen-2.5-math-instruct/full/GenPRM-77k-train-3:4-prompt6 "qwen-2.5-math-instruct"  "/home/adminad/MCTS-RM/GenRM/prompts/GenPRM/GenPRM-evaluate-new6-qwen.txt"