NOW_DIR=$(cd `dirname $0`; pwd)
cd $NOW_DIR
echo "当前代码运行所在的路径为： $NOW_DIR"
SCORE_PATH=$1
PRMBENCH_DATASET_PATH=$2
MODEL_NAME="GenPRM"

python ../code/new_get_result.py\
	    --score_path $SCORE_PATH \
        --prm_dataset_path $PRMBENCH_DATASET_PATH\
        --model_name $MODEL_NAME
