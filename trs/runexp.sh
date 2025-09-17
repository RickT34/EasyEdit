#!/bin/bash -e
ALGO=$1
DATA_JSON=$2
LABEL=$3
MODEL_PATH="/Data2/tangrui/EasyEdit/trs/models/LLama-3-8B-Instruct"
MODEL_NAME="llama3-8b"
PARAMS=(
  "$ALGO 6 All"
  # "$ALGO 3 4q4"
  # "$ALGO 5 2q4"
  # "$ALGO 6 3q4"
)

function exp(){
    ./lazyeditor.py --editing_method $1 --model_name "$MODEL_NAME" --device $2 --ds_range $3 --label "$LABEL" --model_path "$MODEL_PATH" --data_json "$DATA_JSON"
}


# 创建带时间戳的日志目录
timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="./explogs/${LABEL}/${ALGO}"
mkdir -p "$log_dir"
echo "所有实验日志将保存在: $log_dir"

# 启动所有实验
pids=()
for i in "${!PARAMS[@]}"; do
  # 生成唯一日志文件名（带时间戳）
    log_file="${log_dir}/exp_${timestamp}_${i}.log"
    
    # 执行实验并记录日志
    msg="启动实验 #$i: ${PARAMS[$i]}"
    echo "$msg" > "$log_file"

    exp ${PARAMS[$i]} >> "$log_file" 2>&1 &

    pids+=($!)
    echo "$msg  pid=$!"
done

# 等待所有实验完成
for pid in ${pids[@]}; do
    wait $pid
    echo "等待实验完成 pid=$pid..."
done
wait

./mail.py "ICT-v2" "Exp Done: $ALGO $LABEL $DATA_JSON"

exit 0
