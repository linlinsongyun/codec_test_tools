#!/bin/bash

# 检查是否提供了两个路径作为参数
# gt_path /mnt/nas1/zhangruonan/nas/zhangruonan/eval/eval_pipline/pgc_60IP_testset_swap1
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <gt_path> <pred_path>"
  exit 1
fi

# 获取输入的文件路径
gt_path=$1
pred_path=$2

# 运行 mcd.py 计算 MCD
echo "Running MCD calculation..."
python mel_distance_eval.py "$gt_path" "$pred_path"

# # 运行 stft_distance.py 计算 STFT Distance
echo "Running STFT distance calculation..."
python stft_distance_eval.py "$gt_path" "$pred_path"

echo "Running pesq calculation..."
python pesq_try.py "$gt_path" "$pred_path"

echo "Running si_sdr calculation..."
python si_sdr_eval.py "$gt_path" "$pred_path"

echo "Running stoi calculation..."
python stoi_eval.py "$gt_path" "$pred_path"

echo "Test completed!"
