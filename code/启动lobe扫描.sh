#!/bin/bash
# 工作站 lobe 扫描启动脚本
# 使用：在工作站上 bash 启动lobe扫描.sh
# 前提：密度扫描已完成并把结果同步到了 d:\TanHuangWork\HBN4/results

# 1. 清空 lobe features checkpoint（之前可能脏的）
rm -f /mnt/data1/home/tanhuang/code/EEG_code/HBN4/results/tables/lobe_features_checkpoint.pkl
rm -f /mnt/data1/home/tanhuang/code/EEG_code/HBN4/results/tables/lobe_features.csv

# 2. 检查磁盘空间（确保有 5 GB+）
df -h /mnt/data1

# 3. 启动 lobe 扫描（必须等密度扫描完全结束）
cd /mnt/data1/home/tanhuang/code/EEG_code/HBN4/code

# 建议用 nohup 后台跑，万一断线也不中断
nohup python run_lobe_allometry.py --n-jobs 4 > /tmp/lobe_run.log 2>&1 &

echo "Lobe scan PID: $!"
echo "Log: /tmp/lobe_run.log"
echo "Result CSV will be at:"
echo "  /mnt/data1/home/tanhuang/code/EEG_code/HBN4/results/tables/tae_lobe_allometry.csv"