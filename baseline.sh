#!/bin/bash

# 1.启动整个脚本在后台运行
nohup bash -c '
# 启动一个后台任务
nohup  python -u main.py --cfg configs/proteins/baseline_gcn.yaml > new_logs/proteins/gcn.log &
wait
nohup  python -u main.py --cfg configs/proteins/baseline_sage.yaml > new_logs/proteins/sage.log &

' > new_logs/proteins/baseline.log 2>&1 &