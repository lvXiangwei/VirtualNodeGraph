#!/bin/bash

# 1.启动整个脚本在后台运行
nohup bash -c '
# 启动一个后台任务
config="configs/model3/sage.yaml"
for num_parts in 8 64 128; do
    for attn_dropout in -1 0.3 0.5 0.6; do
        for merge_dropout in -1 0.3 0.5 0.6; do
            for vntran_act in "tanh" "sigmoid" "ReLU" "None"; do
                python -u main.py --cfg $config --num_parts $num_parts --attn_dropout $attn_dropout --merge_dropout $merge_dropout --vntran_act $vntran_act > new_logs/model3/${num_parts}.${attn_dropout}.${merge_dropout}.${vntran_act}.log 
            done
        done
    done
done
wait

' > new_logs/model3/model3_tuning.log 2>&1 &