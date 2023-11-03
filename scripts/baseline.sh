### 1. gcn
nohup python -u main.py --cfg configs/baseline_gcn.yaml > logs/baseline/gcn/gcn.log &

### 2. sage
nohup python -u main.py --cfg configs/baseline_sage.yaml > logs/baseline/sage/sage.log &

### 3. virtual gcn 
nohup python -u main.py --cfg configs/virtual_gcn.yaml > logs/virtual/virtual_gcn.log & # batchnorm
nohup python -u main.py --cfg configs/virtual_gcn2.yaml > logs/virtual/virtual_gcn2.log & # layerNorm
nohup python -u main.py --cfg configs/virtual_sage.yaml > logs/virtual/virtual_sage.log & # batchnorm

nohup python -u main.py --cfg configs/virtual_sage2.yaml > logs/virtual/virtual_sage2.log & # layerNorm
nohup python -u main.py --cfg configs/virtual_sage3.yaml > logs/virtual/virtual_sage3.log & # layerNorm, maybe current sota

### jknet 
nohup python -u main.py --cfg configs/virtual_jknet.yaml > logs/jknet/virtual_jknet.log &

### baseline: one virtual node
nohup python -u main.py --cfg configs/baseline_gcn_one_virtual.yaml > logs/baseline/gcn/gcn_one_virtual.log &
nohup python -u main.py --cfg configs/baseline_sage_one_virtual.yaml > logs/baseline/sage/sage_one_virtual.log &
nohup python -u main.py --cfg configs/baseline_jknet_one_virtual.yaml > logs/baseline/jknet/baseline_jknet_one_virtual.log &