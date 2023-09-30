# VirtualNodeGraph
This repository focuses on virtual nodes used for node classification.


## Experiment
### 1. Virtual node on full graph

- setting:
    - dataset: arxiv
    - model: gcn
    - sampling method: None, full graph training
    - 5 runs

- scripts:
    ```python
    nohup python -u train.py > logs/baseline_gcn.log & # baseline gcn

    nohup python -u train.py --use_virtual > logs/virtualdata_gcn.log & # baseline `
    ```



- result:
    - w virtual node:     75.45 ± 0.16
    - w/o virtual node:   71.91 ± 0.29


### 2. Virtual node on different sampling method

#### 2.1 neighbor sampling 

-  settings: 
    - neighbors:[6, 5, 5], (form ibmb)
    - parts: [100]

- scripts:

    ```python
    # change config.py, line 13, use_virtual = False
    python -u train_neighbor_sampling.py > logs/baseline_sampling.log # config.py, w/o virtual node

    # change config.py, line 13, use_virtual = True
    python -u train_neighbor_sampling.py > logs/virtualdata_neighborsampling.log # config.py, w virtual node

    ```
- results:

    - w virtual node:     82.22
    - w/o virtual node:   70.41 

       
        

