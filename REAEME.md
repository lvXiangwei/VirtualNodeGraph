#### 1. Baseline
    
1.1 stardard model: 
- gcn
    - 71.74 ± 0.29
- sage
    - 71.49 ± 0.27
- gat(16机子, dgl)
    - 73.08 ± 0.26

1.2 deep model:
- JK 
    - 72.19 ± 0.21
- RevGAT
    - acc: 

#### 2. Virtual Node Model Design

2.1 Only One Virtual Node(baseline)
- gcn
    - 72.08 ± 0.38 
- sage
    - 72.12 ± 0.24
- gat
    - acc
- JK
    - 71.99 ± 0.11


2.2 Model design
- gcn
    - 72.21 ± 0.34, (virtual_gcn2.yaml)
- sage
    - 72.64 ± 0.13 (virtual_sage3.yaml)
- gat
    - None
- JKNet
    - 72.14 ± 0.12 (virtual_jknet.yaml)


----
### Obgn-arxiv Result: 

| Model                                    | Acc          |
| ---------------------------------------- | ------------ |
| GCN                                      | 71.74 ± 0.29 |
| GraphSage                                | 71.49 ± 0.27 |
| GAT                                      | 73.08 ± 0.26 |
| JKNet (GCN-based)                        | 72.19 ± 0.21 |
|                                          |              |
| GCN + One Virtual Node                   | 72.08 ± 0.38 |
| GraphSage + One Virtual Node             | 72.12 ± 0.24 |
| GAT + One Virtual Node                   |              |
| JKNet (GCN-based) + One Virtual Node     | 71.99 ± 0.11 |
|                                          |              |
| GCN + Mutiple Virtual Node               | 72.21 ± 0.24 |
| GraphSage + Mutiple Virtual Node         | 72.64 ± 0.13 |
| GAT + Mutiple Virtual Node               |              |
| JKNet (GCN-based) + Mutiple Virtual Node | 72.14 ± 0.12 |


---
model.py: current result, logs/
model2.py: logs/mlp reduction
model3.py: logs/attention_v1
