### model4：model design

<!-- - $h_v^l = f(h_v^{l-1}, h_u^{l-1} | u \in \mathcal{N}(v), h_{g})$ -->
- $g$: virtual node embedding, (C, H)
- $h_v$: graph node embedding, (N, H)
- $L$: layers
#### Graph node 
- ${\hat{h}_v^{l}} = GNUpdate(...)$, $l = 1..L$
    - graph node 更新，main gnn network
    - $conv(h_v^{l-1}, h_u^{l-1} | u \in \mathcal{N}(v))$ (1)
- $g_{trans}^{l} = GNTrans(g^{l-1})$
    - virtual node 向graph node传递信息
    -  $tanh(W_1^l(h_g^{l-1}))$ (2)
- $h_v^{l} = GNMerge(g_{trans}^{l}, {\hat{h}_v^{l}})$
    - 混合两种信息
    - $(1 - z^l) * {\hat{h}_v^{l}} + z^l * g_{trans}^l$ (3)
    - $z^l = \sigma(W_2^l{\hat{h}_v^{l}} + G_2^lg_{trans}^l) $ (4)

#### Virtual node
- $\hat{g}^l= VNUpdate(g^{l-1})$ (5)
    - virtual node 更新
    - $W_4g^{l-1}$
- $h_{trans}^l = VNTrans(h_v^{l-1}, g^{l-1})$ (6)
    - graph node向virutal node传递信息
    - $a_{i,j}^{l}=a^{l \top}W_3[g_i;h_{v_j}]$, # （1，）
    - $g_i^l = \sum_{j \in clutser(i)} a_{i,j}^lh_{v_j}^l$ #（1，H）
    - $h_{trans}^l = stack(g_i^l), i =[1,...,C]$ # (C, H)
- $h_g^l = VNMerge(h_{trans}^l， \hat{g}^l)$
    - 混合两种信息
    - $(1 - z^l) * h_{trans}^l + z^l * \hat{h}_g^l$ 
    - $ z^l = \sigma(W_3^lh_v' + G_3^lg_{trans}^l) $

- parms:  
```python
```
#### Experiment:
- part: 64
- attn dropout: 0.6
   - acc:  

- attn dropout: 0.6, merge dropout: 0.6
    - acc: 

### model5：
- 加入cluster之间连边
    - (5)改为 $\hat{g}_i^l = conv(g_j^{l-1}, g_k^{l-1} | k \in \mathcal{N}_g(v))$

### model6: 
- 尝试只在隐层引入设计 

### model7:
- minibatch
    - neighborsampling
 - VNTrans:
- GraphWarp:


