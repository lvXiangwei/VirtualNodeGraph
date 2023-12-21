import threading
from collections import defaultdict
import random
import multiprocessing
# 创建一个默认值为0的字典，用于计算节点之间的边数
A = [[0 for j in range(100)] for i in range(100)]

#   39561252
N = 80000000
a = [i for i in range(N)]
b = [i for i in range(N)]
cluster_map = {i: i % 100 for i in range(N)}


results = multiprocessing.Manager().list()

# 定义一个线程函数，用于并行处理任务
from tqdm import tqdm
import numpy as np
def process_data(start_nodes, end_nodes):
    adj = np.zeros((100, 100), dtype=int)
    for i in tqdm(range(len(start_nodes))):
        start_node = start_nodes[i]
        end_node = end_nodes[i]
        cluster_id_start = cluster_map[start_node]
        cluster_id_end = cluster_map[end_node]
        adj[cluster_id_start][cluster_id_end] += 1
    results.append(adj)

num_processes = 10
batch_size = len(a) // num_processes
processes = []

import time 
start = time.time()

# 分配数据给各个线程
for i in range(num_processes):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    process = multiprocessing.Process(target=process_data, args=(a[batch_start:batch_end], b[batch_start:batch_end]))
    processes.append(process)
    process.start()

for process in processes:
    process.join()
# process_data(a, b)
adj = np.zeros((100, 100), dtype=int)
for res in results:
    adj += res
end = time.time()

# 将字典转换为矩阵
N = len(cluster_map)
# import ipdb;ipdb.set_trace()
print("cost: ", end-start)
# A_matrix = [[A[i][j] for j in range(N // 100)] for i in range(N // 100)]

# print(A_matrix)