import threading
from collections import defaultdict
import random

# 创建一个默认值为0的字典，用于计算节点之间的边数
A = [[0 for j in range(100)] for i in range(100)]

#   39561252
N = 80000000
a = [i for i in range(N)]
b = [i for i in range(N)]
cluster_map = {i: i % 100 for i in range(N)}
# 定义一个线程锁
lock = threading.Lock()

# 定义一个函数，用于更新计数
def update_count(start_node, end_node):
    cluster_id_start = cluster_map[start_node]
    cluster_id_end = cluster_map[end_node]
    with lock:
        A[cluster_id_start][cluster_id_end] += 1

# 定义一个线程函数，用于并行处理任务
from tqdm import tqdm
def process_data(start_nodes, end_nodes):
    for i in tqdm(range(len(start_nodes))):
        start_node = start_nodes[i]
        end_node = end_nodes[i]
        update_count(start_node, end_node)

# 设置线程数量
num_threads = 40

# 计算每个线程处理的数据量
batch_size = len(a) // num_threads

# 创建线程列表
threads = []

import time 
start = time.time()
# 分配数据给各个线程
for i in range(num_threads):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    thread = threading.Thread(target=process_data, args=(a[batch_start:batch_end], b[batch_start:batch_end]))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()
# process_data(a, b)
end = time.time()

# 将字典转换为矩阵
N = len(cluster_map)
# import ipdb;ipdb.set_trace()
print("cost: ", end-start)
# A_matrix = [[A[i][j] for j in range(N // 100)] for i in range(N // 100)]

# print(A_matrix)