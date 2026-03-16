#!/bin/bash
# wrapper.sh

# 1. 计算当前的 Local Rank (0-7)
RANK=$PMI_RANK
if [ -z "$RANK" ]; then
    RANK=$OMPI_COMM_WORLD_LOCAL_RANK
fi
if [ -z "$RANK" ]; then
    exec "$@"
fi

# 2. 计算对应的 NUMA 节点
# 鲲鹏920通常有4个Socket，8个NUMA节点（每个节点24核）
# 我们简单粗暴地让 Rank N 绑定到 NUMA N
NUMA_NODE=$((RANK % 8))

# 3. 显式绑定并运行
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

# 使用 numactl 将进程强制绑定到对应的 NUMA 节点内存和CPU
exec numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE "$@"