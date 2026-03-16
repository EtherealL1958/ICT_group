#!/bin/bash

# 大页内存设置
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    echo "Transparent Huge Pages enabled."
else
    echo "Warning: Cannot enable Huge Pages (Permission denied)."
fi

export OMP_NUM_THREADS=24

# 强制重写 hpcg.dat 为 60秒 模式，快速验证
rm -f hpcg.dat
echo "HPCG benchmark input file" > hpcg.dat
echo "Sandia National Laboratories; University of Tennessee, Knoxville" >> hpcg.dat
echo "64 64 64" >> hpcg.dat
echo "60" >> hpcg.dat

echo "Starting HPCG with NUMACTL wrapper..."

chmod +x ./wrapper.sh

# 运行 wrapper
mpiexec -np 8 ./wrapper.sh ./bin/xhpcg