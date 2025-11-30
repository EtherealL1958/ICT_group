import os
import sys
import numpy as np

loss = 1e-3  # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
minimum = 1e-10

def verify_result(real_result_path, golden_path, debug_num=10):
    # 读取实际结果和预期结果
    real_result = np.fromfile(real_result_path, dtype=np.float16)
    golden = np.fromfile(golden_path, dtype=np.float16)

    # 计算绝对误差
    abs_error = np.abs(real_result - golden)
    # 计算相对误差
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    rel_error = abs_error / (deno + minimum)

    # 判断是否通过
    result_atol = abs_error <= loss
    result_rtol = rel_error <= loss

    # 调试信息
    print(f"=== Debug Info (first {debug_num} elements) ===")
    print("Real result:", real_result[:debug_num])
    print("Golden result:", golden[:debug_num])
    print("Absolute error:", abs_error[:debug_num])
    print("Relative error:", rel_error[:debug_num])
    print("Absolute error check (<= loss):", result_atol[:debug_num])
    print("Relative error check (<= loss):", result_rtol[:debug_num])
    print("============================================\n")

    # 判定逻辑
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:
            print("[ERROR] result error")
            return False

    print("test pass")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <real_result.bin> <golden_result.bin>")
        sys.exit(1)
    verify_result(sys.argv[1], sys.argv[2])
