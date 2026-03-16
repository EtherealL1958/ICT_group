/*
 * KUPL HBW特性 编程Demo
 * 编译命令： clang++ hbw.cpp -o hbw -O3 -lkupl
 * 运行命令：./hbw
 */

#include <stdio.h>
#include <unistd.h>

int main() {
    char *ptr;
    /* 检测系统中是否存在HBW内存 */
    if (kupl_hbw_check_available() == 0) {
        printf("HBW memory undetected, skipping this example\n");
        return 0;
    }

    /* 获取当前的HBW内存分配策略，并将其设置为KUPL_HBW_POLICY_BIND模式 */
    printf("Current HBW policy is %d.\n", kupl_hbw_get_policy());
    kupl_hbw_set_policy(KUPL_HBW_POLICY_BIND);

    /* 尝试分配1024字节的HBW内存 */
    if ((ptr = (char *) kupl_hbw_malloc(1024)) == nullptr) {
        printf("Problem in allocating memory\n");
    }
    printf("%p", ptr);

    /* 检验分配的内存是否是HBW内存 */
    int result = kupl_hbw_verify(ptr, 1024, KUPL_HBW_TOUCH_PAGES);
    printf("local test verify result is %d\n", result);

    /* 释放分配的HBW内存 */
    kupl_hbw_free(ptr);
    return 0;
}