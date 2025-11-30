#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__
void add_custom(GM_ADDR x_addr, GM_ADDR y_addr, GM_ADDR z_addr,
                GM_ADDR workspace, GM_ADDR tiling)
{
    // 读取 Tiling 信息
    GET_TILING_DATA(tiling_data, tiling);

    int32_t elem_num = tiling_data.elem_num;
    int32_t dtype = tiling_data.dtype;

    // 映射 GM 地址
    if (dtype == 0) {              // fp16
        GlobalTensor<half> xG((half*)x_addr);
        GlobalTensor<half> yG((half*)y_addr);
        GlobalTensor<half> zG((half*)z_addr);

        TPipe pipe;
        constexpr int32_t BLOCK = 1024;

        LocalTensor<half> xUB = pipe.alloc_local_tensor<half>({BLOCK});
        LocalTensor<half> yUB = pipe.alloc_local_tensor<half>({BLOCK});
        LocalTensor<half> zUB = pipe.alloc_local_tensor<half>({BLOCK});

        int32_t offset = 0;
        while (offset < elem_num) {
            int32_t len = Min(BLOCK, elem_num - offset);

            pipe.gm2ub(xUB, xG[offset], len);
            pipe.gm2ub(yUB, yG[offset], len);
            pipe.wait();

            add(zUB, xUB, yUB, len);

            pipe.ub2gm(zG[offset], zUB, len);
            pipe.wait();

            offset += len;
        }

    } else if (dtype == 1) {       // float32
        GlobalTensor<float> xG((float*)x_addr);
        GlobalTensor<float> yG((float*)y_addr);
        GlobalTensor<float> zG((float*)z_addr);

        TPipe pipe;
        constexpr int32_t BLOCK = 1024;

        LocalTensor<float> xUB = pipe.alloc_local_tensor<float>({BLOCK});
        LocalTensor<float> yUB = pipe.alloc_local_tensor<float>({BLOCK});
        LocalTensor<float> zUB = pipe.alloc_local_tensor<float>({BLOCK});

        int32_t offset = 0;
        while (offset < elem_num) {
            int32_t len = Min(BLOCK, elem_num - offset);

            pipe.gm2ub(xUB, xG[offset], len);
            pipe.gm2ub(yUB, yG[offset], len);
            pipe.wait();

            add(zUB, xUB, yUB, len);

            pipe.ub2gm(zG[offset], zUB, len);
            pipe.wait();

            offset += len;
        }

    } else {                       // int32
        GlobalTensor<int32_t> xG((int32_t*)x_addr);
        GlobalTensor<int32_t> yG((int32_t*)y_addr);
        GlobalTensor<int32_t> zG((int32_t*)z_addr);

        TPipe pipe;
        constexpr int32_t BLOCK = 1024;

        LocalTensor<int32_t> xUB = pipe.alloc_local_tensor<int32_t>({BLOCK});
        LocalTensor<int32_t> yUB = pipe.alloc_local_tensor<int32_t>({BLOCK});
        LocalTensor<int32_t> zUB = pipe.alloc_local_tensor<int32_t>({BLOCK});

        int32_t offset = 0;
        while (offset < elem_num) {
            int32_t len = Min(BLOCK, elem_num - offset);

            pipe.gm2ub(xUB, xG[offset], len);
            pipe.gm2ub(yUB, yG[offset], len);
            pipe.wait();

            add(zUB, xUB, yUB, len);

            pipe.ub2gm(zG[offset], zUB, len);
            pipe.wait();

            offset += len;
        }
    }
}
