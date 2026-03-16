#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        //考生补充初始化代码

        // 通过tiling结构体传递tiling信息
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / (tileNum * BUFFER_NUM);

        // 设置GM起始地址
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(float)); // temp1
        pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(float)); // temp2

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        // alloc tensor from queue memory 
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<half>();

        // 每次处理大小为 tileLength的数据
        DataCopy(xLocal, xGm[progress *this->tileLength], this->tileLength);

        // enque to VECIN
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // deque from VECIN
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // allocate temporary buffer
        LocalTensor<float> temp1 = tempBuf1.AllocTensor<float>();
        LocalTensor<float> temp2 = tempBuf2.AllocTensor<float>();
        
        float scalar1 = static_cast<float>(-1.0f);
        float scalar2 = static_cast<float>(1.0f);

        // half -> float 精度转换
        Cast(temp1, xLocal, RoundMode::CAST_NONE, this->tileLength);
        // -x
        Muls(temp1, temp1, scalar1, this->tileLength);
        // exp(-x)
        Exp(temp1, temp1, this->tileLength);
        // 1 + exp(-x)
        Adds(temp1, temp1, scalar2, this->tileLength);
        // 1 / (1 + exp(-x))
        Duplicate(temp2, scalar2, this->tileLength);
        temp1 = temp2 / temp1;
        // float -> half 精度转换
        Cast(yLocal, temp1, RoundMode::CAST_ROUND, this->tileLength);       
        
        // EnQue to VECOUT
        outQueueY.EnQue(yLocal);

        // free input tensors
        inQueueX.FreeTensor(xLocal);
        tempBuf1.FreeTensor(temp1);
        tempBuf2.FreeTensor(temp2);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // DeQue from VECOUT
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        DataCopy(yGm[progress *this->tileLength], yLocal, this->tileLength);

        // free output tensors
        outQueueY.FreeTensor(yLocal);
    }

private:
    // 内存管理对象
    TPipe pipe;

    // 创建 VECIN / VECOUT 队列 
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // 临时存储
    TBuf<TPosition::VECCALC> tempBuf1;
    TBuf<TPosition::VECCALC> tempBuf2;

    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoid op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}