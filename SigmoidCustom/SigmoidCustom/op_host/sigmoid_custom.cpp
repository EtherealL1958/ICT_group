#include "sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

// 输入规模：[8, 2048]，元素类型为 half，参见文档矢量编程/多核tiling一栏
constexpr uint32_t CORE_NUM = 8; // 核数
constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t BLOCK_DIM = CORE_NUM; // 8块block

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SigmoidCustomTilingData tiling;
    // 输入0的原始元素总数
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 设置并行block数
    context->SetBlockDim(BLOCK_DIM);

    // 写入tiling参数
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
 
	// 5. 将 tiling 序列化给 kernel
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // Sigmoid无需workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

	return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SigmoidCustom : public OpDef {
public:
    explicit SigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b")
                      .AddConfig("ascend310b");
    }
};

OP_ADD(SigmoidCustom);
}
