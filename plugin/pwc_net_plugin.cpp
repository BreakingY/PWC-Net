#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

using namespace nvinfer1;


// ============================================================
// CUDA implementation
// ============================================================

void launch_correlation(
    const float* input1,
    const float* input2,
    float* output,
    void* workspace,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream);
void launch_backwarp(
    const float* input,
    const float* flow,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream);
// ============================================================
//  Plugin
// ============================================================

class CorrelationPlugin : public IPluginV2DynamicExt
{
public:
    CorrelationPlugin() = default;

    CorrelationPlugin(const void* serialData, size_t serialLength)
    {
        // Plugin has no serialized parameters.
    }

    // --------------------------------------------------------
    // IPluginV2
    // --------------------------------------------------------

    const char* getPluginType() const noexcept override
    {
        return "Correlation";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    DimsExprs getOutputDimensions(
        int outputIndex,
        const DimsExprs* inputs,
        int nbInputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        DimsExprs output(inputs[0]);

        // [B, C, H, W] -> [B, 81, H, W]
        output.d[1] = exprBuilder.constant(81);

        return output;
    }

    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs,
        int nbInputs,
        const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override
    {
        const int batch = inputs[0].dims.d[0];
        const int channels = inputs[0].dims.d[1];
        const int height = inputs[0].dims.d[2];
        const int width = inputs[0].dims.d[3];

        const size_t elements =
            static_cast<size_t>(batch) *
            (height + 8) *
            (width + 8) *
            channels;

        return elements * sizeof(float) * 2;
    }

    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override
    {
        const int batch = inputDesc[0].dims.d[0];
        const int channels = inputDesc[0].dims.d[1];
        const int height = inputDesc[0].dims.d[2];
        const int width = inputDesc[0].dims.d[3];

       launch_correlation(
        static_cast<const float*>(inputs[0]),
        static_cast<const float*>(inputs[1]),
        static_cast<float*>(outputs[0]),
        workspace,
        batch,
        channels,
        height,
        width,
        stream);

        return 0;
    }

    int initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
    }

    size_t getSerializationSize() const noexcept override
    {
        return 0;
    }

    void serialize(void* buffer) const noexcept override
    {
    }

    void destroy() noexcept override
    {
        delete this;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new CorrelationPlugin();
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    // --------------------------------------------------------
    // IPluginV2Ext
    // --------------------------------------------------------

    DataType getOutputDataType(
        int index,
        const DataType* inputTypes,
        int nbInputs) const noexcept override
    {
        return DataType::kFLOAT;
    }


    void configurePlugin(
        const DynamicPluginTensorDesc* inputs,
        int nbInputs,
        const DynamicPluginTensorDesc* outputs,
        int nbOutputs) noexcept override
    {
    }

    // --------------------------------------------------------
    // IPluginV2DynamicExt
    // --------------------------------------------------------

    bool supportsFormatCombination(
        int pos,
        const PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == TensorFormat::kLINEAR;
    }

private:
    std::string mNamespace;
};

class BackwarpPlugin : public IPluginV2DynamicExt
{
public:
    BackwarpPlugin() = default;

    BackwarpPlugin(const void* serialData, size_t serialLength)
    {
        // No serialized parameters.
    }

    const char* getPluginType() const noexcept override
    {
        return "Backwarp";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    int getNbOutputs() const noexcept override
    {
        return 1;
    }

    DimsExprs getOutputDimensions(
        int outputIndex,
        const DimsExprs* inputs,
        int nbInputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        // input:  [B, C, H, W]
        // flow:   [B, 2, H, W]
        // output: [B, C, H, W]

        return inputs[0];
    }

    size_t getWorkspaceSize(
        const PluginTensorDesc* inputs,
        int nbInputs,
        const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(
        const PluginTensorDesc* inputDesc,
        const PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override
    {
        const int batch =
            inputDesc[0].dims.d[0];

        const int channels =
            inputDesc[0].dims.d[1];

        const int height =
            inputDesc[0].dims.d[2];

        const int width =
            inputDesc[0].dims.d[3];

        launch_backwarp(
            static_cast<const float*>(inputs[0]),
            static_cast<const float*>(inputs[1]),
            static_cast<float*>(outputs[0]),
            batch,
            channels,
            height,
            width,
            stream);

        return 0;
    }

    int initialize() noexcept override
    {
        return 0;
    }

    void terminate() noexcept override
    {
    }

    size_t getSerializationSize() const noexcept override
    {
        return 0;
    }

    void serialize(void* buffer) const noexcept override
    {
    }

    void destroy() noexcept override
    {
        delete this;
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        return new BackwarpPlugin();
    }

    void setPluginNamespace(
        const char* pluginNamespace) noexcept override
    {
        mNamespace =
            pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(
        int index,
        const DataType* inputTypes,
        int nbInputs) const noexcept override
    {
        return DataType::kFLOAT;
    }

    void configurePlugin(
        const DynamicPluginTensorDesc* inputs,
        int nbInputs,
        const DynamicPluginTensorDesc* outputs,
        int nbOutputs) noexcept override
    {
    }

    bool supportsFormatCombination(
        int pos,
        const PluginTensorDesc* inOut,
        int nbInputs,
        int nbOutputs) noexcept override
    {
        return inOut[pos].type == DataType::kFLOAT &&
               inOut[pos].format == TensorFormat::kLINEAR;
    }

private:
    std::string mNamespace;
};
// ============================================================
// Plugin Creator
// ============================================================

class CorrelationPluginCreator : public IPluginCreator
{
public:
    CorrelationPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }

    const char* getPluginName() const noexcept override
    {
        return "Correlation";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(
        const char* name,
        const PluginFieldCollection* fc) noexcept override
    {
        return new CorrelationPlugin();
    }

    IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        return new CorrelationPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
};

class BackwarpPluginCreator : public IPluginCreator
{
public:
    BackwarpPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }

    const char* getPluginName() const noexcept override
    {
        return "Backwarp";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV2* createPlugin(
        const char* name,
        const PluginFieldCollection* fc) noexcept override
    {
        return new BackwarpPlugin();
    }

    IPluginV2* deserializePlugin(
        const char* name,
        const void* serialData,
        size_t serialLength) noexcept override
    {
        return new BackwarpPlugin(
            serialData,
            serialLength);
    }

    void setPluginNamespace(
        const char* pluginNamespace) noexcept override
    {
        mNamespace =
            pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
};
// ============================================================
// Register plugin
// ============================================================

REGISTER_TENSORRT_PLUGIN(CorrelationPluginCreator);
REGISTER_TENSORRT_PLUGIN(BackwarpPluginCreator);