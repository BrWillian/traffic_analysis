#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>

namespace Yolo
{
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[6];
    };
    struct alignas(float) Detection {

        float bbox[4];
        float conf;
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int getNbOutputs() const noexcept override;

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

        int initialize() noexcept override;

        virtual void terminate() noexcept override {}

        virtual size_t getWorkspaceSize(int maxBatchSize) const noexcept override { return 0; }

        virtual int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

        virtual size_t getSerializationSize() const noexcept override;

        virtual void serialize(void* buffer) const noexcept override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override;

        const char* getPluginType() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        IPluginV2IOExt* clone() const noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;

        DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

        void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

        void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept override;

        void detachFromContext() noexcept override;

    private:

        void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
        int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        const PluginFieldCollection* getFieldNames() noexcept override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char* libNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;
    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
}

#endif  // _YOLO_LAYER_H
