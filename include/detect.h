#ifndef DETECT_H
#define DETECT_H

#include <chrono>
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include "logger.hpp"
#include "yololayer.h"
#include <fstream>
#include <string>
#include <vector>
#include "utils.h"
#include <assert.h>

#if defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#define CDECL __attribute__((__cdecl))
#else
//  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #define CDECL
    #pragma warning Unknown dynamic link import/export semantics.
#endif

using namespace nvinfer1;
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

static Logger gLogger;

namespace Vehicle
{
    class Detect
    {
    private:

        uint8_t batchSize{};
        uint8_t numClasses{};
        uint32_t outputSize{};
        uint16_t inputH{};
        uint16_t inputW{};
        uint32_t maxInputSize{};
        const char* inputBlobName{};
        const char* outputBlobName{};
        float nmsThresh{};
        float confThresh{};

        float *imgBuffer{};
        float *outputBuffer{};
        std::vector<void *> buffers{};
        uint8_t inputIndex{};
        uint8_t outputIndex{};

        struct TRTDelete{
            template<class T>
            void operator()(T* obj) const
            {
                delete obj;
            }
        };

        template<class T>
        using TRTptr = std::unique_ptr<T, TRTDelete>;

        TRTptr<nvinfer1::IRuntime> runtime{nullptr};
        TRTptr<nvinfer1::ICudaEngine> engine{nullptr};
        TRTptr<nvinfer1::IExecutionContext> context{nullptr};
        TRTptr<nvinfer1::IBuilder> builder{nullptr};
        TRTptr<nvinfer1::INetworkDefinition> network{nullptr};
        TRTptr<nvonnxparser::IParser> parser{nullptr};
        TRTptr<nvinfer1::IBuilderConfig> builderCfg;

    protected:
        void preprocessImage(const cv::Mat& img, float* imgBufferArray) const;
        static float iou(float lbox[4], float rbox[4]);
        void nms(std::vector<Yolo::Detection>& res, float *output) const;
        static bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);

    public:
        Detect();
        void createContextExecution();

        //std::string doInference(cv::Mat& img);
        std::vector<Yolo::Detection> doInference(cv::Mat& img);


        const char* getVersion();

        const char* getWVersion();

    };
}

#endif // DETECT_H
