#ifndef DETECT_H
#define DETECT_H

#include <chrono>
#include <iostream>
#include <numeric>
#include <cmath>
#include <NvInfer.h>
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
#include "../generated/weights.h"

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

#define MODEL_TYPE_VEHICLE     (0x00000000)
#define MODEL_TYPE_PLATE       (0x00000001)
#define MODEL_TYPE_OCR         (0x00000002)
#define MODEL_TYPE_COLOR       (0x00000003)

using namespace nvinfer1;
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

static Logger gLogger;

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
    uint32_t modelType{};

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

protected:
    void preprocessImage(const cv::Mat& img, float* imgBufferArray) const;
    void preprocessImageCls(const cv::Mat& img, float* imgBufferArray) const;
    void createContextExecution();

    // OBJECT DETECTION METHODS
    static float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<Yolo::Detection>& res, float *output) const;
    static bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);
    cv::Rect getRect(cv::Mat& img, float bbox[4]);

    // CLASSIFICATION METHODS
    static std::vector<float> softmax(float *output_buffer, int n);
    static int getClasse(std::vector<float> &res, int n);
    static std::vector<int> topk(const std::vector<float>& vec, int k);

public:
    Detect();
    Detect(uint32_t modelType);
    ~Detect();

    std::vector<Yolo::Detection> doInference(cv::Mat& img);
    int doInferenceCls(cv::Mat& img);
};


#endif // DETECT_H
