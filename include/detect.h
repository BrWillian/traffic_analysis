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
#include "yololayer.h"
#include <fstream>
#include <string>
#include <vector>
#include "utils.h"
#include <assert.h>
#include <memory>
#include "../generated/weights.h"
#include "../meta/types.h"
#include "../meta/logger.hpp"

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

class Detect
{
public:
    Detect() = default;
    ~Detect();

    virtual std::vector<Yolo::Detection> doInference(cv::Mat& img) = 0;

protected:
    static constexpr uint8_t batchSize = 1;
    static constexpr char* inputBlobName = "images";
    static constexpr char* outputBlobName = "output";
    static constexpr uint32_t outputSize = 100 * sizeof(Yolo::Detection) / sizeof(float) + 1;

    float nmsThresh{};
    float confThresh{};
    uint16_t inputH{};
    uint16_t inputW{};
    uint8_t numClasses{};

    float r_w{};
    float r_h{};
    cv::Rect roi{};
    cv::Mat out{};
    std::vector<Yolo::Detection> result{};

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

    virtual void preprocessImage(const cv::Mat& img, float* imgBufferArray) = 0;
    virtual void createContextExecution() = 0;

    // OBJECT DETECTION METHODS
    static float iou(float lbox[4], float rbox[4]);
    void nms(std::vector<Yolo::Detection>& res, float *output) const;
    static bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);
    cv::Rect getRect(cv::Mat& img, float bbox[4]) const;

    // CLASSIFICATION METHODS
    static std::vector<float> softmax(float *output_buffer, int n);
    int getClasse(std::vector<float> &res, int n);
    static std::vector<int> topk(const std::vector<float>& vec, int k);
};


#endif // DETECT_H
