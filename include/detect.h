#ifndef DETECT_H
#define DETECT_H

#include <iostream>
#include <memory>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>

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
        const char* inputBlobName{};
        const char* outputBlobName{};

        uint8_t device{};
        float nmsThresh{};
        float confThresh{};


        typedef struct {
            template<class T>
            void operator()(T* obj) const{
                delete obj;
            }
        }TRTDelete;

        template<class T>
        using TRTptr = std::unique_ptr<T, TRTDelete>;

        TRTptr<nvinfer1::IHostMemory> serializedModel{nullptr};
        TRTptr<nvinfer1::IRuntime> runtime{nullptr};
        TRTptr<nvinfer1::ICudaEngine> engine{nullptr};
        TRTptr<nvinfer1::IExecutionContext> context{nullptr};
        TRTptr<nvinfer1::IBuilder> builder{nullptr};
        TRTptr<nvinfer1::INetworkDefinition> network{nullptr};
        TRTptr<nvonnxparser::IParser> parser{nullptr};
        TRTptr<nvinfer1::IBuilderConfig> builderCfg;


        void preprocessImage(const cv::Mat& img, float* imgBufferArray) const;


    public:
        Detect();

        const char* getVersion(){
            return "1.0.0";
        }
        const char* getWVersion(){
            return "1.0.0";
        }

    };
}

#endif // DETECT_H
