//
// Created by willian on 30/09/23.
//

#include "include/color.h"

ColorCls::ColorCls() {
    this->inputH = 224;
    this->inputW = 224;
    this->numClasses = 7;
    this->confThresh = 0.001;

    createContextExecution();
}


void ColorCls::preprocessImage(const cv::Mat &img, float *imgBufferArray) {
    out = cv::Mat(inputH, inputW, CV_8UC3);
    cv::resize(img, out, out.size(), cv::INTER_LINEAR);
    int i = 0;
    for (int row = 0; row < inputH; ++row) {
        uchar* uc_pixel = out.data + row * out.step;
        for (int col = 0; col < inputW; ++col) {
            imgBufferArray[i] = (float)uc_pixel[2] / 255.0;
            imgBufferArray[i + inputH * inputW] = (float)uc_pixel[1] / 255.0;
            imgBufferArray[i + 2 * inputH * inputW] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
}

std::vector<Yolo::Detection> ColorCls::doInference(cv::Mat &img) {
    result.clear();

    preprocessImage(img, imgBuffer);

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], imgBuffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, nullptr));
    context->enqueue(batchSize, buffers.data(), nullptr, nullptr);

    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, nullptr));
    std::vector<float> output_model = this->softmax(outputBuffer, this->outputSize);

    int idx = this->getClasse(output_model, this->outputSize);

    result.emplace_back(
            Yolo::Detection{
                    {0,0,0,0},
                    0,
                    float(idx)}
    );

    return result;
}

void ColorCls::createContextExecution() {
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));

    this->imgBuffer = new float[batchSize * 3 * inputH * inputW];
    this->outputBuffer = new float[batchSize * outputSize];

    this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(color_engine, color_engine_len)));

    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));

    this->inputIndex = engine->getBindingIndex(inputBlobName);
    this->outputIndex = engine->getBindingIndex(outputBlobName);

    this->buffers = std::vector<void *>(engine->getNbBindings());
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * inputH * inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));
}
