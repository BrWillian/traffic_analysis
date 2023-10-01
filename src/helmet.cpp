//
// Created by willian on 01/10/23.
//

#include "include/helmet.h"

HelmetDet::HelmetDet() {
    this->inputH = 320;
    this->inputW = 320;
    this->numClasses = 2;
    this->confThresh = 0.5;
    this->nmsThresh = 0.4;

    createContextExecution();
}

void HelmetDet::preprocessImage(const cv::Mat &img, float *imgBufferArray) {
    r_w = inputW / (img.cols*1.0);
    r_h = inputH / (img.rows*1.0);

    if (r_h > r_w) {
        roi.width = inputW;
        roi.height = r_w * img.rows;
        roi.x = 0;
        roi.y = (inputH - roi.height) / 2;
    } else {
        roi.width = r_h* img.cols;
        roi.height = inputH;
        roi.x = (inputW - roi.width) / 2;
        roi.y = 0;
    }
    out = cv::Mat(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::resize(img, out(roi), roi.size(), cv::INTER_LINEAR);
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

std::vector<Yolo::Detection> HelmetDet::doInference(cv::Mat &img) {
    result.clear();

    this->preprocessImage(img, imgBuffer);

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], imgBuffer, batchSize * 3 * inputH * inputW * sizeof(float),
                               cudaMemcpyHostToDevice, nullptr));

    context->enqueue(batchSize, buffers.data(), nullptr, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, nullptr));

    nms(result, outputBuffer);

    for(auto &it: result){
        cv::Rect r = getRect(img, it.bbox);
        it.bbox[0] = r.x;
        it.bbox[1] = r.y;
        it.bbox[2] = r.width;
        it.bbox[3] = r.height;
    }

    return result;
}

void HelmetDet::createContextExecution() {
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));

    this->imgBuffer = new float[batchSize * 3 * inputH * inputW];
    this->outputBuffer = new float[batchSize * outputSize];

    this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(helmet_engine, helmet_engine_len)));

    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));

    this->inputIndex = engine->getBindingIndex(inputBlobName);
    this->outputIndex = engine->getBindingIndex(outputBlobName);

    this->buffers = std::vector<void *>(engine->getNbBindings());
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * inputH * inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));
}
