#include "../include/detect.h"
#include "../generated/weights.h"

Vehicle::Detect::Detect()
{
    cudaSetDevice(0);
    this->outputSize = 1000 * sizeof(Yolo::Detection) / sizeof(float) + 1;
    this->maxInputSize = 1920*1080;
    this->outputBlobName = "prob";
    this->inputBlobName = "data";
    this->confThresh = 0.5;
    this->nmsThresh = 0.4;
    this->batchSize = 1;
    this->inputH = 640;
    this->inputW = 640;
    this->numClasses = 6;
}

void Vehicle::Detect::preprocessImage(const cv::Mat &img, float *imgBufferArray) const{
    int w, h, x, y;
    float r_w = this->inputW / (img.cols*1.0);
    float r_h = this->inputH / (img.rows*1.0);
    if (r_h > r_w) {
        w = this->inputW;
        h = r_w * img.rows;
        x = 0;
        y = (this->inputH - h) / 2;
    } else {
        w = r_h* img.cols;
        h = this->inputH;
        x = (this->inputW - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat out(this->inputH, this->inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    for (int i = 0; i < this->inputH * this->inputW; i++) {
        imgBufferArray[3 * this->inputH * this->inputW + i] = out.at<cv::Vec3b>(i)[2] / 255.0;
        imgBufferArray[3 * this->inputH * this->inputW + i + this->inputH * this->inputW] = out.at<cv::Vec3b>(i)[1] / 255.0;
        imgBufferArray[3 * this->inputH * this->inputW + i + 2 * this->inputH * this->inputW] = out.at<cv::Vec3b>(i)[0] / 255.0;
    }
}

void Vehicle::Detect::createContextExecution(){
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));
    this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(vehicle_engine, vehicle_engine_len)));
    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));
}

cv::Mat Vehicle::Detect::doInference(cv::Mat &img){
    std::vector<void *> buffers(engine->getNbBindings());
    
    float imgBuffer[batchSize * 3 * inputH * inputW];
    float outputBuffer[batchSize * outputSize];
    
    preprocessImage(img, imgBuffer);
    
    const int inputIndex = engine->getBindingIndex(inputBlobName);
    const int outputIndex = engine->getBindingIndex(outputBlobName);
    
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * inputH * inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], img_buffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers.data(), stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}
float Vehicle::Detect::iou(std::array<float, _Tp2> lbox, std::array<float, _Tp2> rbox) const{

}
