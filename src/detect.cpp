#include "../include/detect.h"
#include "../generated/weights.h"

Vehicle::Detect::Detect()
{
    cudaSetDevice(0);
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
