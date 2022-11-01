#include "../include/detect.h"
#include "../generated/weights.h"
#include "../generated/version.h"

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
    float r_w = inputW / (img.cols*1.0);
    float r_h = inputH / (img.rows*1.0);
    if (r_h > r_w) {
        w = inputW;
        h = r_w * img.rows;
        x = 0;
        y = (inputH - h) / 2;
    } else {
        w = r_h* img.cols;
        h = inputH;
        x = (inputW - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    for (int i = 0; i < inputH * inputW; i++) {
        imgBufferArray[i] = out.at<cv::Vec3b>(i)[2] / 255.0;
        imgBufferArray[i + inputH * inputW] = out.at<cv::Vec3b>(i)[1] / 255.0;
        imgBufferArray[i + 2 * inputH * inputW] = out.at<cv::Vec3b>(i)[0] / 255.0;
    }
}

void Vehicle::Detect::createContextExecution(){
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));
    this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(vehicle_engine, vehicle_engine_len)));
    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));
}

std::vector<Yolo::Detection> Vehicle::Detect::doInference(cv::Mat &img){
    std::vector<Yolo::Detection> result{};
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

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], imgBuffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers.data(), stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    nms(result, outputBuffer);

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));

    return result;
}
float Vehicle::Detect::iou(float lbox[4], float rbox[4]){
    std::array<float, 4> interBox = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f),
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f),
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f),
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f),
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}
void Vehicle::Detect::nms(std::vector<Yolo::Detection>& res, float *output) const{
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + det_size * i + 4] <= confThresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nmsThresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}
bool Vehicle::Detect::cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}
const char* Vehicle::Detect::getVersion(){
    return VERSION "-" GIT_BRANCH "-" GIT_COMMIT_HASH;
}
const char* Vehicle::Detect::getWVersion(){
    return W_VERSION "-" W_HASH;
}
