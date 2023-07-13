#include "../include/detect.h"
#include "../generated/weights.h"

Detect::Detect()
{
    cudaSetDevice(0);
    this->outputSize = 1000 * sizeof(Yolo::Detection) / sizeof(float) + 1;
    this->maxInputSize = 1920*1080;
    this->outputBlobName = "prob";
    this->inputBlobName = "data";
    this->confThresh = 0.1;
    this->nmsThresh = 0.1;
    this->batchSize = 1;
    this->inputH = 640;
    this->inputW = 640;
    this->numClasses = 6;

    createContextExecution();
}

Detect::Detect(uint32_t modelType) {
    cudaSetDevice(0);
    this->maxInputSize = 1920*1080;
    this->outputBlobName = "prob";
    this->inputBlobName = "data";
    this->confThresh = 0.4;
    this->nmsThresh = 0.5;
    this->batchSize = 1;
    this->modelType = modelType;

    switch(modelType){
        case MODEL_TYPE_VEHICLE:
            this->outputSize = 50 * sizeof(Yolo::Detection) / sizeof(float) + 1;
            this->inputH = 640;
            this->inputW = 640;
            this->numClasses = 6;
            break;
        case MODEL_TYPE_PLATE:
            this->outputSize = 10 * sizeof(Yolo::Detection) / sizeof(float) + 1;
            this->inputH = 320;
            this->inputW = 320;
            this->numClasses = 1;
            break;
        case MODEL_TYPE_OCR:
            this->outputSize = 100 * sizeof(Yolo::Detection) / sizeof(float) + 1;
            this->inputH = 320;
            this->inputW = 320;
            this->numClasses = 36;
            this->confThresh = 0.1;
            this->nmsThresh = 0.8;
            break;
    }

    createContextExecution();
}

Detect::~Detect(){
    this->runtime.reset(nullptr);
    this->context.reset(nullptr);
    this->engine.reset(nullptr);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}
void Detect::preprocessImage(const cv::Mat &img, float *imgBufferArray) const{

    float r_w = inputW / (img.cols*1.0);
    float r_h = inputH / (img.rows*1.0);
    cv::Rect roi;
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
    cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::resize(img, out(roi), roi.size(), cv::INTER_NEAREST);

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

void Detect::createContextExecution(){
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));

    switch(this->modelType){
        case MODEL_TYPE_VEHICLE:
            this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(vehicle_engine, vehicle_engine_len)));
            break;
        case MODEL_TYPE_PLATE:
            this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(plate_engine, plate_engine_len)));
            break;
        case MODEL_TYPE_OCR:
            this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(ocr_engine, ocr_engine_len)));
            break;
    }
    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));

    this->imgBuffer = new float[batchSize * 3 * inputH * inputW];
    this->outputBuffer = new float[batchSize * outputSize];
    
    this->inputIndex = engine->getBindingIndex(inputBlobName);
    this->outputIndex = engine->getBindingIndex(outputBlobName);
    
    this->buffers = std::vector<void *>(engine->getNbBindings());
    
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * inputH * inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));       
}

std::vector<Yolo::Detection> Detect::doInference(cv::Mat &img){
    std::vector<Yolo::Detection> result{};
 
    preprocessImage(img, imgBuffer);

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], imgBuffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, nullptr));
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
float Detect::iou(float lbox[4], float rbox[4]){
    float interBox[] = {
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
void Detect::nms(std::vector<Yolo::Detection>& res, float *output) const{
    int det_size = sizeof(Yolo::Detection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 20; i++) {
        if (output[1 + det_size * i + 4] <= confThresh) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
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
bool Detect::cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}
cv::Rect Detect::getRect(cv::Mat &img, float bbox[4]){
    float l, r, t, b;
    float r_w = inputW / (img.cols * 1.0);
    float r_h = inputH / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (inputH - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (inputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (inputW - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (inputW - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}