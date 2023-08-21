#include "../include/detect.h"

Detect::Detect()
{
    cudaSetDevice(0);
    this->outputSize = 1000 * sizeof(Yolo::Detection) / sizeof(float) + 1;
    this->maxInputSize = 1920*1080;
    this->outputBlobName = "output";
    this->inputBlobName = "images";
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
    this->outputBlobName = "output";
    this->inputBlobName = "images";
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
        case MODEL_TYPE_COLOR:
            this->numClasses = 7;
            this->outputSize = this->numClasses;
            this->inputW = 224;
            this->inputH = 224;
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
void Detect::preprocessImageCls(const cv::Mat &img, float *imgBufferArray) const {
    cv::Mat resized(inputH, inputW, CV_8UC3);
    cv::resize(img, resized, resized.size(), cv::INTER_LINEAR);

    int i = 0;
    for (int row = 0; row < inputH; ++row) {
        uchar* uc_pixel = resized.data + row * resized.step;
        for (int col = 0; col < inputW; ++col) {
            imgBufferArray[i] = (float)uc_pixel[2] / 255.0;
            imgBufferArray[i + inputH * inputW] = (float)uc_pixel[1] / 255.0;
            imgBufferArray[i + 2 * inputH * inputW] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
};

void Detect::createContextExecution(){
    this->runtime = static_cast<std::unique_ptr<nvinfer1::IRuntime, TRTDelete>>(std::move(nvinfer1::createInferRuntime(gLogger)));

    this->imgBuffer = new float[batchSize * 3 * inputH * inputW];
    this->outputBuffer = new float[batchSize * outputSize];

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
        case MODEL_TYPE_COLOR:
            this->engine = static_cast<std::unique_ptr<nvinfer1::ICudaEngine, TRTDelete>>(std::move(runtime->deserializeCudaEngine(color_engine, color_engine_len)));
            break;
    }
    this->context = static_cast<std::unique_ptr<nvinfer1::IExecutionContext, TRTDelete>>(std::move(engine->createExecutionContext()));
    
    this->inputIndex = engine->getBindingIndex(inputBlobName);
    this->outputIndex = engine->getBindingIndex(outputBlobName);
    
    this->buffers = std::vector<void *>(engine->getNbBindings());
    
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * inputH * inputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));       
}

std::vector<Yolo::Detection> Detect::doInference(cv::Mat &img){
    if (this->modelType == MODEL_TYPE_COLOR) {
        std::cerr<<"[ERROR] Invalid model type."<<std::endl;
        return std::vector<Yolo::Detection>{};
    }
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
        l = bbox[0];
        r = bbox[2];
        t = bbox[1] - (inputH - r_w * img.rows) / 2;
        b = bbox[3] - (inputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - (inputW - r_h * img.cols) / 2;
        r = bbox[2] - (inputW - r_h * img.cols) / 2;
        t = bbox[1];
        b = bbox[3];
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}
// CLASSIFICATION METHODS
std::vector<float> Detect::softmax(float *output_buffer, int n) {
    std::vector<float> res;
    float sum = 0.0f;
    float t;
    for (int i = 0; i < n; i++) {
        t = expf(output_buffer[i]);
        res.push_back(t);
        sum += t;
    }
    for (int i = 0; i < n; i++) {
        res[i] /= sum;
    }
    return res;
}
int Detect::getClasse(std::vector<float> &res, int n) {
    float maxScore = -1.0f;
    int maxIndex = -1;
    for (int i = 0; i < n; ++i) {
        if (res[i] > maxScore) {
            maxScore = res[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}
std::vector<int> Detect::topk(const std::vector<float>& vec, int k) {
    std::vector<int> topk_index;
    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2) { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), k);

    for (int i = 0; i < k_num; ++i) {
        topk_index.push_back(vec_index[i]);
    }

    return topk_index;
}
int Detect::doInferenceCls(cv::Mat &img){
    if (this->modelType != MODEL_TYPE_COLOR) {
        std::cerr<<"[ERROR] Invalid model type."<<std::endl;
        return 7;
    }

    preprocessImageCls(img, imgBuffer);

    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], imgBuffer, batchSize * 3 * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, nullptr));
    context->enqueue(batchSize, buffers.data(), nullptr, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(outputBuffer, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, nullptr));

    std::vector<float> output_model = this->softmax(outputBuffer, this->outputSize);
    //std::vector<int> idxs = this->topk(output_model, 1);
    int idx = this->getClasse(output_model, this->outputSize);

    return idx;
}