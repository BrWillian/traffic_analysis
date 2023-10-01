#include "../include/detect.h"

Detect::~Detect(){
    this->runtime.reset(nullptr);
    this->context.reset(nullptr);
    this->engine.reset(nullptr);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
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
    for (int i = 0; i < output[0]; i++) {
        if (output[1 + det_size * i + 4] <= confThresh) continue;
        Yolo::Detection det{};
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto & it : m) {
        auto& dets = it.second;
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
cv::Rect Detect::getRect(cv::Mat &img, float bbox[4]) const {
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