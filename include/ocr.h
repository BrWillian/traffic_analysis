//
// Created by willian on 30/09/23.
//

#ifndef TRAFFIC_ANALYSIS_OCR_H
#define TRAFFIC_ANALYSIS_OCR_H
#include "detect.h"
#include <vector>
#include "../meta/types.h"

class OcrDet : public Detect {
public:
    OcrDet();
    void preprocessImage(const cv::Mat& img, float* imgBufferArray) override;
    std::vector<Yolo::Detection> doInference(cv::Mat &img) override;
    void createContextExecution() override;
};


#endif //TRAFFIC_ANALYSIS_OCR_H
