//
// Created by willian on 01/10/23.
//

#ifndef TRAFFIC_ANALYSIS_HELMET_H
#define TRAFFIC_ANALYSIS_HELMET_H
#include "detect.h"
#include <vector>
#include "types.h"


class HelmetDet : public Detect {
public:
    HelmetDet();
    void preprocessImage(const cv::Mat& img, float* imgBufferArray) override;
    std::vector<Yolo::Detection> doInference(cv::Mat &img) override;
    void createContextExecution() override;
};
#endif //TRAFFIC_ANALYSIS_HELMET_H
