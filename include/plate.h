//
// Created by willian on 30/09/23.
//

#ifndef TRAFFIC_ANALYSIS_PLATE_H
#define TRAFFIC_ANALYSIS_PLATE_H
#include "detect.h"

class PlateDet : public Detect{
public:
    PlateDet();
    void preprocessImage(const cv::Mat& img, float* imgBufferArray) override;
    std::vector<Yolo::Detection> doInference(cv::Mat &img) override;
    void createContextExecution() override;
};


#endif //TRAFFIC_ANALYSIS_PLATE_H
