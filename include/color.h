//
// Created by willian on 13/07/23.
//

#ifndef TRAFFIC_ANALYSIS_COLOR_H
#define TRAFFIC_ANALYSIS_COLOR_H


#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "yololayer.h"

struct SpecificColor
{
    std::string color;
    cv::Scalar lowerBound;
    cv::Scalar upperBound;
};

class ColorDetector {
public:
    ColorDetector() {}
    std::vector<std::string> detectSpecificColor(const std::vector<Yolo::Detection>& detections, const cv::Mat& img);
private:
    cv::Scalar extractDominantColor(const cv::Mat& image);
    std::string detectSpecificColor(const cv::Scalar& color);

    std::vector<SpecificColor> specificColors = {
            {"Amarelo", cv::Scalar(20, 100, 100), cv::Scalar(40, 255, 255)},
            {"Branco", cv::Scalar(0, 0, 200), cv::Scalar(180, 40, 255)},
            {"Prata", cv::Scalar(0, 0, 120), cv::Scalar(180, 40, 200)},
            {"Preto", cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 70)},
            {"Cinza", cv::Scalar(0, 0, 70), cv::Scalar(180, 40, 120)},
            {"Vermelho", cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255)},
            {"Azul", cv::Scalar(90, 100, 100), cv::Scalar(130, 255, 255)},
            {"Marrom", cv::Scalar(0, 100, 20), cv::Scalar(20, 255, 200)},
            {"Verde", cv::Scalar(40, 100, 100), cv::Scalar(70, 255, 255)}
    };
};


#endif //TRAFFIC_ANALYSIS_COLOR_H
