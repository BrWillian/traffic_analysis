//
// Created by willian on 13/07/23.
//

#include "include/color.h"

std::vector<std::string> ColorDetector::detectSpecificColor(const std::vector<Yolo::Detection>& detections, const cv::Mat& img){
    std::vector<std::string> colors;

    for (const auto& detection : detections) {
        float x = detection.bbox[0];
        float y = detection.bbox[1];
        float width = detection.bbox[2];
        float height = detection.bbox[3];

        cv::Rect bbox(static_cast<int>(x), static_cast<int>(y), static_cast<int>(width), static_cast<int>(height));
        cv::Mat croppedImage = img(bbox);

        cv::Scalar dominantColor = extractDominantColor(croppedImage);

        std::string specificColor = detectSpecificColor(dominantColor);

        colors.push_back(specificColor);
    }

    return colors;
}

