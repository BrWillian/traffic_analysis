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

cv::Scalar ColorDetector::extractDominantColor(const cv::Mat& img) {
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);

    int hbins = 180;
    int histSize[] = { hbins };
    float hranges[] = { 0, 180 };
    const float* ranges[] = { hranges };
    cv::MatND hist;
    int channels[] = { 0 };
    cv::calcHist(&hsvImage, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    double maxVal = 0;
    int maxIdx = 0;
    for (int i = 0; i < hbins; i++) {
        float binVal = hist.at<float>(i);
        if (binVal > maxVal) {
            maxVal = binVal;
            maxIdx = i;
        }
    }

    float hue = (maxIdx * 180) / hbins;

    return cv::Scalar(hue, 255, 255);
}