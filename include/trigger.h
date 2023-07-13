//
// Created by willian on 26/04/23.
//

#ifndef TRAFFIC_ANALYSIS_TRIGGER_H
#define TRAFFIC_ANALYSIS_TRIGGER_H

#include "yololayer.h"
#include "detect.h"
#include <opencv2/opencv.hpp>
#include <map>

class Trigger {
public:
    static std::vector<bool> checkLinePassage(const std::vector<Yolo::Detection>& detections);
    static std::vector<Yolo::Detection> getVehicles(Detect &det, cv::Mat &frame);
    static std::vector<std::string> getplateOcr(Detect &plateDet, Detect &ocrDet, std::vector<bool>& trigged, std::vector<Yolo::Detection>& vehicles, cv::Mat &frame);
    //static void deleteFalseDetections(std::vector<Yolo::Detection>& detections, std::vector<bool>& indicesToRemove);
    //static std::map<int, Yolo::Detection> filterObjects(std::vector<Yolo::Detection> &vehicles, std::vector<std::pair<int, std::vector<float>>> &idx);

    static inline float Margin;
    static inline std::vector<std::pair<cv::Point, cv::Point>> Lines;
private:
    static std::vector<Yolo::Detection> objects;
    static inline std::vector<std::string> platesOcr;
    static float calculateIntercept(const cv::Point& point, float slope);
    static float calculateSlope(const cv::Point& point1, const cv::Point& point2);
    static std::string getOcr(Detect& ocrDet, std::vector<Yolo::Detection> &plates, cv::Mat& plate, cv::Mat& frame);

    // TRACKER FUNCTIONS
    static inline std::map<int, Yolo::Detection> appeared;

    // OCR FUNCTIONS
    static bool compareByLength(const Yolo::Detection& a, const Yolo::Detection& b);
    static bool compareByConfidence(const Yolo::Detection& a, const Yolo::Detection& b);
    static bool compareByHeight(const Yolo::Detection& a, const Yolo::Detection& b);
};


#endif //TRAFFIC_ANALYSIS_TRIGGER_H
