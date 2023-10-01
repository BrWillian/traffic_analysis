//
// Created by willian on 22/08/23.
//

#ifndef TRAFFIC_ANALYSIS_CORE_H
#define TRAFFIC_ANALYSIS_CORE_H

#include "yololayer.h"
#include "detect.h"
#include "tracker.h"
#include "../meta/types.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <map>
#include <future>
#include "ocr.h"
#include "plate.h"
#include "color.h"
#include "vehicle.h"
#include "helmet.h"

class TrafficCore {
public:
    TrafficCore();
    TrafficCore(Detect *vehicleDet, Detect *plateDet, Detect *ocrDet, Detect *colorCls, Detect *HelmetDet, Tracker *trackerDet);
    ~TrafficCore();

    void checkLinePassage(std::vector<Vehicle::Detection>& detections);
    void getVehicles(cv::Mat &frame, std::vector<Vehicle::Detection>& detections);
    void getColors(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void getplateOcr(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void checkHelmet(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void setIdVehicles(std::vector<Vehicle::Detection>& vehicles);
    void setMargin(int margin);
    void setLines(std::vector<std::pair<cv::Point, cv::Point>> Lines);
    void setpermittedClasses(std::vector<std::vector<std::string>> permittedClasses);
    void parseConfig();


private:
    // CONSTRUCTORS
    Detect *vehicleDet;
    Detect *plateDet;
    Detect *ocrDet;
    Detect *colorCls;
    Detect *helmetDet;
    Tracker *trackerDet;


    int Margin;
    std::vector<std::pair<cv::Point, cv::Point>> Lines;
    std::vector<std::vector<std::string>> permittedClasses;
    Vehicle::Detection vehicle{};

    static float calculateIntercept(const cv::Point& point, float slope);
    static float calculateSlope(const cv::Point& point1, const cv::Point& point2);

    std::string getOcr(std::vector<Yolo::Detection> &plates, cv::Mat& plate, int* plate_roi, cv::Mat& frame);

    // TRACKER FUNCTIONS
    // static inline std::map<int, Yolo::Detection> appeared;

    // OCR FUNCTIONS
    static bool compareByLength(const Yolo::Detection& a, const Yolo::Detection& b);
    static bool compareByConfidence(const Yolo::Detection& a, const Yolo::Detection& b);
    static bool compareByHeight(const Yolo::Detection& a, const Yolo::Detection& b);
    static void checkBbox(cv::Rect& bbox,const cv::Mat& frame);

    // CLASSES
    static inline std::string color_classes[] = {"preta", "azul", "cinza", "verde", "vermelha", "branca", "amarela", "desconhecida"};
    static inline std::string ocr_classes[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    static inline std::string vehicle_classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};
};


#endif //TRAFFIC_ANALYSIS_CORE_H
