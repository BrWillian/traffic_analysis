//
// Created by willian on 22/08/23.
//

#ifndef TRAFFIC_ANALYSIS_CORE_H
#define TRAFFIC_ANALYSIS_CORE_H

#include "yololayer.h"
#include "tracker.h"
#include "../meta/types.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "ocr.h"
#include "plate.h"
#include "color.h"
#include "vehicle.h"
#include "brand.h"

class TrafficCore {
public:
    TrafficCore();
    TrafficCore(Detect *vehicleDet, Detect *plateDet, Detect *ocrDet, Detect *colorCls, Detect *brandCls, Tracker *trackerDet);
    ~TrafficCore();

    std::map<uint16_t, Vehicle::Detection> vehiclesTrigger;

    // Core
    void checkLinePassage(std::vector<Vehicle::Detection>& detections);
    void getVehicles(cv::Mat &frame, std::vector<Vehicle::Detection>& detections);
    void getColors(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void getBrands(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void getplateOcr(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame);
    void setIdVehicles(std::vector<Vehicle::Detection>& vehicles);

    // Get Passage
    void getTriggeds(std::vector<Vehicle::Detection>& vehicles);
    bool isVehicleDetectionEmpty(const Vehicle::Detection &detection);

    // Set Configuration
    void setMargin(int margin);
    void setLines(std::vector<std::pair<cv::Point, cv::Point>> Lines);
    void setpermittedClasses(std::vector<std::vector<std::string>> permittedClasses);
    void parseConfig();

    // Set Configuration Check Vehicle Stop
    void setPolygons(std::vector<std::vector<cv::Point>> polygons);
    void setStopTime(int time);

    // Check Vehicle Stopping
    void checkPolyInside(std::vector<Vehicle::Detection>& detections);
    bool isVehicleInPolygon(const std::vector<cv::Point>& polygon,  const Vehicle::Detection& vehicle);



private:
    // CONSTRUCTORS
    Detect *vehicleDet;
    Detect *plateDet;
    Detect *ocrDet;
    Detect *colorCls;
    Detect *brandCls;
    Tracker *trackerDet;


    int Margin;
    std::vector<std::vector<cv::Point>> Polygons;
    int stopTime;
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
    static inline std::string color_classes[] = {"desconhecida", "preta", "azul", "cinza", "verde", "vermelha", "branca", "amarela"};
    static inline std::string ocr_classes[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    static inline std::string vehicle_classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};
    static inline std::string vehicle_brands[] = {"desconhecido", "chevrolet_onix","citroen_c3","fiat_palio","fiat_siena","fiat_strada","fiat_toro","fiat_uno","ford_ecosport","ford_fiesta","ford_ka","ford_ranger","gm_agile","gm_celta","gm_classic","gm_corsa","gm_cruze","gm_montana","gm_prisma","gm_s10","gm_vectra","honda_civic","honda_fit","hyundai_hb20","mmc_l200","peugeot_206","peugeot_208","renault_duster","renault_logan","renault_sandero","toyota_corolla","toyota_etios","toyota_hilux","vw_amarok","vw_crossfox","vw_fox","vw_gol","vw_golf","vw_jetta","vw_parati","vw_polo","vw_saveiro","vw_up","vw_voyage"};
};


#endif //TRAFFIC_ANALYSIS_CORE_H
