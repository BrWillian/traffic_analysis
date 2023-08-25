//
// Created by willian on 22/08/23.
//

#include <utility>

#include "include/core.h"

//TrafficCore::TrafficCore(Detect *vehicleDet, Detect *plateDet, Detect *ocrDet, Detect *colorCls, Tracker* trackerDet) {
//    this->vehicleDet = vehicleDet;
//    this->plateDet = plateDet;
//    this->ocrDet = ocrDet;
//    this->colorCls = colorCls;
//    this->trackerDet = trackerDet;
//}
TrafficCore::TrafficCore() {
    this->vehicleDet = new Detect(MODEL_TYPE_VEHICLE);
    this->plateDet = new Detect(MODEL_TYPE_PLATE);
    this->ocrDet = new Detect(MODEL_TYPE_OCR);
    this->colorCls = new Detect(MODEL_TYPE_COLOR);
    this->trackerDet = new Tracker();
}

TrafficCore::~TrafficCore() {
    delete this->vehicleDet;
    delete this->plateDet;
    delete this->ocrDet;
    delete this->colorCls;
    delete this->trackerDet;
}

void TrafficCore::checkLinePassage(std::vector<Vehicle::Detection>& detections) {
    for (auto vehicle = detections.begin(); vehicle != detections.end();) {
        bool passedLine = false;
        int lineIndex = -1;

        for (size_t i = 0; i < this->Lines.size(); ++i) {
            const auto &line = TrafficCore::Lines[i];
            const cv::Point &line_point1 = line.first;
            const cv::Point &line_point2 = line.second;

            float line_slope = TrafficCore::calculateSlope(line_point1, line_point2);
            float line_intercept = TrafficCore::calculateIntercept(line_point1, line_slope);
            float line_min_x = std::min(line_point1.x, line_point2.x);
            float line_max_x = std::max(line_point1.x, line_point2.x);

            float expected_y = line_slope * vehicle->centroid.x + line_intercept;

            if (vehicle->centroid.x >= line_min_x && vehicle->centroid.x <= line_max_x &&
                std::abs(vehicle->centroid.y - expected_y) <= this->Margin) {
                passedLine = true;
                lineIndex = static_cast<int>(i);
                break;
            }
        }

        vehicle->faixa = static_cast<int>(lineIndex);

        if (!passedLine) {
            vehicle = detections.erase(vehicle);
        } else {
            ++vehicle;
        }
    }
}

float TrafficCore::calculateIntercept(const cv::Point &point, float slope) {
    return point.y - slope * point.x;
}

float TrafficCore::calculateSlope(const cv::Point &point1, const cv::Point &point2) {
    return (point2.y - point1.y) / (point2.x - point1.x);
}

void TrafficCore::getVehicles(cv::Mat &frame, std::vector<Vehicle::Detection>& detections) {
    detections.clear();

    std::vector<Yolo::Detection> yoloVehicles = this->vehicleDet->doInference(frame);
    detections.reserve(yoloVehicles.size());

    for (const auto& yoloVehicle : yoloVehicles) {
        cv::Rect r(yoloVehicle.bbox[0], yoloVehicle.bbox[1], yoloVehicle.bbox[2], yoloVehicle.bbox[3]);
        TrafficCore::checkBbox(r, frame);
        vehicle.bbox[0] = r.x; vehicle.bbox[1] = r.y; vehicle.bbox[2] = r.width; vehicle.bbox[3] = r.height;
        vehicle.centroid = {r.x + r.width / 2, r.y + r.height / 2};
        vehicle.class_name = vehicle_classes[static_cast<int>(yoloVehicle.class_id)];

        detections.emplace_back(std::move(vehicle));
    }
}

void TrafficCore::getColors(std::vector<Vehicle::Detection>& vehicles, cv::Mat &frame) {
    for(auto & vehicle : vehicles){
        cv::Rect r(vehicle.bbox[0], vehicle.bbox[1], vehicle.bbox[2], vehicle.bbox[3]);
        TrafficCore::checkBbox(r, frame);
        cv::Mat image_roi = frame(r);

        int vehicle_color = this->colorCls->doInferenceCls(image_roi);
        vehicle.color = color_classes[(int)vehicle_color];
    }
}

void TrafficCore::getplateOcr(std::vector<Vehicle::Detection> &vehicles, cv::Mat &frame) {
    for(auto & vehicle : vehicles){
        cv::Rect r(vehicle.bbox[0], vehicle.bbox[1], vehicle.bbox[2], vehicle.bbox[3]);
        TrafficCore::checkBbox(r, frame);
        cv::Mat image_roi = frame(r);
        std::vector<Yolo::Detection> plates = plateDet->doInference(image_roi);
        if(!plates.empty()){
            int plate_roi[4];
            std::string plate = this->getOcr(plates, image_roi, plate_roi, frame);
            vehicle.ocr = plate;
            vehicle.plate = true;
            vehicle.plate_bbox[0] = plate_roi[0] + r.x;
            vehicle.plate_bbox[1] = plate_roi[1] + r.y;
            vehicle.plate_bbox[2] = plate_roi[2];
            vehicle.plate_bbox[3] = plate_roi[3];

        }else {
            vehicle.plate = false;
        }

    }
}

std::string TrafficCore::getOcr(std::vector<Yolo::Detection> &plates, cv::Mat &plate, int* plate_roi, cv::Mat &frame) {
    if (plates.empty()) {
        return "";
    }
    for (auto & i : plates) {

        std::string newPlate;
        cv::Rect r(i.bbox[0], i.bbox[1], i.bbox[2], i.bbox[3]);
        r.x = std::max(0, r.x - r.width / 5);
        r.y = std::max(0, r.y - (int)(r.height / 1.5));
        r.width += r.width * 2 / 5;
        r.height += r.height * (2 / 1.5);
        r.height = std::min(r.height, frame.rows - r.y);
        r.width = std::min(r.width, frame.cols - r.x);
        TrafficCore::checkBbox(r, frame);

        cv::Mat image_roi = plate(r);
        TrafficCore::cvtPlate(image_roi);

        std::vector<Yolo::Detection> chars = this->ocrDet->doInference(image_roi);
        plate_roi[0] = r.x; plate_roi[1] = r.y; plate_roi[2] = r.width; plate_roi[3] = r.height;
        if (chars.size() < 7) {
            continue;
        }

        std::sort(chars.begin(), chars.end(), compareByConfidence);
        int counter = std::min(7, (int)chars.size());
        std::vector<Yolo::Detection> plateTmp;

        for (int j = 0; j < counter; j++) {
            plateTmp.push_back(chars[j]);
        }

        if (image_roi.cols < image_roi.rows * 1.4) {
            std::sort(plateTmp.begin(), plateTmp.end(), compareByHeight);
            std::vector<Yolo::Detection> topRow;
            std::vector<Yolo::Detection> bottomRow;

            topRow.insert(topRow.end(), plateTmp.begin(), plateTmp.begin() + 3);
            bottomRow.insert(bottomRow.end(), plateTmp.end() - 4, plateTmp.end());

            std::sort(topRow.begin(), topRow.end(), compareByLength);
            std::sort(bottomRow.begin(), bottomRow.end(), compareByLength);

            plateTmp.clear();
            for(auto a: topRow){
                plateTmp.push_back(a);
            }
            for(auto a: bottomRow){
                plateTmp.push_back(a);
            }
        } else {
            std::sort(plateTmp.begin(), plateTmp.end(), compareByLength);
        }

        for (auto & j : plateTmp) {
            newPlate += ocr_classes[(int)j.class_id];
        }
        return newPlate;
    }

    return "";
}

bool TrafficCore::compareByLength(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.bbox[0] < b.bbox[0];
}

bool TrafficCore::compareByConfidence(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.conf > b.conf;
}

bool TrafficCore::compareByHeight(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.bbox[1] < b.bbox[1];
}

void TrafficCore::setMargin(int margin) {
    this->Margin = margin;
}
void TrafficCore::setLines(std::vector<std::pair<cv::Point, cv::Point>> lines) {
    this->Lines = std::move(lines);
}

void TrafficCore::setIdVehicles(std::vector<Vehicle::Detection> &vehicles) {
    this->trackerDet->update(vehicles);
}

void TrafficCore::parseConfig() {
    std::ifstream file("config.yaml");
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo de configuração." << std::endl;
        exit(1);
    }
    YAML::Node root = YAML::Load(file);

    int margem;
    std::vector<std::pair<cv::Point, cv::Point>> lines;

    margem = root["margem"].as<int>();

    YAML::Node faixas = root["faixas"];
    for (const auto& faixa : faixas) {
        auto nome = faixa["nome"].as<std::string>();

        YAML::Node pt1 = faixa["pt1"];
        YAML::Node pt2 = faixa["pt2"];

        cv::Point pt1_xy(pt1[0].as<int>(), pt1[1].as<int>());
        cv::Point pt2_xy(pt2[0].as<int>(), pt2[1].as<int>());
        lines.emplace_back(pt1_xy, pt2_xy);

        std::cout << "Faixa: " << nome << std::endl;
        std::cout << "Ponto 1: (" << pt1[0].as<int>() << ", " << pt1[1].as<int>() << ")" << std::endl;
        std::cout << "Ponto 2: (" << pt2[0].as<int>() << ", " << pt2[1].as<int>() << ")" << std::endl;
        std::cout << std::endl;
    }
    setMargin(margem);
    setLines(lines);
}

void TrafficCore::checkBbox(cv::Rect& bbox, const cv::Mat& frame) {
    int x = std::max(bbox.x, 0);
    int y = std::max(bbox.y, 0);
    int width = std::min(bbox.width, frame.cols - x);
    int height = std::min(bbox.height, frame.rows - y);
    bbox = (width <= 0 || height <= 0) ? cv::Rect(1,1,1,1) : cv::Rect(x, y, width, height);
}

void TrafficCore::cvtPlate(cv::Mat &image_roi) {
    cv::Mat image_roi_grayscale;
    cv::cvtColor(image_roi, image_roi_grayscale, cv::COLOR_RGB2GRAY);
    cv::Mat gray3Channel(image_roi.size(), CV_8UC3);
    cv::Mat grayChannels[] = { image_roi_grayscale, image_roi_grayscale, image_roi_grayscale };
    cv::merge(grayChannels, 3, gray3Channel);
    gray3Channel.copyTo(image_roi);
}
