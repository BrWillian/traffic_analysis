//
// Created by willian on 26/04/23.
//

#include "../include/trigger.h"


float Trigger::calculateSlope(const cv::Point& point1, const cv::Point& point2) {
    return (point2.y - point1.y) / (point2.x - point1.x);
}

float Trigger::calculateIntercept(const cv::Point& point, float slope) {
    return point.y - slope * point.x;
}

std::vector<bool> Trigger::checkLinePassage(const std::vector<Yolo::Detection>& detections) {
    std::vector<bool> results;

    for (const auto& detection : detections) {
        // Calculate the centroid of the bounding box
        float centroid_x = (detection.bbox[0] + detection.bbox[2] + detection.bbox[0]) / 2;
        float centroid_y = (detection.bbox[1] + detection.bbox[3] + detection.bbox[1]) / 2;

        bool passedLine = false;

        for (const auto& line : Trigger::Lines) {
            const cv::Point& line_point1 = line.first;
            const cv::Point& line_point2 = line.second;

            float line_slope = Trigger::calculateSlope(line_point1, line_point2);
            float line_intercept = Trigger::calculateIntercept(line_point1, line_slope);
            float line_min_x = std::min(line_point1.x, line_point2.x);
            float line_max_x = std::max(line_point1.x, line_point2.x);

            float expected_y = line_slope * centroid_x + line_intercept;

            if (centroid_x >= line_min_x && centroid_x <= line_max_x && std::abs(centroid_y - expected_y) <= Trigger::Margin) {
                passedLine = true;
                break;
            }
        }

        results.push_back(passedLine);
    }

    return results;
}

std::vector<Yolo::Detection> Trigger::getVehicles(Detect &det, cv::Mat &frame) {
    return det.doInference(frame);
}

std::vector<std::string> Trigger::getplateOcr(Detect &plateDet, Detect &ocrDet, std::vector<bool>& trigged, std::vector<Yolo::Detection>& vehicles, cv::Mat &frame) {
    platesOcr.clear();

    vehicles.erase(std::remove_if(vehicles.begin(), vehicles.end(), [&](const Yolo::Detection& vehicle) {
        size_t i = &vehicle - &vehicles[0];
        return !trigged[i];
    }), vehicles.end());

    for(auto & vehicle : vehicles){
        cv::Rect r(vehicle.bbox[0], vehicle.bbox[1], vehicle.bbox[2], vehicle.bbox[3]);
        int x = std::max(r.x, 0);
        int y = std::max(r.y, 0);
        int width = std::min(r.x + r.width, frame.cols) - x;
        int height = std::min(r.y + r.height, frame.rows) - y;
        cv::Rect adjustedRect(x, y, width, height);
        cv::Mat image_roi = frame(adjustedRect);
        std::vector<Yolo::Detection> plates = plateDet.doInference(image_roi);
        std::string plate = Trigger::getOcr(ocrDet, plates, image_roi, frame);
        platesOcr.emplace_back(plate);
    }

    return platesOcr;
}


std::string Trigger::getOcr(Detect& ocrDet, std::vector<Yolo::Detection> &plates, cv::Mat &plate, cv::Mat &frame) {
    if (plates.size() == 0) {
        return "";
    }

    const char* classes[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

    for (size_t i = 0; i < plates.size(); i++) {
        std::string newPlate = "";
        cv::Rect r(plates[i].bbox[0], plates[i].bbox[1], plates[i].bbox[2], plates[i].bbox[3]);

        r.x = std::max(0, r.x - r.width / 5);
        r.y = std::max(0, r.y - (int)(r.height / 1.5));
        r.width += r.width * 2 / 5;
        r.height += r.height * (2 / 1.5);
        r.height = std::min(r.height, frame.rows - r.y);
        r.width = std::min(r.width, frame.cols - r.x);

        cv::Mat image_roi = plate(r);
        std::vector<Yolo::Detection> chars = ocrDet.doInference(image_roi);

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

        for (size_t j = 0; j < plateTmp.size(); j++) {
            newPlate += classes[(int)plateTmp[j].class_id];
        }
        return newPlate;
    }

    return "";
}


//void Trigger::deleteFalseDetections(std::vector<Yolo::Detection>& detections, std::vector<bool>& indicesToRemove) {
//    auto removeBegin = std::remove_if(detections.begin(), detections.end(), [&](const Yolo::Detection& detection) {
//        std::size_t index = &detection - &detections[0];
//        return !indicesToRemove[index];
//    });
//
//    detections.erase(removeBegin, detections.end());
//    indicesToRemove.erase(indicesToRemove.begin() + (removeBegin - detections.begin()), indicesToRemove.end());
//}

bool Trigger::compareByLength(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.bbox[0] < b.bbox[0];
}

bool Trigger::compareByConfidence(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.conf > b.conf;
}

bool Trigger::compareByHeight(const Yolo::Detection &a, const Yolo::Detection &b) {
    return a.bbox[1] < b.bbox[1];
}

//std::map<int, Yolo::Detection>
//Trigger::filterObjects(std::vector<Yolo::Detection> &vehicles, std::vector<std::pair<int, std::vector<float>>>& idx) {
//
//}

