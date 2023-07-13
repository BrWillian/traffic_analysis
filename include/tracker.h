#ifndef TRACKER_H
#define TRACKER_H

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "../include/yololayer.h"

class Tracker {
public:
    explicit Tracker(int maxDisappeared);

    void register_Object(const std::vector<float>& bbox);
    void deleteObject(int objectID);
    std::vector<std::pair<int, std::vector<float>>> update(std::vector<Yolo::Detection>& detections);

private:
    double calcIoU(const std::vector<float>& bbox1, const std::vector<float>& bbox2);

    int nextObjectID;
    int maxDisappeared;
    std::vector<std::pair<int, std::vector<float>>> objects;
    std::unordered_map<int, int> disappeared;
};

#endif // TRACKER_H
