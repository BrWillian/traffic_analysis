#ifndef TRAFFIC_ANALYSIS_TRACKER_H
#define TRAFFIC_ANALYSIS_TRACKER_H

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "yololayer.h"
#include "../meta/types.h"

class Tracker {
public:
    Tracker();

    void register_Object(const std::vector<float>& bbox);
    void deleteObject(int objectID);
    void update(std::vector<Vehicle::Detection>& detections);

private:
    double calcIoU(const std::vector<float>& bbox1, const std::vector<float>& bbox2);

    int nextObjectID;
    int maxDisappeared;
    std::vector<std::pair<int, std::vector<float>>> objects;

    std::unordered_map<int, int> disappeared;
};

#endif //TRAFFIC_ANALYSIS_TRACKER_H
