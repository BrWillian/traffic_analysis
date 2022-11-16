#ifndef TRACKER_H
#define TRACKER_H

#include <map>
#include <set>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>
#include "../include/yololayer.h"

namespace Vehicle
{
    class Tracker
    {
    public:
        explicit Tracker(int maxDisappeared);

        void register_Object(int cX, int cY);

        std::vector<std::pair<int, std::pair<int, int>>> update(std::vector<Yolo::Detection> &dets);

        // <ID, centroids>
        std::vector<std::pair<int, std::pair<int, int>>> objects;
        void deleteObject(int objectID);
    private:
        int maxDisappeared;

        int nextObjectID;

        static double calcDistance(double x1, double y1, double x2, double y2);

        // <ID, count>
        std::map<int, int> disappeared;

        //std::vector<float>::size_type findMin(const std::vector<float> &v, std::vector<float>::size_type pos = 0);
    };
}

#endif // TRACKER_H
