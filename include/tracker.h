#ifndef TRACKER_H
#define TRACKER_H

#include <map>
#include <cmath>
#include <vector>

namespace Vehicle
{
    class Tracker
    {
    public:
        Tracker(int maxDisappeared);

        void registerObj(int cX, int cY);

        std::vector<std::pair<int, std::pair<int, int>>> objects;

        std::map<int, std::vector<std::pair<int, int>>> path_keeper;

        std::vector<std::pair<int, std::pair<int, int>>> update(std::vector<std::vector<int>> boxes);
    private:
        int nextObjectID;

        int maxDisappeared;

        std::map<int, int> disappeared;

        static double calcDistance(double x1, double y, double x2, double y2);
    };
}

#endif // TRACKER_H
