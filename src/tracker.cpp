#include "../include/tracker.h"

Vehicle::Tracker::Tracker(int maxDisappeared)
{
    this->nextObjectID = 0;
    this->maxDisappeared = maxDisappeared;
}
double Vehicle::Tracker::calcDistance(double x1, double y, double x2, double y2){
    double x = x1 - x2;
    double y = y1 - y2;
    return std::sqrt((x * x) + (y * y));
}
void Vehicle::Tracker::registerObj(int cX, int cY){
    int objID = this->nextObjectID;
    this->objects.push_back({objID, {cX, cY}});
    this->disappeared.insert({objID, 0});
    this->nextObjectID += 1;
}
