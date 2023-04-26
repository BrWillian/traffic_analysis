#ifndef POLYGON_H
#define POLYGON_H

#include <opencv2/opencv.hpp>
#include "../include/yololayer.h"

class Polygon
{
public:

    Polygon(std::vector<std::vector<cv::Point>> &polygons);
    std::vector<Yolo::Detection> checkAreaBoxes(std::vector<Yolo::Detection> &boxes);


private:

    struct Line{
        cv::Point p1, p2;
    };

    std::vector<std::vector<cv::Point>> polygons{};
    static bool checkInside(std::vector<cv::Point> poly, cv::Point &point);
    static bool onLine(Line l1, cv::Point p);
    static int direction(cv::Point a, cv::Point b, cv::Point c);
    static bool isIntersect(Line l1, Line l2);
};

#endif // POLYGON_H
