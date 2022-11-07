#ifndef POLYGON_H
#define POLYGON_H

#include <opencv2/opencv.hpp>

namespace Vehicle
{
    class Polygon
    {
    public:

        Polygon(std::vector<std::vector<cv::Point>> &polygons);
        std::vector<std::vector<int>> checkAreaBoxes(std::vector<std::vector<int>> &boxes);


    private:

        struct Line{
            cv::Point p1, p2;
        };

        std::vector<std::vector<cv::Point>> polygons{};
        bool checkInside(std::vector<cv::Point> poly, cv::Point point);
        bool onLine(Line l1, cv::Point p);
        int direction(cv::Point a, cv::Point b, cv::Point c);
        bool isIntersect(Line l1, Line l2);
    };
}

#endif // POLYGON_H
