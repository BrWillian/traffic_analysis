#include "../include/polygon.h"

Polygon::Polygon(std::vector<std::vector<cv::Point>>& polygons){
    this->polygons = polygons;
}
bool Polygon::onLine(Line l1, cv::Point p){
    if(p.x <= std::max(l1.p1.x, l1.p2.x)
            && p.x <= std::min(l1.p1.x, l1.p2.x)
            && (p.y <= std::max(l1.p1.y, l1.p2.y)
                && p.y <= std::min(l1.p1.y, l1.p2.y)))
        return true;
    return false;
}
int Polygon::direction(cv::Point a, cv::Point b, cv::Point c){
    int val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y);

    if(val == 0) return 0;
    else if(val < 0) return 2;
    return 1;
}
bool Polygon::isIntersect(Line l1, Line l2){
    int dir1 = direction(l1.p1, l1.p2, l2.p1);
    int dir2 = direction(l1.p1, l1.p2, l2.p2);
    int dir3 = direction(l2.p1, l2.p2, l1.p1);
    int dir4 = direction(l2.p1, l2.p2, l1.p2);

    if (dir1 != dir2 && dir3 != dir4) return true;

    if (dir1 == 0 && onLine(l1, l2.p1)) return true;

    if (dir2 == 0 && onLine(l1, l2.p2)) return true;

    if (dir3 == 0 && onLine(l2, l1.p1)) return true;

    if (dir4 == 0 && onLine(l2, l1.p2)) return true;

    return false;
}
bool Polygon::checkInside(std::vector<cv::Point> poly, cv::Point &point){
    if(poly.size() < 3){
        return false;
    }

    Line exline = {point, {9999, point.y}};

    int count = 0;

    int i = 0;

    do{
        Line side = {poly[i], poly[(i+1) % poly.size()]};
        if(Polygon::isIntersect(side, exline)){
            if(Polygon::direction(side.p1, point, side.p2) == 0){
                return Polygon::onLine(side, point);
            }
            count++;
        }
        i = (i + 1) % poly.size();
    }while(i != 0);

    return count & 1;
}
std::vector<Yolo::Detection> Polygon::checkAreaBoxes(std::vector<Yolo::Detection> &boxes){
    std::vector<Yolo::Detection> boxTmp{};
    for(auto &poly: polygons)
    {
        for(auto &box: boxes)
        {
            int cX = int(box.bbox[0] + box.bbox[2] / 2.0);
            int cY = int(box.bbox[1] + box.bbox[3] / 2.0);
            cv::Point point{cX, cY};

            if(Polygon::checkInside(poly, point)){
               boxTmp.push_back(box);
            }
        }
    }

    return boxTmp;
}

