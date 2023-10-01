//
// Created by willian on 23/08/23.
//

#ifndef TRAFFIC_ANALYSIS_TYPES_H
#define TRAFFIC_ANALYSIS_TYPES_H

#include <string>

namespace Vehicle
{
    struct Point{
        int x, y;
    };

    struct Detection {
        int bbox[4];
        std::string class_name;
        uint16_t id;
        int faixa;
        Point centroid;
        bool plate;
        int plate_bbox[4];
        std::string ocr;
        std::string color;
        bool without_helmet;
        int persons_bike;
    };
}

#endif //TRAFFIC_ANALYSIS_TYPES_H
