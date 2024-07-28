//
// Created by willian on 23/08/23.
//

#ifndef TRAFFIC_ANALYSIS_TYPES_H
#define TRAFFIC_ANALYSIS_TYPES_H

#include <string>
#include <chrono>

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
        int brand_model_id;
        std::string brand_model;
        bool passedLine = false;
        // Parada e velocidade
        std::chrono::time_point<std::chrono::steady_clock> entry_time; // Horário de entrada no laço
        std::chrono::time_point<std::chrono::steady_clock> exit_time;   // Horário de saída do laço
        bool entry_time_set = false; // Flag para verificar se entry_time foi definido
        bool exit_time_set = false;  // Flag para verificar se exit_time foi definido
        bool is_in_trigger = false;     // Indica se está dentro do trigger
        bool sent = false;
    };
}

#endif //TRAFFIC_ANALYSIS_TYPES_H
