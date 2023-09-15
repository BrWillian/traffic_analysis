//
// Created by willian on 17/07/23.
//
#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "../../include/detect.h"
#include "../../include/trigger.h"

std::vector<std::string> list_files(const char* path) {
    std::vector<std::string> result;

    DIR* p_dir = opendir(path);
    if (p_dir != nullptr) {

        struct dirent* p_file = nullptr;
        while ((p_file = readdir(p_dir)) != nullptr) {
            if (strcmp(p_file->d_name, ".") != 0 && strcmp(p_file->d_name, "..") != 0) {
                std::string cur_file_name(p_file->d_name);
                result.push_back(cur_file_name);
            }
        }

        closedir(p_dir);
    }
    return result;
}

int main(int argc, char *argv[]) {
    auto *vh = new Detect(MODEL_TYPE_VEHICLE);
    auto *plate = new Detect(MODEL_TYPE_PLATE);
    auto *ocr = new Detect(MODEL_TYPE_OCR);
    auto *color = new Detect(MODEL_TYPE_COLOR);

    int inference_mean = 0;

    std::vector<std::string> files = list_files(argv[1]);

    std::string classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};


    for (auto &file: files) {
        std::string root(argv[1]);
        std::string abs_path = root + file;
        std::cout << abs_path << std::endl;
        //auto start = std::chrono::high_resolution_clock::now();
        try {
            cv::Mat img = cv::imread(abs_path, cv::IMREAD_COLOR);

            auto vehicles = Trigger::getVehicles(*vh, img);
            std::vector<bool> trigged(1, true);
            //auto plates = Trigger::getplateOcr(*plate, *ocr, trigged, vehicles, img);

            for (auto &vehicle: vehicles) {
                if (!(vehicle.class_id == 1 || vehicle.class_id == 2 || vehicle.class_id == 3)) {

                    cv::Rect r(vehicle.bbox[0], vehicle.bbox[1], vehicle.bbox[2], vehicle.bbox[3]);
                    int x = std::max(r.x, 0);
                    int y = std::max(r.y, 0);
                    int width = std::min(r.x + r.width, img.cols) - x;
                    int height = std::min(r.y + r.height, img.rows) - y;
                    cv::Rect adjustedRect(x, y, width, height);
                    cv::Mat image_roi = img(adjustedRect);

                    auto start = std::chrono::high_resolution_clock::now();

                    std::vector<std::string> vehicle_color = Trigger::getColors(*color, vehicles, image_roi);
                    std::cout<<vehicle_color[0]<<std::endl;

                    auto end = std::chrono::high_resolution_clock::now();

                    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

                    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms Color" << std::endl;



//                    if (!plates.empty() && plates.size() == 1) {
//                        auto now = std::chrono::system_clock::now();
//                        auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
//                                now.time_since_epoch()).count();
//                        cv::imwrite(argv[2] +
//                        classes[(int) vehicle.class_id] + "_" + plates[0] + "_" + std::to_string(timestamp) +
//                        ".jpg", image_roi);
//                    }
                }
            }
//
//            auto end = std::chrono::high_resolution_clock::now();
//
//            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//
//            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        } catch (const std::exception &e) {
            std::cout<<e.what()<<std::endl;
        }
    }
}