#include <iostream>
#include <chrono>
#include "../include/detect.h"

int main(int argc, char *argv[])
{
    auto *vh = new Vehicle::Detect();

    vh->createContextExecution();

    std::string images_inf[] = {"/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_10.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_100.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1003.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1004.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1005.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1006.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1015.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1019.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1027.jpg","/home/willian/projects/vizentec/traffic_analysis_fixed/imgs/img_rv_1030.jpg"};

    int num_images = sizeof(images_inf)/sizeof(images_inf[0]);

    for (int b = 0; b < num_images; b++) {
        cv::Mat img = cv::imread(images_inf[b]);
        auto start = std::chrono::high_resolution_clock::now();

        auto res = vh->doInference(img);

        for(auto &it: res){
            std::cout<<it.class_id<<std::endl;
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Inferência: " << b + 1 << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;

}
