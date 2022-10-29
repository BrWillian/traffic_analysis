#include <iostream>
#include <chrono>
#include "../include/detect.h"

int main(int argc, char *argv[])
{
    Vehicle::Detect *vh = new Vehicle::Detect();

    vh->createContextExecution();

    std::string images_inf[] = {};

    int num_images = sizeof(images_inf)/sizeof(images_inf[0]);

    for (int b = 0; b < num_images; b++) {
        cv::Mat img = cv::imread(images_inf[b], cv::IMREAD_UNCHANGED);
        auto start = std::chrono::high_resolution_clock::now();

    }



    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;

}
