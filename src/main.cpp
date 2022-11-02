#include <iostream>
#include <chrono>
#include "../include/detect.h"

int main(int argc, char *argv[])
{
    auto *vh = new Vehicle::Detect();

    vh->createContextExecution();

    std::string images_inf[] = {"/root/imagem/img_100.jpg","/root/imagem/img_101.jpg","/root/imagem/img_102.jpg","/root/imagem/img_103.jpg","/root/imagem/img_104.jpg","/root/imagem/img_105.jpg","/root/imagem/img_106.jpg","/root/imagem/img_107.jpg","/root/imagem/img_108.jpg","/root/imagem/img_109.jpg"};
    int num_images = sizeof(images_inf)/sizeof(images_inf[0]);

    for (int b = 0; b < num_images; b++) {
        cv::Mat img = cv::imread(images_inf[b]);
        auto start = std::chrono::high_resolution_clock::now();

        auto res = vh->doInference(img);        

        auto end = std::chrono::high_resolution_clock::now();


//        for(auto &it: res){
//            std::cout<<it.class_id<<std::endl;
//        }

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Inferência: " << b + 1 << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;
    vh->~Detect();

}
