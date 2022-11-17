#include <iostream>
#include "../meta/wrapper.hpp"
#include <chrono>


int main(){
    vehicle_t *vh = C_vehicleDetect();

    std::string images_inf[] = {"/root/imagem/img_231.jpg","/root/imagem/img_36.jpg","/root/imagem/img_507.jpg","/root/imagem/img_93.jpg",
                                "/root/imagem/img_232.jpg","/root/imagem/img_370.jpg","/root/imagem/img_508.jpg","/root/imagem/img_94.jpg",
                                "/root/imagem/img_233.jpg","/root/imagem/img_371.jpg","/root/imagem/img_509.jpg","/root/imagem/img_95.jpg",
                                "/root/imagem/img_234.jpg","/root/imagem/img_372.jpg","/root/imagem/img_50.jpg","/root/imagem/img_96.jpg",
                                "/root/imagem/img_235.jpg","/root/imagem/img_373.jpg","/root/imagem/img_510.jpg","/root/imagem/img_97.jpg",
                                "/root/imagem/img_236.jpg","/root/imagem/img_374.jpg","/root/imagem/img_511.jpg","/root/imagem/img_98.jpg",
                                "/root/imagem/img_237.jpg","/root/imagem/img_375.jpg","/root/imagem/img_512.jpg","/root/imagem/img_99.jpg",
                                "/root/imagem/img_238.jpg","/root/imagem/img_376.jpg","/root/imagem/img_513.jpg","/root/imagem/img_9.jpg"};

    std::vector<std::vector<cv::Point>> polygons{{cv::Point(60, 340), cv::Point(1115, 335), cv::Point(1270, 610), cv::Point(0, 640)}};

    int num_images = sizeof(images_inf)/sizeof(images_inf[0]);
    for (int b = 0; b < num_images; b++) {
        cv::Mat img = cv::imread(images_inf[b]);


        cv::Mat layer = cv::Mat::zeros(img.size(), CV_8UC3);

        cv::fillPoly(layer, polygons, cv::Scalar(0, 0, 255));
        cv::addWeighted(img, 1.0, layer, 0.3, 0, img);

        auto start = std::chrono::high_resolution_clock::now();

        std::string resultJson = doInference(vh, img);

        std::cout<<resultJson<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Inferência: " << b + 1 << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::imshow("janela", img);
        cv::waitKey(0);
    }

    std::cout<<C_getVersion()<<std::endl;
    std::cout<<C_getWVersion()<<std::endl;
    C_vehicleDetectDestroy(vh);
    return 0;
}
