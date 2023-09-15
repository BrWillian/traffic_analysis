#include <iostream>
#include "../meta/wrapper.h"
#include <chrono>
#include <vector>


int main(int argc, char *argv[]){
    vehicle_t *vh = C_vehicleDetect();


    cv::VideoCapture cap(argv[1]);

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat image;

    while(true) {

        cap >> image;

        if (image.empty()) {
            break;
        }

        cv::imshow("frame", image);
        cv::waitKey(1);

        auto start = std::chrono::high_resolution_clock::now();

        std::string resultJson = doInference(vh, image);

        std::cout<<resultJson<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Inferência: " << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    std::cout<<C_getVersion()<<std::endl;
    std::cout<<C_getWVersion()<<std::endl;
    C_vehicleDetectDestroy(vh);
    return 0;
}
