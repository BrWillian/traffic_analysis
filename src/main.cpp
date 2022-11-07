#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include "../include/tracker.h"
#include "../include/detect.h"
#include "../include/polygon.h"

int main(int argc, char *argv[])
{
    auto *vh = new Vehicle::Detect();
    auto *tracker = new Vehicle::Tracker(0);

    vh->createContextExecution();

//    std::string images_inf[] = {"/root/imagem/img_100.jpg","/root/imagem/img_101.jpg","/root/imagem/img_102.jpg","/root/imagem/img_103.jpg","/root/imagem/img_104.jpg","/root/imagem/img_105.jpg","/root/imagem/img_106.jpg","/root/imagem/img_107.jpg","/root/imagem/img_108.jpg","/root/imagem/img_109.jpg"};
//    int num_images = sizeof(images_inf)/sizeof(images_inf[0]);

//    for (int b = 0; b < num_images; b++) {
//        cv::Mat img = cv::imread(images_inf[b]);
//        auto start = std::chrono::high_resolution_clock::now();

//        auto res = vh->doInference(img);

//        auto end = std::chrono::high_resolution_clock::now();


////        for(auto &it: res){
////            std::cout<<it.class_id<<std::endl;
////        }

//        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//        std::cout << "Inferência: " << b + 1 << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//    }





    cv::VideoCapture cap(argv[1]);

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;

    std::string classes[] = {"carro", "moto", "onibus" "van", "caminhao", "caminhonete"};

    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 7, cv::Size(1280,960));

    std::vector<std::vector<cv::Point>> polygons{
        {cv::Point(67, 250), cv::Point(492, 242), cv::Point(602, 708), cv::Point(5, 717)},
        {cv::Point(541, 238), cv::Point(983, 236), cv::Point(1187, 662), cv::Point(668, 708)}
                                                };

    auto *checkArea = new Vehicle::Polygon(polygons);

    while(1){

        cap >> frame;

        if(frame.empty()){
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        auto res = vh->doInference(frame);

        cv::Mat layer = cv::Mat::zeros(frame.size(), CV_8UC3);

        cv::fillPoly(layer, polygons, cv::Scalar(0, 0, 255));
        cv::addWeighted(frame, 0.5, layer, 0.3, 0, frame);

        std::vector<std::vector<int>> boxes;

        for(auto &it: res){
            cv::Rect r = vh->getRect(frame, it.bbox);
            cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 1);
            cv::putText(frame, classes[(int)it.class_id], cv::Point(r.x, r.y), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
            boxes.insert(boxes.end(), {r.x, r.y, r.width, r.height});
        }

        auto filtredObjects = checkArea->checkAreaBoxes(boxes);

        auto objects = tracker->update(filtredObjects);

        for(auto &it: objects){
                cv::circle(frame, cv::Point(it.second.first, it.second.second), 5, cv::Scalar(0, 0, 255), -1);
                std::string ID = std::to_string(it.first);
                cv::putText(frame, ID, cv::Point(it.second.first - 10, it.second.second - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
        }


        auto end = std::chrono::high_resolution_clock::now();

        cv::imshow("name", frame);

        char c=(char)cv::waitKey(30);
            if(c==27)
              break;

        video.write(frame);


        //auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        //std::cout << "Inferência: " << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    cap.release();
    //video.release();

    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;
    vh->~Detect();

}
