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

    //vh->createContextExecution();

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

    const std::string classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};

    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(960,720));

//    std::vector<std::vector<cv::Point>> polygons{
//        {cv::Point(43, 403), cv::Point(529, 393), cv::Point(583, 624), cv::Point(2, 639)},
//        {cv::Point(594, 510), cv::Point(655, 744), cv::Point(1275, 717), cv::Point(1187, 492)}
//                                                };

    std::vector<std::vector<cv::Point>> polygons{{cv::Point(60, 340), cv::Point(1115, 335), cv::Point(1270, 610), cv::Point(0, 640)}};

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
        cv::addWeighted(frame, 1.0, layer, 0.3, 0, frame);


//        for(auto &it: res){
//            cv::rectangle(frame, cv::Point(it.bbox[0], it.bbox[1]), cv::Point(it.bbox[2]+it.bbox[0], it.bbox[3]+it.bbox[1]), cv::Scalar(0, 255, 0), 1);
//            cv::putText(frame, classes[(int)it.class_id], cv::Point(it.bbox[0], it.bbox[1]), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
//        }

        checkArea->checkAreaBoxes(res);

        auto objects = tracker->update(res);

        for(size_t i = 0; i<objects.size(); i++)
        {
            cv::putText(frame, classes[(int)res[i].class_id], cv::Point(res[i].bbox[0], res[i].bbox[1]), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
            cv::rectangle(frame, cv::Point(res[i].bbox[0], res[i].bbox[1]), cv::Point(res[i].bbox[2]+res[i].bbox[0], res[i].bbox[3]+res[i].bbox[1]), cv::Scalar(0, 255, 0), 1);
            cv::circle(frame, cv::Point(objects[i].second.first, objects[i].second.second), 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame, std::to_string(objects[i].first), cv::Point(objects[i].second.first - 10, objects[i].second.second - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
        }

//        for(auto &it: objects){
//            cv::circle(frame, cv::Point(it.second.first, it.second.second), 5, cv::Scalar(0, 0, 255), -1);
//            std::string ID = std::to_string(it.first);
//            cv::putText(frame, ID, cv::Point(it.second.first - 10, it.second.second - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
//        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));




        cv::imshow("name", frame);

        char c=(char)cv::waitKey(1);
            if(c==27)
              break;

        //cv::resize(frame, frame, cv::Size(960,720));
        //video.write(frame);
        auto end = std::chrono::high_resolution_clock::now();


        //auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        //std::cout << "Inferência: " << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    }
    cap.release();
    //video.release();

    vh->~Detect();

}
