#include <iostream>
#include <chrono>
#include <future>
#include <opencv2/opencv.hpp>
#define USE_WRAPPER 0

#if USE_WRAPPER == 0
#include "../../include/core.h"
#endif

int main(int argc, char *argv[])
{
    auto trafficCore = TrafficCore();
    trafficCore.parseConfig();
//    trafficCore.setLines({std::make_pair(cv::Point(40, 450), cv::Point(530, 440)),std::make_pair(cv::Point(590, 440), cv::Point(1100, 450))});
//    trafficCore.setMargin(30);

    std::vector<Vehicle::Detection> vehicles{};

    cv::VideoCapture cap(argv[1]);

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int delay = static_cast<int>(1000 / fps);

    cv::Mat frame;
//
//
//    std::vector<std::pair<cv::Point, cv::Point>> lines{
//            std::make_pair(cv::Point(40, 275), cv::Point(530, 265)),
//            std::make_pair(cv::Point(590, 265), cv::Point(1100, 275)),
//    };

    //cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(960,720));
    while(true) {

        cap >> frame;
        cv::Mat copy_frame = frame.clone();

        if (frame.empty()) {
            break;
        }

        try {

            auto start = std::chrono::high_resolution_clock::now();

            trafficCore.getVehicles(copy_frame, vehicles);

            trafficCore.checkLinePassage(vehicles);

            std::future<void> async_colors = std::async(std::launch::async, [&]() {
                return trafficCore.getColors(vehicles, copy_frame);
            });

            std::future<void> async_ocr = std::async(std::launch::async, [&]() {
                return trafficCore.getplateOcr(vehicles, copy_frame);
            });

            std::future<void> async_brand = std::async(std::launch::async, [&]() {
                return trafficCore.getBrands(vehicles, copy_frame);
            });

            async_colors.get();
            async_ocr.get();
            async_brand.get();

            trafficCore.setIdVehicles(vehicles);

            trafficCore.getTriggeds(vehicles);

            auto end = std::chrono::high_resolution_clock::now();

            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            /*std::cout << "Inferência: " << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
*/
            cv::line(frame, cv::Point(40, 275), cv::Point(530, 265), cv::Scalar(0, 255, 0));
            cv::line(frame, cv::Point(590, 265), cv::Point(1080, 275), cv::Scalar(0, 255, 0));


            for(auto& it: trafficCore.vehiclesTrigger) {
                if(it.second.entry_time != std::chrono::steady_clock::time_point() && it.second.exit_time != std::chrono::steady_clock::time_point() && !it.second.is_in_trigger){
                    cv::Rect roi(it.second.bbox[0], it.second.bbox[1], it.second.bbox[2], it.second.bbox[3]);
                    cv::Mat image_roi = copy_frame(roi);
                    std::cout << "---------------" << std::endl;

                    std::cout << "PASSED: " << it.second.passedLine << std::endl;
                    std::cout << "ID: " << std::to_string(it.second.id) << std::endl;
                    std::cout << "CLASSE: " << it.second.class_name << std::endl;
                    std::cout << "OCR: " << it.second.ocr << std::endl;
                    std::cout << "COR: " << it.second.color << std::endl;
                    std::cout << "FAIXA: " << std::to_string(it.second.faixa) << std::endl;
                    std::cout << "PLACA: " << it.second.plate << std::endl;
                    std::cout << "PLACA_BBOX: " << "x: " << it.second.plate_bbox[0] << "y: " << it.second.plate_bbox[1] << "w: " << it.second.plate_bbox[2] << "h: " << it.second.plate_bbox[3] << std::endl;
                    std::cout << "MARCA_MODELO: " << it.second.brand_model << std::endl;

                    std::cout << "ENTRY TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            it.second.entry_time.time_since_epoch()).count() << " ms\n";
                    std::cout << "EXIT TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            it.second.exit_time.time_since_epoch()).count() << " ms\n";
                    std::cout << "DURATION: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            it.second.exit_time - it.second.entry_time).count() << " ms\n";
                    std::cout << "---------------" << std::endl;


                    trafficCore.vehiclesTrigger[it.first] = Vehicle::Detection{};
                    //cv::imshow(it.second.ocr, image_roi);
                    //cv::waitKey(0);
                }

            }


/*
            for (auto & vehicle : vehicles) {
                if(vehicle.passedLine) {
                    cv::Rect roi(vehicle.bbox[0], vehicle.bbox[1], vehicle.bbox[2], vehicle.bbox[3]);
                    cv::Mat image_roi = copy_frame(roi);
                    std::cout << "---------------" << std::endl;

                    std::cout << "PASSED: " << vehicle.passedLine << std::endl;
                    std::cout << "ID: " << std::to_string(vehicle.id) << std::endl;
                    std::cout << "CLASSE: " << vehicle.class_name << std::endl;
                    std::cout << "OCR: " << vehicle.ocr << std::endl;
                    std::cout << "COR: " << vehicle.color << std::endl;
                    std::cout << "FAIXA: " << std::to_string(vehicle.faixa) << std::endl;
                    std::cout << "PLACA: " << vehicle.plate << std::endl;
                    std::cout << "PLACA_BBOX: " << "x: " << vehicle.plate_bbox[0] << "y: " << vehicle.plate_bbox[1] << "w: " << vehicle.plate_bbox[2] << "h: " << vehicle.plate_bbox[3] << std::endl;
                    std::cout << "MARCA_MODELO: " << vehicle.brand_model << std::endl;

                    std::cout << "ENTRY TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            vehicle.entry_time.time_since_epoch()).count() << " ms\n";
                    std::cout << "EXIT TIME: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            vehicle.exit_time.time_since_epoch()).count() << " ms\n";
                    std::cout << "DURATION: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                            vehicle.exit_time - vehicle.entry_time).count() << " ms\n";
//                cv::Rect roi_plate(vehicle.plate_bbox[0],vehicle.plate_bbox[1], vehicle.plate_bbox[2], vehicle.plate_bbox[3]);
//                cv::Mat plate = copy_frame(roi_plate);
                    std::cout << "---------------" << std::endl;
//                cv::circle(frame, cv::Point(vehicle.centroid.x, vehicle.centroid.y),20,
//                           cv::Scalar(255, 0, 0), -1);
                    cv::imshow("Teste", image_roi);
//                cv::imshow("plate", plate);
                    cv::waitKey(0);
                }
            }*/
            if (cv::waitKey(delay) >= 0) {
                break;
            }

        } catch (std::exception &e){
            std::cout<<e.what()<<std::endl;
        }
        std::cout<<trafficCore.vehiclesTrigger.size()<<std::endl;
        cv::resize(frame, frame, cv::Size(960,720));
        cv::imshow("name", frame);
        //video.write(frame);


    }

    //cap.release();
    //video.release();
}