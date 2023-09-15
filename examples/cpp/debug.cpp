#include <iostream>
#include <chrono>
#include <future>
#include <opencv2/opencv.hpp>
#define USE_WRAPPER 0

#if USE_WRAPPER == 0
    #include "../../include/tracker.h"
    #include "../../include/trigger.h"
    #include "../../include/detect.h"
    #include "../../include/yololayer.h"
#else
    #include "../meta/wrapper.h"
#endif

inline const char * const BoolToString(bool b)
{
    return b ? " true" : " false";
}


int main(int argc, char *argv[])
{
#if USE_WRAPPER == 0
    auto *vh = new Detect(MODEL_TYPE_VEHICLE);
    auto *pd = new Detect(MODEL_TYPE_PLATE);
    auto *od = new Detect(MODEL_TYPE_OCR);
    auto *tracker = new Tracker();
    auto *dc = new Detect(MODEL_TYPE_COLOR);

    Trigger::Margin = 30;
    Trigger::Lines = {
            std::make_pair(cv::Point(40, 275), cv::Point(530, 275)),
            std::make_pair(cv::Point(590, 265), cv::Point(1100, 265)),
    };
#else
    vehicle_t *vh = C_vehicleDetect();
#endif


    cv::VideoCapture cap(argv[1]);

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    cv::Mat frame;

    const std::string classes[] = {"carro", "moto", "onibus", "caminhao", "van", "caminhonete"};


    std::vector<std::pair<cv::Point, cv::Point>> lines{
        std::make_pair(cv::Point(40, 275), cv::Point(530, 265)),
        std::make_pair(cv::Point(590, 265), cv::Point(1100, 275)),
    };

#if USE_WRAPPER == 0
    //cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(960,720));
    while(true) {

        cap >> frame;
        cv::Mat copy_frame = frame.clone();

        if (frame.empty()) {
            break;
        }

        try {

            auto start = std::chrono::high_resolution_clock::now();

            std::vector<Yolo::Detection> vehicles = Trigger::getVehicles(*vh, copy_frame);

            std::vector<bool> trigged = Trigger::checkLinePassage(vehicles);

            Trigger::filterObjects(vehicles, trigged);

            std::future<std::vector<std::string>> colors_future = std::async(std::launch::async, [&]() {
                return Trigger::getColors(*dc, vehicles, copy_frame);
            });

            std::future<std::vector<std::string>> ocr_future = std::async(std::launch::async, [&]() {
                return Trigger::getplateOcr(*pd, *od, vehicles, copy_frame);
            });


            auto colors = colors_future.get();
            auto results = ocr_future.get();

            tracker->update(vehicles);

            auto end = std::chrono::high_resolution_clock::now();


            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "Inferência: " << " FPS: " << 1000.0 / (static_cast<double>(time.count())) << std::endl;

            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            cv::line(frame, cv::Point(40, 350), cv::Point(530, 340), cv::Scalar(0, 255, 0));
            cv::line(frame, cv::Point(590, 340), cv::Point(1080, 350), cv::Scalar(0, 255, 0));


            for (size_t i = 0; i < vehicles.size(); i++) {
                cv::putText(frame, classes[(int) vehicles[i].class_id],
                            cv::Point(vehicles[i].bbox[0], vehicles[i].bbox[1]), cv::FONT_HERSHEY_DUPLEX, 1,
                            cv::Scalar(0, 255, 0));
                cv::rectangle(frame, cv::Point(vehicles[i].bbox[0], vehicles[i].bbox[1]),
                              cv::Point(vehicles[i].bbox[2] + vehicles[i].bbox[0],
                                        vehicles[i].bbox[3] + vehicles[i].bbox[1]), cv::Scalar(0, 255, 0), 1);
                cv::circle(frame, cv::Point((vehicles[i].bbox[0] + vehicles[i].bbox[2] + vehicles[i].bbox[0]) / 2,
                                            (vehicles[i].bbox[1] + vehicles[i].bbox[3] + vehicles[i].bbox[1]) / 2), 5,
                           cv::Scalar(0, 0, 255), -1);
                cv::putText(frame, std::to_string(vehicles[i].id),
                            cv::Point((vehicles[i].bbox[0] + vehicles[i].bbox[2] + vehicles[i].bbox[0]) / 2,
                                      (vehicles[i].bbox[1] + vehicles[i].bbox[3] + vehicles[i].bbox[1]) / 2),
                            cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0));
                std::cout << "ID: " + std::to_string(vehicles[i].id) + " Color: " + colors[i] << std::endl;

                cv::Rect roi(vehicles[i].bbox[0], vehicles[i].bbox[1], vehicles[i].bbox[2], vehicles[i].bbox[3]);
                cv::Mat image_roi = copy_frame(roi);
                auto plates = pd->doInference(image_roi);
                for (auto &plate: plates) {
                    cv::rectangle(image_roi, cv::Point(plate.bbox[0], plate.bbox[1]),
                                  cv::Point(plate.bbox[2] + plate.bbox[0], plate.bbox[3] + plate.bbox[1]),
                                  cv::Scalar(0, 0, 255), 1);
                }


                cv::imshow(results[i] +" Color: " + colors[i] +  " ID: " + std::to_string(vehicles[i].id), image_roi);
                cv::waitKey(0);
            }
            char c = (char) cv::waitKey(90);
            if (c == 27)
                break;

        } catch (std::exception &e){
            std::cout<<e.what()<<std::endl;
        }

        cv::resize(frame, frame, cv::Size(960,720));
        cv::imshow("name", frame);
        //video.write(frame);

    }

    cap.release();
    //video.release();

    vh->~Detect();
#else
    while(1) {

        cap >> frame;

        if (frame.empty()) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        std::string resultJson = doInference(vh, frame);

        std::cout<<resultJson<<std::endl;

        auto end = std::chrono::high_resolution_clock::now();

        cv::Mat layer = cv::Mat::zeros(frame.size(), CV_8UC3);

        cv::fillPoly(layer, polygons, cv::Scalar(0, 0, 255));
        cv::addWeighted(frame, 1.0, layer, 0.3, 0, frame);

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::imshow("teste", frame);
        char c=(char)cv::waitKey(30);
        if(c==27)
          break;
    }
    cap.release();
    C_vehicleDetectDestroy(vh);
#endif

}