#include <string>
#include <future>
#include "../meta/wrapper.h"
#include "../include/detect.h"
#include "../include/tracker.h"
#include "../generated/version.h"
#include "../include/trigger.h"
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>


std::pair<float, std::vector<std::pair<cv::Point, cv::Point>>> CDECL parseConfig(){
    std::ifstream file("config.yaml");
    if (!file) {
        std::cerr << "Erro ao abrir o arquivo de configuração." << std::endl;
        exit(1);
    }
    YAML::Node root = YAML::Load(file);

    int margem;
    std::vector<std::pair<cv::Point, cv::Point>> lines;

    margem = root["margem"].as<int>();

    YAML::Node faixas = root["faixas"];
    for (const auto& faixa : faixas) {
        std::string nome = faixa["nome"].as<std::string>();

        YAML::Node pt1 = faixa["pt1"];
        YAML::Node pt2 = faixa["pt2"];

        cv::Point pt1_xy(pt1[0].as<int>(), pt1[1].as<int>());
        cv::Point pt2_xy(pt2[0].as<int>(), pt2[1].as<int>());
        lines.emplace_back(pt1_xy, pt2_xy);

        std::cout << "Faixa: " << nome << std::endl;
        std::cout << "Ponto 1: (" << pt1[0].as<int>() << ", " << pt1[1].as<int>() << ")" << std::endl;
        std::cout << "Ponto 2: (" << pt2[0].as<int>() << ", " << pt2[1].as<int>() << ")" << std::endl;
        std::cout << std::endl;
    }

    return std::make_pair(margem, lines);
};


vehicle_t* CDECL C_vehicleDetect(){
    vehicle_t* objwrapper;

    Detect *vehicle_detect = new Detect(MODEL_TYPE_VEHICLE);
    Detect *plate_detect = new Detect(MODEL_TYPE_PLATE);
    Detect *ocr_detect = new Detect(MODEL_TYPE_OCR);
    Detect *color_infer = new Detect(MODEL_TYPE_COLOR);
    Tracker *tracker = new Tracker(0);

    std::pair<float, std::vector<std::pair<cv::Point, cv::Point>>> config = parseConfig();

    Trigger::Lines = config.second;
    Trigger::Margin = config.first;

    objwrapper = (__typeof__(objwrapper)) malloc(sizeof(*objwrapper));

    objwrapper->vehicle_detect = vehicle_detect;
    objwrapper->plate_detect = plate_detect;
    objwrapper->ocr_detect = ocr_detect;
    objwrapper->color_infer = color_infer;
    objwrapper->obj_tracker = tracker;

    return objwrapper;
}

void CDECL C_vehicleDetectDestroy(vehicle_t* vh){
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    delete static_cast<Detect*>(vh->vehicle_detect);
    delete static_cast<Detect*>(vh->plate_detect);
    delete static_cast<Detect*>(vh->ocr_detect);
    delete static_cast<Tracker*>(vh->obj_tracker);
    free(vh);
}

std::string Serialize(const std::vector<Detection>& res) {
    std::stringstream ss;
    ss << "{\"detections\": [";
    for (auto vh = res.begin(); vh != res.end();) {
        ss << "{\"id\":" << vh->obj_id;
        ss << ",\"placa\":\"" << vh->plate;
        ss << "\",\"cor\":\"" << vh->color;
        ss << "\",\"classe\":\"" << vh->class_name;
        ss << "\",\"centroid\":{\"x\":" << vh->centroid.x;
        ss << ",\"y\":" << vh->centroid.y << "}";
        ss << ",\"x\":" << vh->bbox[0];
        ss << ",\"y\":" << vh->bbox[1];
        ss << ",\"w\":" << vh->bbox[2];
        ss << ",\"h\":" << vh->bbox[3];
        if (++vh == res.end()) {
            ss << "}";
        } else {
            ss << "},";
        }
    }
    ss << "]";
    ss << "}";

    return ss.str();
}
const char* CDECL C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize){
    if (vh == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    Detect* vehicle_detect = static_cast<Detect*>(vh->vehicle_detect);
    Detect* plate_detect = static_cast<Detect*>(vh->plate_detect);
    Detect* ocr_detect = static_cast<Detect*>(vh->ocr_detect);
    Detect* color_infer = static_cast<Detect*>(vh->color_infer);
    Tracker* tracker = static_cast<Tracker*>(vh->obj_tracker);

    std::vector<uchar> data(imgData, imgData + imgSize);
    cv::Mat img = cv::imdecode(cv::Mat(data), -1);

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize({}).c_str());
    }

    std::vector<Detection> res;

    try {
        auto vehicles = Trigger::getVehicles(*vehicle_detect, img);
        auto trigged = Trigger::checkLinePassage(vehicles);
        Trigger::filterObjects(vehicles, trigged);

        std::future<std::vector<std::string>> colors_future = std::async(std::launch::async, [&]() {
            return Trigger::getColors(*color_infer, vehicles, img);
        });

        std::future<std::vector<std::string>> ocr_future = std::async(std::launch::async, [&]() {
            return Trigger::getplateOcr(*plate_detect, *ocr_detect, vehicles, img);
        });

        auto colors = colors_future.get();
        auto plates = ocr_future.get();

        auto objects = tracker->update(vehicles);

        for (size_t i = 0; i < vehicles.size(); i++) {
            Detection det_t;
            det_t.bbox[0] = vehicles[i].bbox[0];
            det_t.bbox[1] = vehicles[i].bbox[1];
            det_t.bbox[2] = vehicles[i].bbox[2] + vehicles[i].bbox[0];
            det_t.bbox[3] = vehicles[i].bbox[3] + vehicles[i].bbox[1];
            det_t.centroid.x = (vehicles[i].bbox[0] + vehicles[i].bbox[2] + vehicles[i].bbox[0]) / 2;
            det_t.centroid.y = (vehicles[i].bbox[1] + vehicles[i].bbox[3] + vehicles[i].bbox[1]) / 2;
            det_t.class_name = Trigger::vehicle_classes[(int)vehicles[i].class_id].c_str();
            det_t.conf = vehicles[i].conf;
            det_t.obj_id = objects[i].first;
            det_t.plate = plates[i].c_str();
            det_t.color = colors[i].c_str();
            res.push_back(det_t);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize({}).c_str());
    }

    return strdup(Serialize(res).c_str());
}
const char* CDECL C_getVersion(){
    return VERSION "-" GIT_BRANCH "-" GIT_COMMIT_HASH;
}
const char* CDECL C_getWVersion(){
    return W_VERSION "-" W_HASH;
}

std::string CDECL doInference(vehicle_t* vh, cv::Mat& img){
    if (vh == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    Detect* vehicle_detect = static_cast<Detect*>(vh->vehicle_detect);
    Detect* plate_detect = static_cast<Detect*>(vh->plate_detect);
    Detect* ocr_detect = static_cast<Detect*>(vh->ocr_detect);
    Detect* color_infer = static_cast<Detect*>(vh->color_infer);
    Tracker* tracker = static_cast<Tracker*>(vh->obj_tracker);

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize({}).c_str());
    }

    std::vector<Detection> res;

    try {
        auto vehicles = Trigger::getVehicles(*vehicle_detect, img);
        auto trigged = Trigger::checkLinePassage(vehicles);
        Trigger::filterObjects(vehicles, trigged);

        std::future<std::vector<std::string>> colors_future = std::async(std::launch::async, [&]() {
            return Trigger::getColors(*color_infer, vehicles, img);
        });

        std::future<std::vector<std::string>> ocr_future = std::async(std::launch::async, [&]() {
            return Trigger::getplateOcr(*plate_detect, *ocr_detect, vehicles, img);
        });

        auto colors = colors_future.get();
        auto plates = ocr_future.get();

        auto objects = tracker->update(vehicles);

        for (size_t i = 0; i < vehicles.size(); i++) {
            Detection det_t;
            det_t.bbox[0] = vehicles[i].bbox[0];
            det_t.bbox[1] = vehicles[i].bbox[1];
            det_t.bbox[2] = vehicles[i].bbox[2] + vehicles[i].bbox[0];
            det_t.bbox[3] = vehicles[i].bbox[3] + vehicles[i].bbox[1];
            det_t.centroid.x = (vehicles[i].bbox[0] + vehicles[i].bbox[2] + vehicles[i].bbox[0]) / 2;
            det_t.centroid.y = (vehicles[i].bbox[1] + vehicles[i].bbox[3] + vehicles[i].bbox[1]) / 2;
            det_t.class_name = classes[(int)vehicles[i].class_id];
            det_t.conf = vehicles[i].conf;
            det_t.obj_id = objects[i].first;
            det_t.plate = plates[i].c_str();
            det_t.color = colors[i].c_str();
            res.push_back(det_t);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize({}).c_str());
    }

    return strdup(Serialize(res).c_str());
}