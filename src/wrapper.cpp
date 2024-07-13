#include "../meta/wrapper.h"
#include <future>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "../generated/version.h"

vehicle_t* CDECL C_vehicleDetect(){
    vehicle_t* objwrapper;
    auto *TrafficAnalysis = new TrafficCore();
    TrafficAnalysis->parseConfig();

    auto *detections = new std::vector<Vehicle::Detection>;

    auto *imageContainer = new cv::Mat();

    auto *stringResult = new std::stringstream;

    objwrapper = (__typeof__(objwrapper)) malloc(sizeof(*objwrapper));

    objwrapper->trafficCore = TrafficAnalysis;
    objwrapper->vehicles = detections;
    objwrapper->image = imageContainer;
    objwrapper->ss = stringResult;

    return objwrapper;
}

void CDECL C_vehicleDetectDestroy(vehicle_t* vh){
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    delete static_cast<TrafficCore*>(vh->trafficCore);
    delete static_cast<std::vector<Vehicle::Detection>*>(vh->vehicles);
    delete static_cast<cv::Mat*>(vh->image);
    free(vh);
}
std::string Serialize(vehicle_t* vh, const std::vector<Vehicle::Detection>& res) {
    vh->ss->str("");
    *vh->ss << "{\"detections\": [";

    for (size_t i = 0; i < res.size(); ++i) {
        const Vehicle::Detection& detection = res[i];
        *vh->ss << "{";
        *vh->ss << "\"id\":" << detection.id << ",";
        *vh->ss << "\"ocr\":\"" << detection.ocr << "\",";
        *vh->ss << "\"placa\":" << std::boolalpha << detection.plate << ",";
        *vh->ss << "\"faixa\":" << detection.faixa + 1 << ",";
        *vh->ss << "\"cor\":\"" << detection.color << "\",";
        *vh->ss << "\"classe\":\"" << detection.class_name << "\",";
        *vh->ss << "\"veiculo_bbox\":{\"x\":" << detection.bbox[0] << ",";
        *vh->ss << "\"y\":" << detection.bbox[1] << ",";
        *vh->ss << "\"w\":" << detection.bbox[2] << ",";
        *vh->ss << "\"h\":" << detection.bbox[3] << "},";
        *vh->ss << "\"placa_bbox\":{\"x\":" << detection.plate_bbox[0] << ",";
        *vh->ss << "\"y\":" << detection.plate_bbox[1] << ",";
        *vh->ss << "\"w\":" << detection.plate_bbox[2] << ",";
        *vh->ss << "\"h\":" << detection.plate_bbox[3] << "},";
        *vh->ss << "\"marca_modelo_id\":" << detection.brand_model_id << ",";
        *vh->ss << "\"marca_modelo\":\"" << detection.brand_model << "\"";
        *vh->ss << "}";
        if (i != res.size() - 1) {
            *vh->ss << ",";
        }
    }

    *vh->ss << "]}";
    return vh->ss->str();
}
const char* CDECL C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize){
    if (vh == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    std::vector<uchar> data(imgData, imgData + imgSize);
    *vh->image = cv::imdecode(cv::Mat(data), -1);

    if (vh->image->empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    try {
        vh->trafficCore->getVehicles(*vh->image, *vh->vehicles);

        vh->trafficCore->checkLinePassage(*vh->vehicles);

        std::future<void> async_colors = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getColors(*vh->vehicles, *vh->image);
        });

        std::future<void> async_ocr = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getplateOcr(*vh->vehicles, *vh->image);
        });

        std::future<void> async_brand = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getBrands(*vh->vehicles, *vh->image);
        });

        async_colors.get();
        async_ocr.get();
        async_brand.get();

        vh->trafficCore->setIdVehicles(*vh->vehicles);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    return strdup(Serialize(vh, *vh->vehicles).c_str());
}
const char* CDECL C_getVersion(){
    return VERSION "-" GIT_BRANCH "-" GIT_COMMIT_HASH;
}
const char* CDECL C_getWVersion(){
    return W_VERSION "-" W_HASH;
}
void CDECL C_loadConfig(vehicle_t* vh){
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    vh->trafficCore->parseConfig();
}
std::string CDECL doInference(vehicle_t* vh, cv::Mat& img){
    if (vh == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    try {
        vh->trafficCore->getVehicles(img, *vh->vehicles);

        vh->trafficCore->checkLinePassage(*vh->vehicles);

        std::future<void> async_colors = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getColors(*vh->vehicles, img);
        });

        std::future<void> async_ocr = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getplateOcr(*vh->vehicles, img);
        });

        std::future<void> async_brand = std::async(std::launch::async, [&]() {
            return vh->trafficCore->getBrands(*vh->vehicles, *vh->image);
        });

        async_colors.get();
        async_ocr.get();
        async_brand.get();

        vh->trafficCore->setIdVehicles(*vh->vehicles);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize(nullptr, {}).c_str());
    }

    return strdup(Serialize(vh, *vh->vehicles).c_str());
}