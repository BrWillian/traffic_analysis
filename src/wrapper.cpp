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

    objwrapper = (__typeof__(objwrapper)) malloc(sizeof(*objwrapper));

    objwrapper->trafficCore = TrafficAnalysis;
    objwrapper->vehicles = detections;
    objwrapper->image = imageContainer;

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
std::string Serialize(const std::vector<Vehicle::Detection>& res) {

    std::stringstream ss;
    ss << "{\"detections\": [";

    for (size_t i = 0; i < res.size(); ++i) {
        const Vehicle::Detection& vh = res[i];
        ss << "{";
        ss << "\"id\":" << vh.id << ",";
        ss << "\"ocr\":\"" << vh.ocr << "\",";
        ss << "\"placa\":" << std::boolalpha << vh.plate << ",";
        ss << "\"faixa\":" << vh.faixa + 1 << ",";
        ss << "\"cor\":\"" << vh.color << "\",";
        ss << "\"classe\":\"" << vh.class_name << "\",";
        ss << "\"x\":" << vh.bbox[0] << ",";
        ss << "\"y\":" << vh.bbox[1] << ",";
        ss << "\"w\":" << vh.bbox[2] << ",";
        ss << "\"h\":" << vh.bbox[3] << ",";
        ss << "\"centroid\":{\"x\":" << vh.centroid.x << ",";
        ss << "\"y\":" << vh.centroid.y << "},";
        ss << "\"placa_bbox\":{\"x\":" << vh.plate_bbox[0] << ",";
        ss << "\"y\":" << vh.plate_bbox[1] << ",";
        ss << "\"w\":" << vh.plate_bbox[2] << ",";
        ss << "\"h\":" << vh.plate_bbox[3] << "}";
        ss << "}";
        if (i != res.size() - 1) {
            ss << ",";
        }
    }

    ss << "]}";
    return ss.str();
}
const char* CDECL C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize){
    if (vh == nullptr) {
        std::cerr << "[ERROR] Received invalid pointer" << std::endl;
    }

    std::vector<uchar> data(imgData, imgData + imgSize);
    *vh->image = cv::imdecode(cv::Mat(data), -1);

    if (vh->image->empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize({}).c_str());
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

        async_colors.get();
        async_ocr.get();

        vh->trafficCore->setIdVehicles(*vh->vehicles);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize({}).c_str());
    }

    return strdup(Serialize(*vh->vehicles).c_str());
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

    if (img.empty()) {
        std::cerr << "[ERROR] Failed to decode image" << std::endl;
        return strdup(Serialize({}).c_str());
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

        async_colors.get();
        async_ocr.get();

        vh->trafficCore->setIdVehicles(*vh->vehicles);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return strdup(Serialize({}).c_str());
    }

    return strdup(Serialize(*vh->vehicles).c_str());
}