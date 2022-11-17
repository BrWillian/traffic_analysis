#include <string>
#include "../meta/wrapper.h"
#include "../include/detect.h"
#include "../include/tracker.h"
#include "../include/polygon.h"
#include "../generated/version.h"
#include <opencv2/opencv.hpp>


vehicle_t* CDECL C_vehicleDetect(){
    vehicle_t* objwrapper;

    Vehicle::Detect *vh = new Vehicle::Detect();
    Vehicle::Tracker *tracker = new Vehicle::Tracker(0);

    std::vector<std::vector<cv::Point>> polygons{{cv::Point(60, 340), cv::Point(1115, 335), cv::Point(1270, 610), cv::Point(0, 640)}};

    Vehicle::Polygon *checkArea = new Vehicle::Polygon(polygons);

    objwrapper = (typeof(objwrapper)) malloc(sizeof(*objwrapper));

    objwrapper->detectObj = vh;
    objwrapper->trackerObj = tracker;
    objwrapper->areaCheck = checkArea;

    return objwrapper;
}

void CDECL C_vehicleDetectDestroy(vehicle_t* vh){
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    delete static_cast<Vehicle::Detect*>(vh->detectObj);
    delete static_cast<Vehicle::Tracker*>(vh->trackerObj);
    delete static_cast<Vehicle::Polygon*>(vh->areaCheck);
    free(vh);
}

std::string Serialize(std::vector<Detection> &res){
    std::stringstream ss;
    ss << "{\"detections\": [";
    for(auto vh = res.begin(); vh!=res.end();)
    {
        ss << "{\"id\":"<<vh->obj_id;
        ss << ",\"classe\":\""<<vh->class_name;
        ss << "\",\"centroid\":["<<vh->centroid.x;
        ss << ","<<vh->centroid.y<<"]";
        ss << ",\"x\":"<<vh->bbox[0];
        ss << ",\"y\":"<<vh->bbox[1];
        ss << ",\"w\":"<<vh->bbox[2];
        ss << ",\"h\":"<<vh->bbox[3];
        if(++vh == res.end()){
           ss<<"}";
        }else {
           ss<<"},";
        }
    }
    ss << "]";
    ss << "}";

    return ss.str();
}
const char* CDECL C_doInference(vehicle_t* vh, const char* imgData, size_t imgSize){
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    Vehicle::Detect *det;
    Vehicle::Tracker *tracker;
    Vehicle::Polygon *checkArea;

    det = static_cast<Vehicle::Detect*>(vh->detectObj);
    tracker = static_cast<Vehicle::Tracker*>(vh->trackerObj);
    checkArea = static_cast<Vehicle::Polygon*>(vh->areaCheck);

    std::vector<uchar> data(imgData, imgData + imgSize);
    cv::Mat img = cv::imdecode(cv::Mat(data), -1);

    std::vector<Yolo::Detection> detects = det->doInference(img);

    checkArea->checkAreaBoxes(detects);

    std::vector<std::pair<int, std::pair<int, int>>> objects = tracker->update(detects);

    std::vector<Detection> res;

    for(size_t i = 0; i<objects.size(); i++)
    {
        Detection det_t;
        det_t.bbox[0] = detects[i].bbox[0];
        det_t.bbox[1] = detects[i].bbox[1];
        det_t.bbox[2] = detects[i].bbox[2]+detects[i].bbox[0];
        det_t.bbox[3] = detects[i].bbox[3]+detects[i].bbox[1];
        det_t.class_name = classes[(int)detects[i].class_id];
        det_t.conf = detects[i].conf;
        det_t.obj_id = objects[i].first;
        det_t.centroid.x = objects[i].second.first;
        det_t.centroid.y = objects[i].second.second;
        res.push_back(det_t);
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
    if(vh == nullptr){
        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
    }
    Vehicle::Detect *det;
    Vehicle::Tracker *tracker;
    Vehicle::Polygon *checkArea;

    det = static_cast<Vehicle::Detect*>(vh->detectObj);
    tracker = static_cast<Vehicle::Tracker*>(vh->trackerObj);
    checkArea = static_cast<Vehicle::Polygon*>(vh->areaCheck);

    std::vector<Yolo::Detection> detects = det->doInference(img);

    checkArea->checkAreaBoxes(detects);

    std::vector<std::pair<int, std::pair<int, int>>> objects = tracker->update(detects);

    std::vector<Detection> res;

    for(size_t i = 0; i<objects.size(); i++)
    {
        Detection det_t;
        det_t.bbox[0] = detects[i].bbox[0];
        det_t.bbox[1] = detects[i].bbox[1];
        det_t.bbox[2] = detects[i].bbox[2]+detects[i].bbox[0];
        det_t.bbox[3] = detects[i].bbox[3]+detects[i].bbox[1];
        det_t.class_name = classes[(int)detects[i].class_id];
        det_t.conf = detects[i].conf;
        det_t.obj_id = objects[i].first;
        det_t.centroid.x = objects[i].second.first;
        det_t.centroid.y = objects[i].second.second;
        res.push_back(det_t);
    }

    return Serialize(res);
}
