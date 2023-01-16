#include <string>
#include "../meta/wrapper.h"
#include "../include/detect.h"
#include "../include/tracker.h"
#include "../include/polygon.h"
#include "../generated/version.h"
#include <libconfig.h++>
#include <opencv2/opencv.hpp>


vehicle_t* CDECL C_vehicleDetect(){
    vehicle_t* objwrapper;

    Vehicle::Detect *vh = new Vehicle::Detect();
    Vehicle::Tracker *tracker = new Vehicle::Tracker(0);

    std::vector<std::vector<cv::Point>> polygons = getPolygons();//{{cv::Point(60, 340), cv::Point(1115, 335), cv::Point(1270, 610), cv::Point(0, 640)}};

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
//        ss << "\",\"centroid\":["<<vh->centroid.x;
//        ss << ","<<vh->centroid.y<<"]";
        ss << "\",\"x\":"<<vh->bbox[0];
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
const char* CDECL C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize){
//    if(vh == nullptr){
//        std::cerr<<"[ERROR] Received invalid pointer"<<std::endl;
//    }
    Vehicle::Detect *det;
    Vehicle::Tracker *tracker;
    Vehicle::Polygon *checkArea;

    det = static_cast<Vehicle::Detect*>(vh->detectObj);
    tracker = static_cast<Vehicle::Tracker*>(vh->trackerObj);
    checkArea = static_cast<Vehicle::Polygon*>(vh->areaCheck);


//    int h = 960;
//    int w = 1280;
//    int slice_size = 1280*960; // w*h

//    cv::Mat R(h,w,CV_8U, imgData);
//    cv::Mat G(h,w,CV_8U, imgData+slice_size);
//    cv::Mat B(h,w,CV_8U, imgData+(2*slice_size));

//    cv::Mat chan[] {B,G,R}; // BGR order for opencv !
//    cv::Mat final;
//    cv::merge(chan, 3, final);

//    unsigned char *img_data_ptr = (unsigned char*) &imgData;

//    std::vector<uchar> data(imgData, imgData + imgSize);
    //memcpy(vh->imgBuffer.data, imgData, 1280*960*3*sizeof(uchar));


//    cv::imshow("",final);
//    cv::waitKey(0);


    //cv::imwrite("img.jpg", img);

    std::vector<uchar> data(imgData, imgData + imgSize);
//    memcpy(vh->imgBuffer.data, imgData, 1280*960*3*sizeof(uchar));
    cv::Mat img = cv::imdecode(cv::Mat(data), -1);
//    cv::imwrite("img.jpg", img);
//    vh->imgBuffer.reshape(3, 1280);
//    cv::imwrite("img.jpg", vh->imgBuffer);


    auto detects = det->doInference(img);

    checkArea->checkAreaBoxes(detects);

    auto objects = tracker->update(detects);

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
//        det_t.centroid.x = objects[i].second.first;
//        det_t.centroid.y = objects[i].second.second;
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
//        det_t.centroid.x = objects[i].second.first;
//        det_t.centroid.y = objects[i].second.second;
        res.push_back(det_t);
    }

    return Serialize(res);
}
std::vector<std::vector<cv::Point>> getPolygons() {
    libconfig::Config cfg;
    const char* cfg_file = "config.cfg";
    try {
        cfg.readFile(cfg_file);
    }catch(const libconfig::FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        exit(EXIT_FAILURE);
    }
    catch(const libconfig::ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                  << " - " << pex.getError() << std::endl;
        exit(EXIT_FAILURE);
    }
    const libconfig::Setting& root = cfg.getRoot();
    const libconfig::Setting &polys = root["polygons"];
    std::vector<std::vector<int>> global_cords;
    for (int i = 0; i < polys.getLength(); ++i) {
        const libconfig::Setting &poly = polys[i];
        std::vector<int> cords;
        for(int j=0; j<poly.getLength(); ++j){
            const libconfig::Setting &point = poly[i];

            for(int k=0; k<point.getLength(); k++)
            {
                const libconfig::Setting &cord = poly[j];
                cords.push_back(cord[k]);
            }
        }
        global_cords.push_back(cords);
    }
    std::vector<std::vector<cv::Point>> polygons;

    for(auto & global_cord : global_cords)
    {
        std::vector<cv::Point> points;
        for(int j=0; j<global_cord.size()-1; j++){
            int k = j++;
            points.emplace_back(global_cord[k], global_cord[j]);
        }
        polygons.push_back(points);
    }
    int conta = 0;
    for(auto& it: polygons){
        std::cout<<"Added polygon: "<<++conta<<std::endl;
        for(auto &poly: it){
            std::cout<<"X: "<<poly.x<<" Y: "<<poly.y<<std::endl;
        }
    }

    return polygons;
}