#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
#include <string>
#include "../include/core.h"
#include <opencv2/opencv.hpp>
#endif

#if defined(__GNUC__)
//  GCC
#define TRAFFICANALISYS_API __attribute__((visibility("default")))
#define IMPORT
#define CDECL __attribute__((cdecl))
#else
//  do nothing and hope for the best?
#define EXPORT
#define IMPORT
#define CDECL
#pragma warning Unknown dynamic link import/export semantics.
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct VehicleDetect {
    TrafficCore* trafficCore;
    std::vector<Vehicle::Detection>* vehicles;
    cv::Mat* image;
    std::stringstream* ss;
};
#else
struct VehicleDetect {
    void* trafficCore;
    void* vehicles;
    void* image;
    void* ss;
};
#endif

typedef struct VehicleDetect vehicle_t;

TRAFFICANALISYS_API vehicle_t* C_vehicleDetect();

TRAFFICANALISYS_API void C_vehicleDetectDestroy(vehicle_t* vh);

TRAFFICANALISYS_API const char* C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize);

TRAFFICANALISYS_API void C_loadConfig(vehicle_t* vh);

TRAFFICANALISYS_API const char* C_getVersion();

TRAFFICANALISYS_API const char* C_getWVersion();

#ifdef __cplusplus
TRAFFICANALISYS_API std::string doInference(vehicle_t* vh, cv::Mat& img);
#endif

#ifdef __cplusplus
}
#endif

#endif // WRAPPER_H
