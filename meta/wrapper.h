#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/core.h"
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
struct VehicleDetect {
    TrafficCore* trafficCore;
    std::vector<Vehicle::Detection>* vehicles;
    cv::Mat* image;
};
#else
struct VehicleDetect {
    void* TrafficCore;
    void* vehicles;
    void* image;
};
#endif

typedef struct VehicleDetect vehicle_t;

TRAFFICANALISYS_API vehicle_t* C_vehicleDetect();

TRAFFICANALISYS_API void C_vehicleDetectDestroy(vehicle_t* vh);

TRAFFICANALISYS_API const char* C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize);

TRAFFICANALISYS_API const char* C_getVersion();

TRAFFICANALISYS_API const char* C_getWVersion();

#ifdef __cplusplus
TRAFFICANALISYS_API std::string doInference(vehicle_t* vh, cv::Mat& img);
#endif

#endif // WRAPPER_H
