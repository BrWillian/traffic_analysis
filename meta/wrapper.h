#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#include "../include/core.h"
#endif

#if defined(__GNUC__)
//  GCC
#define VEHICLEDETECT_API __attribute__((visibility("default")))
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
    struct VehicleDetect{
        TrafficCore *TrafficCore;
        std::vector<Vehicle::Detection> *vehicles;
        cv::Mat *image;
    };

    typedef struct VehicleDetect vehicle_t;

    VEHICLEDETECT_API vehicle_t* C_vehicleDetect();

    VEHICLEDETECT_API void C_vehicleDetectDestroy(vehicle_t* vh);

    VEHICLEDETECT_API const char* C_doInference(vehicle_t* vh, unsigned char* imgData, int imgSize);

    VEHICLEDETECT_API const char* C_getVersion();

    VEHICLEDETECT_API const char* C_getWVersion();

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
    VEHICLEDETECT_API std::string doInference(vehicle_t* vh, cv::Mat& img);
#endif

#endif // WRAPPER_H
