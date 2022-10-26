cmake_minimum_required(VERSION 3.5)

project(traffic_analysis LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED TENSORRT_LIBS)
    message(WARNING "-DTENSORRT_LIBS não foi utilizada..")
    find_library(NVINFER NAMES nvinfer)
    find_library(NVINFERPLUGIN NAMES nvinfer_plugin)
    find_library(NVPARSERS NAMES nvparsers)
    find_library(NVONNXPARSER NAMES nvonnxparser)
endif()

if(NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER)
    list(APPEND TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)
endif()

add_executable(traffic_analysis
  src/main.cpp
  src/detect.cpp
  src/logger.hpp
)

target_link_libraries(traffic_analysis)
