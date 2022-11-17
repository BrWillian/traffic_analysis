cmake_minimum_required(VERSION 3.5)

project(traffic_analysis LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_definitions(-std=c++14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 14)

execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND crc32 ${CMAKE_CURRENT_SOURCE_DIR}/meta/version.h
        OUTPUT_VARIABLE W_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_definitions(-DW_HASH="${W_HASH}")
add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")
add_definitions(-DGIT_BRANCH="${GIT_BRANCH}")

message(STATUS "Generating version.h")

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/meta/version.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/version.h
)
message(STATUS "Generating weights.h")

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND tar xf weights.tar.gz
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i vehicle.engine
        OUTPUT_VARIABLE GEN_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/meta/weights.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/weights.h
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND rm vehicle.engine
)

find_package(CUDA 10.2 REQUIRED)

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_52;code=sm_52)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

if(NOT DEFINED OPENCV_LIBS)
    message(WARNING "-DOPENCV_LIBS não foi utilizada..")
    find_package(OpenCV REQUIRED)
endif()

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

message("Libraries: " ${OpenCV_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBRARIES})

set(OUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIRECTORY})

file(MAKE_DIRECTORY ${OUT_DIRECTORY})

set (LIB_DEPENDENCIES ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TENSORRT_LIBS})

#cuda_add_library(yololayer SHARED src/yololayer.cu)
#set_target_properties(yololayer PROPERTIES PREFIX "")

#add_library(tracker SHARED src/tracker.cpp)
#set_target_properties(tracker PROPERTIES PREFIX "")

#add_library(polygon SHARED src/polygon.cpp)
#set_target_properties(polygon PROPERTIES PREFIX "")

cuda_add_library(trafficanalysis SHARED
  meta/wrapper.hpp
  src/detect.cpp
  src/tracker.cpp
  src/yololayer.cu
  src/polygon.cpp
)

add_executable(debug
  examples/main_wrapper.cpp
#  debug/main.cpp
)

target_link_libraries(trafficanalysis ${LIB_DEPENDENCIES})
target_link_libraries(debug trafficanalysis)

add_definitions(-O2 -pthread)
