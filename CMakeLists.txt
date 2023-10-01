cmake_minimum_required(VERSION 3.5)

project(traffic_analysis LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_definitions(-std=c++17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INSTALL_RPATH "/usr/local/tensorrt/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
message(STATUS "Generating weights.cpp")

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND unzip weights.zip
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i vehicle.engine
        OUTPUT_VARIABLE VEHICLE_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i plate.engine
        OUTPUT_VARIABLE PLATE_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i ocr.engine
        OUTPUT_VARIABLE OCR_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i color.engine
        OUTPUT_VARIABLE COLOR_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND xxd -i helmet.engine
        OUTPUT_VARIABLE HELMET_WEIGHTS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/meta/weights.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/weights.cpp
)

message(STATUS "Generating weights.cpp")

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated/
        COMMAND bash -c "sed -n -e 's/\\(unsigned char [^=]*\\).*/extern \\1;/p' -e 's/\\(unsigned int [^=]*\\).*/extern \\1;/p' weights.cpp | sed 's/ ;/;/g'"
        OUTPUT_VARIABLE WEIGHTS_HEADER
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
execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND rm plate.engine
)

execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND rm ocr.engine
)
execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND rm color.engine
)
execute_process(
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/meta/
        COMMAND rm helmet.engine
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

if(NOT DEFINED YAMLCPP_LIBS)
    message(WARNING "-YAMLCPP_LIBS não foi utilizada..")
    find_library(YAMLCPP yaml-cpp)
endif()

if(NVINFER AND NVINFERPLUGIN AND NVPARSERS AND NVONNXPARSER)
    list(APPEND TENSORRT_LIBS ${NVINFER} ${NVINFERPLUGIN} ${NVPARSERS} ${NVONNXPARSER})
endif()

find_package(Threads REQUIRED)

message("Libraries: "${OpenCV_LIBS} ${TENSORRT_LIBS} ${YAMLCPP} ${CMAKE_THREAD_LIBS_INIT})

set(OUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUT_DIRECTORY})

file(MAKE_DIRECTORY ${OUT_DIRECTORY})

set (LIB_DEPENDENCIES ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TENSORRT_LIBS} ${YAMLCPP} ${CMAKE_THREAD_LIBS_INIT})

cuda_add_library(trafficanalysis SHARED
        src/wrapper.cpp
        src/detect.cpp
        src/tracker.cpp
        src/yololayer.cu
        generated/weights.cpp
        src/core.cpp
        src/ocr.cpp
        src/color.cpp
        src/plate.cpp
        src/vehicle.cpp
        src/helmet.cpp
)

add_executable(debug
        examples/cpp/example.cpp
        # examples/cpp/debug.cpp
        # examples/cpp/debugCore.cpp
        # examples/cpp/saveImage.cpp
        )

target_link_libraries(trafficanalysis ${LIB_DEPENDENCIES})
target_link_libraries(debug trafficanalysis)

add_definitions(-O2 -pthread)
