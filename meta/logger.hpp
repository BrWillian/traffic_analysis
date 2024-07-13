#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger{
public:
    void log(Severity severity, const char* msg) noexcept override {
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR || (severity == Severity::kINFO))) {
            std::cerr << msg << std::endl;
        }
    }
};

#endif // LOGGER_HPP
