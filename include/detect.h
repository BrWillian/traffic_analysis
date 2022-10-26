#ifndef DETECT_H
#define DETECT_H

#include <iostream>

#if defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#define CDECL __attribute__((__cdecl))
#else
//  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #define CDECL
    #pragma warning Unknown dynamic link import/export semantics.
#endif


class Detect
{
private:

    uint8_t batchSize{};
    uint8_t numClasses{};
    uint32_t outputSize{};
    uint16_t inputH{};
    uint16_t inputW{};
    const char* inputBlobName{};
    const char* outputBlobName{};


    typedef struct {
        template<class T>
        void operator()(T* obj) const{
            delete obj;
        }
    }TRTDelete;

    template<class T>
    using TRTptr = std::unique_ptr<T, TRTDelete>;


public:
    Detect();
};

#endif // DETECT_H
