#ifndef DETECT_H
#define DETECT_H

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


    typedef struct {
        template<class T>
        void operator()(T* obj) const{
            delete obj;
        }
    }TRTDelete;


public:
    Detect();
};

#endif // DETECT_H
