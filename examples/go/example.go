package main

/*
#cgo LDFLAGS: -ltrafficanalysis
#include "../../meta/wrapper.h"
*/
import "C"
import (
    "fmt"
)

func main() {
    pointer := C.C_vehicleDetect()
    fmt.Println(pointer)
    C.C_vehicleDetectDestroy(pointer)
    version := C.GoString(C.C_getVersion())
    w_version := C.GoString(C.C_getWVersion())


    fmt.Println(version)
    fmt.Println(w_version)
}
