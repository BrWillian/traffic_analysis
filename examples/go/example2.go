package main

/*
#cgo LDFLAGS: -L. -ltrafficanalysis
#include "../../meta/wrapper.h"
*/
import "C"
import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"time"
	"unsafe"
)

type VehicleDetect struct {
	ptr *C.vehicle_t
}

func getVersion() string {
	version := C.GoString(C.C_getVersion())

	return version
}

func getWVersion() string {
	w_version := C.GoString(C.C_getWVersion())

	return w_version
}
func VehicleDetectConstruct() VehicleDetect {
	var vh VehicleDetect
	vh.ptr = C.C_vehicleDetect()

	return vh
}
func VehicleDetectDestroy(vh VehicleDetect) {
	C.C_vehicleDetectDestroy(vh.ptr)
}
func VehicleDetectInference(vh VehicleDetect, img image.Image) string {
	buf := new(bytes.Buffer)
	_ = jpeg.Encode(buf, img, nil)

	b := buf.Bytes()

	result := C.C_doInference(vh.ptr, (*C.uchar)(unsafe.Pointer(&b[0])), C.int(buf.Len()))

	return C.GoString(result)
}
func getImageFromFilePath(filePath string) (image.Image, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	image, _, err := image.Decode(f)
	return image, err
}

func main() {
	pointer := VehicleDetectConstruct()
	fmt.Println(pointer)

	files, err := ioutil.ReadDir("/root/imagem/")
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {

		img, _ := getImageFromFilePath("/root/imagem/" + file.Name())

		start := time.Now()

		result := VehicleDetectInference(pointer, img)

		fmt.Println(result)

		duration := time.Since(start)

		fmt.Printf("Time: %d ms\n", duration.Milliseconds())
	}

	fmt.Println("Destruct Pointer")

	VehicleDetectDestroy(pointer)

	fmt.Println("Destroyed Pointer")

	fmt.Println(getVersion())
	fmt.Println(getWVersion())
}

