#include <iostream>
#include <include/detect.h>

int main(int argc, char *argv[])
{
    Vehicle::Detect *vh = new Vehicle::Detect();
    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;

}
