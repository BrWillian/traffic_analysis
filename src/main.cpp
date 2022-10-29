#include <iostream>
#include "../include/detect.h"

int main(int argc, char *argv[])
{
    Vehicle::Detect *vh = new Vehicle::Detect();

    vh->createContextExecution();

    std::cout<<vh->getVersion()<<std::endl;
    std::cout<<vh->getWVersion()<<std::endl;

}
