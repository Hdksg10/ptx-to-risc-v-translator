
#include <EmulatedCUDADevice.h>
using namespace driver;

// Driver global variables initialization
EmulatedCUDADevice* devices[MAX_DEVICES] = {nullptr}; // Initialize all device pointers to nullptr
bool driverInitialized = false;
bool driverDeinitialized = false;

// Static member initialization
const std::string EmulatedCUDADevice::name = "EmulatedCUDADevice";

const static CUuuid_st uuid = {static_cast<char>(0xd2), static_cast<char>(0xdf), static_cast<char>(0x96), static_cast<char>(0xa6), 0x39, 0x2f, 0x29, 0x5c, 0x03, 0x78, static_cast<char>(0xe8), 0x0f, 0x22, static_cast<char>(0xd3), static_cast<char>(0x99), 0x7b
}; // TODO: random generate uuid

int EmulatedCUDADevice::getAttribute(CUdevice_attribute attrib) {
    int pi = 0;
    // TODO: finish all attributes and use configuarble attributes
    switch (attrib) {
        case CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
        {
            pi = 82;
            break;
        }
        case CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR:
        {
            pi = 8;
            break;
        }
        case CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR:
        {
            pi = 6;
            break;
        }
        default:
        {
            pi = 0;
            break;
        }
    }

    return pi;
}

