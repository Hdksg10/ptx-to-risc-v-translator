
#include "CUDAFunction.h"
#include "cuda.h"
#include <EmulatedCUDADevice.h>
namespace driver{

// Driver global variables initialization
EmulatedCUDADevice* devices[MAX_DEVICES] = {nullptr}; // Initialize all device pointers to nullptr
bool driverInitialized = false;
bool driverDeinitialized = false;

// Static member initialization
const std::string EmulatedCUDADevice::name = "EmulatedCUDADevice";

const CUuuid_st EmulatedCUDADevice::uuid = {static_cast<char>(0xd2), static_cast<char>(0xdf), static_cast<char>(0x96), static_cast<char>(0xa6), 0x39, 0x2f, 0x29, 0x5c, 0x03, 0x78, static_cast<char>(0xe8), 0x0f, 0x22, static_cast<char>(0xd3), static_cast<char>(0x99), 0x7b
}; // TODO: random generate uuid

EmulatedCUDADevice::EmulatedCUDADevice() {
    isInitialized = false;
}

void EmulatedCUDADevice::initialize(unsigned int Flags) {
    isInitialized = true;
}
void EmulatedCUDADevice::load(CUDAModule* module) {
    device.load(module->getModule());
}

void EmulatedCUDADevice::launchKernel(CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra ) {
    ir::Dim3 grid(gridDimX, gridDimY, gridDimZ);
    ir::Dim3 block(blockDimX, blockDimY, blockDimZ);
    auto kernel = static_cast<CUDAFunction*>(f->outerKernel);
    // parse parameters
    auto module = kernel->getModule()->getModule();
    auto ptxKernel = module->getKernel(kernel->getName());
    size_t size = 0;
    size_t index = 0;
    auto paramsBuffer = new char[PARAMS_BUFFER_SIZE]; // allocate buffer for parameters
    std::cout << "log1" << std::endl;
    for (auto argument = ptxKernel->arguments.begin(); argument != ptxKernel->arguments.end(); ++argument)
    {
        size_t argSize = argument->getAlignment();
        std::memcpy(paramsBuffer + size, kernelParams[index], argSize);
        size += argSize;
        index++;
    }
    device.launch(module->id(), kernel->getName(), grid, block, sharedMemBytes, paramsBuffer, size);
    std::cout << "log3" << std::endl;
    // delete[] paramsBuffer;
}

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

} // namespace driver