/*
*/
#ifndef EMULATED_CUDA_DEVICE_H
#define EMULATED_CUDA_DEVICE_H

#include <string>
#include <cuda.h>

#include <CUDAContext.h>

namespace driver {
    constexpr int MAX_DEVICES = 1; // Maximum number of devices to emulate
    constexpr int DRIVER_VERSION = 8000; // Version PTX 5.0. CUDA 8.0
    class EmulatedCUDADevice {
    public:
        EmulatedCUDADevice();
        ~EmulatedCUDADevice();
        void initialize(unsigned int Flags);
        void shutdown();
        void executeKernel();
        void allocateMemory();
        void freeMemory();
        void copyToDevice();
        void copyFromDevice();
        
        int getAttribute(CUdevice_attribute attrib);
    
    public:
        bool isInitialized = false;
        
        // cuda device attributes
    public: 
        const static std::string name;
        const static CUuuid_st uuid;
        constexpr static size_t totalMemBytes = 4294967296; // 4 * 1024 * 1024 * 1024 Bytes
    public:
        /* Primary context of device
         *
         * NOTE: Since the device is emulated by our translator, we ensure that only current process can access the emulated device, and we assume that we only have one context in program, so we don't need to manage multiple contexts, and we don't need to implement push and pop context functions.
         *
         * The primary context is created when the device is initialized and destroyed when the device is shutdown.
         */
        driver::CUDAContext* context; 
    };
    extern EmulatedCUDADevice* devices[MAX_DEVICES];
    extern bool driverInitialized;
    extern bool driverDeinitialized;
} // namespace server
#endif // EMULATED_CUDA_DEVICE_H