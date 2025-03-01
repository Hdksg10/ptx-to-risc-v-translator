/* Implemetion of CUDA Driver API
 * TODO: Line 
*/
#include "CUDAContext.h"
#include <cuda.h>

#include <string>
#include <cstring>
#include <EmulatedCUDADevice.h>

extern "C"
{
CUresult CUDAAPI cuDeviceGetCount(int *count) {
    // Implementation to get the number of CUDA devices
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer
    // For the translator, we always return 1 device (the emulated one)
    *count = driver::MAX_DEVICES; 
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit(unsigned int Flags) {
    // Implementation to initialize the CUDA driver
    // This is a placeholder implementation
    for (int i = 0; i < driver::MAX_DEVICES; i++) {
        driver::devices[i] = new driver::EmulatedCUDADevice();
        driver::devices[i]->initialize(Flags);
    }
    driver::driverInitialized = true; // Mark the driver as initialized
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    // Implementation to get the version of the CUDA driver
    if (driverVersion == NULL) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer (invalid argument)
    *driverVersion = driver::DRIVER_VERSION; // Version PTX 5.0. CUDA 8.0
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    // Implementation to get a handle to a device
    // This is a placeholder implementation
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ordinal < 0 || ordinal >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device ordinal
    if (device == NULL) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer (invalid argument)
    // In a real implementation, we would check if the device is available and return a handle to it
    // Here, we simply return the ordinal as the device handle (for simplicity and testing purposes
    *device = (CUdevice)ordinal; 
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    // Implementation to get the name of a device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (name == NULL) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer (invalid argument)
    if (len <= 0) return CUDA_ERROR_INVALID_VALUE; // Check for invalid length
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    // In a real implementation, we would retrieve the actual device name
    // Here, we use a predefined name for simplicity and testing purposes
    // Assuming EmulatedCUDADevice::name is a static member holding the device name (for example, "Emulated CUDA Device")
    std::strncpy(name, driver::EmulatedCUDADevice::name.c_str(), len - 1);
    name[len - 1] = '\0'; // Ensure null termination
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    // Implementation to get the total memory of the device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    *bytes = driver::devices[dev]->totalMemBytes;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    // Implementation to get the UUID of the device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (uuid == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    memcpy(uuid, &driver::devices[dev]->uuid, 16);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    // Implementation to get atributes of the device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (pi == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    
    // This is a placeholder implementation
    // TODO : Implement the actual attribute fetching logic
    *pi = driver::devices[dev]->getAttribute(attrib);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer

    // Check if the primary context is already retained
    if (driver::devices[dev]->context != nullptr) {
        *pctx = driver::devices[dev]->context->getContext();
        return CUDA_SUCCESS;
    }
    // Create a new primary context for the device
    driver::devices[dev]->context = new driver::CUDAContext();
    driver::devices[dev]->context->create(dev, 0); // Assuming default flags for primary context
    *pctx = driver::devices[dev]->context->getContext();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    driver::devices[dev]->context = new driver::CUDAContext();
    driver::devices[dev]->context->create(dev, flags);
    *pctx = driver::devices[dev]->context->getContext();
    return CUDA_SUCCESS;
}

// Note that we havn't implemented the context stack management yet, so cuCtxSetCurrent will not work as expected.
CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ctx == nullptr) return CUDA_ERROR_INVALID_CONTEXT; // Check for valid context handle
    auto context = driver::contextStack.top();
    context->setCtx(ctx);
    return CUDA_SUCCESS;
}
// Note that we havn't implemented the context stack management yet, so cuCtxGetCurrent will not work as expected.
CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    if (driver::contextStack.empty()) {
        *pctx = nullptr;
        return CUDA_SUCCESS;
    }
    auto context = driver::contextStack.top();
    *pctx = context->getContext();
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    *device = context->getContext()->device;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid context
    ctx->destroyed = 1;
    ctx->valid = 0;
    return CUDA_SUCCESS;
}
}