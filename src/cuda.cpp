/* Implemetion of CUDA Driver API
 * TODO: Line 
*/
#include <Interface.h>
#include <cuda.h>


extern "C" CUresult CUDAAPI cuDeviceGetCount(int *count) {
    return cuDeviceGetCount_cpp(count);
}

extern "C" CUresult CUDAAPI cuInit(unsigned int Flags) {
    return cuInit_cpp(Flags);
}

extern "C" CUresult CUDAAPI cuDriverGetVersion(int *driverVersion) {
    return cuDriverGetVersion_cpp(driverVersion);
}

extern "C" CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    return cuDeviceGet_cpp(device, ordinal);
}

extern "C" CUresult CUDAAPI cuDeviceGetName(char *name, int len, CUdevice dev) {
    return cuDeviceGetName_cpp(name, len, dev);
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem_cpp(bytes, dev);
}

extern "C" CUresult CUDAAPI cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    return cuDeviceGetUuid_cpp(uuid, dev);
}

extern "C" CUresult CUDAAPI cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    return cuDeviceGetAttribute_cpp(pi, attrib, dev);
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    return cuDevicePrimaryCtxRetain_cpp(pctx, dev);
}

extern "C" CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate_cpp(pctx, flags, dev);
}

extern "C" CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx) {
    return cuCtxSetCurrent_cpp(ctx);
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext *pctx) {
    return cuCtxGetCurrent_cpp(pctx);
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice *device) {
    return cuCtxGetDevice_cpp(device);
}

extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext ctx) {
    return cuCtxDestroy_cpp(ctx);
}