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

extern "C" CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    return cuModuleLoad_cpp(module, fname);
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule *module, const void *image) {
    return cuModuleLoadData_cpp(module, image);
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    return cuModuleGetFunction_cpp(hfunc, hmod, name);
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    return cuModuleUnload_cpp(hmod);
}

extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr* dptr, size_t bytesize) {
    return cuMemAlloc_cpp(dptr, bytesize);
}

extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    return cuMemFree_cpp(dptr);
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    return cuMemcpyHtoD_cpp(dstDevice, srcHost, ByteCount);
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoH_cpp(dstHost, srcDevice, ByteCount);
}

extern "C" CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    return cuMemsetD32_cpp(dstDevice, ui, N);
}

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    return cuLaunchKernel_cpp(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}