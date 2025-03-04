#ifndef INTERFACE_H
#define INTERFACE_H

#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDAContext.h>
#include <EmulatedCUDADevice.h>

// CUDA Driver API
CUresult CUDAAPI cuDeviceGetCount_cpp(int *count);
CUresult CUDAAPI cuInit_cpp(unsigned int Flags);
CUresult CUDAAPI cuDriverGetVersion_cpp(int *driverVersion);
CUresult CUDAAPI cuDeviceGet_cpp(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetName_cpp(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem_cpp(size_t *bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid_cpp(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute_cpp(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRetain_cpp(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI cuCtxCreate_cpp(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxSetCurrent_cpp(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent_cpp(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetDevice_cpp(CUdevice *device);
CUresult CUDAAPI cuCtxDestroy_cpp(CUcontext ctx);

// CUDA Runtime API
cudaError_t CUDARTAPI cudaDeviceCanAccessPeer_cpp(int *canAccessPeer, int device, int peerDevice);

#endif