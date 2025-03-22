#include <cuda_runtime.h>

#include <Interface.h>

// internel functions
// internel initialization
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant,
    int global);
extern "C" void **__cudaRegisterFatBinary(void *fatCubin);  
extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid,
        dim3 *bDim, dim3 *gDim, int *wSize);
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle);
// internel kernel launch configuration
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    struct CUstream_st *stream = 0);

extern "C" unsigned __cudaPopCallConfiguration(void* param1, void* param2, void* param3, void* param4);

extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    return cudaDeviceCanAccessPeer_cpp(canAccessPeer, device, peerDevice);
} 

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    return cudaSetDevice_cpp(device);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
    return cudaDeviceSynchronize_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    return cudaChooseDevice_cpp(device, prop);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    return cudaDeviceGetAttribute_cpp(value, attr, device);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId)
{
    return cudaDeviceGetByPCIBusId_cpp(device, pciBusId);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    return cudaDeviceGetCacheConfig_cpp(pCacheConfig);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit)
{
    return cudaDeviceGetLimit_cpp(pValue, limit);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    return cudaDeviceGetP2PAttribute_cpp(value, attr, srcDevice, dstDevice);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    return cudaDeviceGetPCIBusId_cpp(pciBusId, len, device);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
    return cudaDeviceGetSharedMemConfig_cpp(pConfig);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
    return cudaDeviceGetStreamPriorityRange_cpp(leastPriority, greatestPriority);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device)
{
    return cudaDeviceGetTexture1DLinearMaxWidth_cpp(maxWidthInElements, fmtDesc, device);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceReset(void)
{
    return cudaDeviceReset_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return cudaDeviceSetCacheConfig_cpp(cacheConfig);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    return cudaDeviceSetLimit_cpp(limit, value);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    return cudaDeviceSetSharedMemConfig_cpp(config);
}

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    return cudaGetDevice_cpp(device);
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    return cudaGetDeviceCount_cpp(count);
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags)
{
    return cudaGetDeviceFlags_cpp(flags);
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    return cudaGetDeviceProperties_cpp(prop, device);
}

extern "C" cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags)
{
    return cudaSetDeviceFlags_cpp(flags);
}

extern "C" cudaError_t CUDARTAPI cudaSetValidDevices(int *deviceArr, int len)
{
    return cudaSetValidDevices_cpp(deviceArr, len);
}

extern "C" const char* CUDARTAPI cudaGetErrorName(cudaError_t error)
{
    return cudaGetErrorName_cpp(error);
}

extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
    return cudaGetErrorString_cpp(error);
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError(void)
{
    return cudaGetLastError_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaPeekAtLastError(void)
{
    return cudaPeekAtLastError_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    return cudaStreamCreate_cpp(pStream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    return cudaStreamCreateWithFlags_cpp(pStream, flags);
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)
{
    return cudaStreamCreateWithPriority_cpp(pStream, flags, priority);
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    return cudaStreamDestroy_cpp(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    return cudaStreamEndCapture_cpp(stream, pGraph);
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    return cudaStreamGetFlags_cpp(stream, flags);
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    return cudaStreamGetPriority_cpp(stream, priority);
}

extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    return cudaStreamIsCapturing_cpp(stream, pCaptureStatus);
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
    return cudaStreamQuery_cpp(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    return cudaStreamSynchronize_cpp(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return cudaStreamWaitEvent_cpp(stream, event, flags);
}

extern "C" cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *mode)
{
    return cudaThreadExchangeStreamCaptureMode_cpp(mode);
}

extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    return cudaStreamBeginCapture_cpp(stream, mode);
}

extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreate_cpp(event);
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    return cudaEventCreateWithFlags_cpp(event, flags);
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
    return cudaEventDestroy_cpp(event);
}

extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    return cudaEventElapsedTime_cpp(ms, start, end);
}

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
{
    return cudaEventQuery_cpp(event);
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return cudaEventRecord_cpp(event, stream);
}

extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    return cudaEventRecordWithFlags_cpp(event, stream, flags);
}

extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
    return cudaEventSynchronize_cpp(event);
}

extern "C" cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    return cudaFuncGetAttributes_cpp(attr, func);
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetAttributes(const void *func, int attr, int value)
{
    return cudaFuncSetAttributes_cpp(func, attr, value);
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)
{
    return cudaFuncSetCacheConfig_cpp(func, cacheConfig);
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config)
{
    return cudaFuncSetSharedMemConfig_cpp(func, config);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return cudaLaunchCooperativeKernel_cpp(func, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return cudaLaunchKernel_cpp(func, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array)
{
    return cudaArrayGetInfo_cpp(desc, extent, flags, array);
}

extern "C" cudaError_t CUDARTAPI cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array)
{
    return cudaArrayGetSparseProperties_cpp(sparseProperties, array);
}

extern "C" cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    return cudaFree_cpp(devPtr);
}

extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array)
{
    return cudaFreeArray_cpp(array);
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    return cudaFreeHost_cpp(ptr);
}

extern "C" cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
    return cudaGetSymbolAddress_cpp(devPtr, symbol);
}

extern "C" cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol)
{
    return cudaGetSymbolSize_cpp(size, symbol);
}

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    return cudaHostAlloc_cpp(ptr, size, flags);
}

extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    return cudaHostGetDevicePointer_cpp(pDevice, pHost, flags);
}

extern "C" cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    return cudaHostGetFlags_cpp(pFlags, pHost);
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    return cudaMalloc_cpp(devPtr, size);
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent)
{
    return cudaMalloc3D_cpp(pitchedDevPtr, extent);
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags)
{
    return cudaMalloc3DArray_cpp(array, desc, extent, flags);
}

extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags)
{
    return cudaMallocArray_cpp(array, desc, width, height, flags);
}

extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    return cudaMallocPitch_cpp(devPtr, pitch, width, height);
}

extern "C" cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    return cudaMemAdvise_cpp(devPtr, count, advice, device);
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
    return cudaMemGetInfo_cpp(free, total);
}

extern "C" cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    return cudaMemPrefetchAsync_cpp(devPtr, count, dstDevice, stream);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaMemcpy_cpp(dst, src, count, kind);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyHtoD(void *dst, const void *src, size_t count)
{
    return cudaMemcpyHtoD_cpp(dst, src, count);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyDtoD(void *dst, const void *src, size_t count)
{
    return cudaMemcpyDtoD_cpp(dst, src, count);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyDtoH(void *dst, const void *src, size_t count)
{
    return cudaMemcpyDtoH_cpp(dst, src, count);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return cudaMemcpyToSymbol_cpp(symbol, src, count, offset, kind);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbolShm(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return cudaMemcpyToSymbolShm_cpp(symbol, src, count, offset, kind);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyShm(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaMemcpyShm_cpp(dst, src, count, kind);
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyIB(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaMemcpyIB_cpp(dst, src, count, kind);
}

extern "C" cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    return cudaMemset_cpp(devPtr, value, count);
}

extern "C" cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return cudaMemset2D_cpp(devPtr, pitch, value, width, height);
}

extern "C" cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return cudaMemset2DAsync_cpp(devPtr, pitch, value, width, height, stream);
}

extern "C" cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    return cudaMemset3D_cpp(pitchedDevPtr, value, extent);
}

extern "C" cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return cudaMemset3DAsync_cpp(pitchedDevPtr, value, extent, stream);
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return cudaMemsetAsync_cpp(devPtr, value, count, stream);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice)
{
    return cudaDeviceDisablePeerAccess_cpp(peerDevice);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return cudaDeviceEnablePeerAccess_cpp(peerDevice, flags);
}

extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
    return cudaDriverGetVersion_cpp(driverVersion);
}

extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
    return cudaRuntimeGetVersion_cpp(runtimeVersion);
}

extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void)
{
    return cudaProfilerStart_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void)
{
    return cudaProfilerStop_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache(void)
{
    return cudaCtxResetPersistingL2Cache_cpp();
}

extern "C" cudaError_t CUDARTAPI cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
    return cudaStreamCopyAttributes_cpp(dst, src);
}