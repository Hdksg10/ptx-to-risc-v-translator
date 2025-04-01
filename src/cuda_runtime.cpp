#include <cuda_runtime.h>

#include <Interface.h>

static cudaError_t __setLastError(cudaError_t error) {
    driver::error = error;
    return error;
}

// internel functions
// internel initialization
extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant,
    int global)
{
    __cudaRegisterVar_cpp(fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global);
}
extern "C" void **__cudaRegisterFatBinary(void *fatCubin)
{
    return __cudaRegisterFatBinary_cpp(fatCubin);
}
extern "C" void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid,
        dim3 *bDim, dim3 *gDim, int *wSize)
{
    __cudaRegisterFunction_cpp(fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}
extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    __cudaUnregisterFatBinary_cpp(fatCubinHandle);
}
// internel kernel launch configuration
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    struct CUstream_st *stream = 0)
{
    return __cudaPushCallConfiguration_cpp(gridDim, blockDim, sharedMem, stream);
}

extern "C" unsigned __cudaPopCallConfiguration(void* param1, void* param2, void* param3, void* param4)
{
    return __cudaPopCallConfiguration_cpp(param1, param2, param3, param4);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    return __setLastError(cudaDeviceCanAccessPeer_cpp(canAccessPeer, device, peerDevice));
} 

extern "C" cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    return __setLastError(cudaSetDevice_cpp(device));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize(void)
{
    return __setLastError(cudaDeviceSynchronize_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
    return __setLastError(cudaChooseDevice_cpp(device, prop));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    return __setLastError(cudaDeviceGetAttribute_cpp(value, attr, device));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId(int *device, const char *pciBusId)
{
    return __setLastError(cudaDeviceGetByPCIBusId_cpp(device, pciBusId));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetCacheConfig(enum cudaFuncCache *pCacheConfig)
{
    return __setLastError(cudaDeviceGetCacheConfig_cpp(pCacheConfig));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetLimit(size_t *pValue, enum cudaLimit limit)
{
    return __setLastError(cudaDeviceGetLimit_cpp(pValue, limit));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    return __setLastError(cudaDeviceGetP2PAttribute_cpp(value, attr, srcDevice, dstDevice));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    return __setLastError(cudaDeviceGetPCIBusId_cpp(pciBusId, len, device));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig(enum cudaSharedMemConfig *pConfig)
{
    return __setLastError(cudaDeviceGetSharedMemConfig_cpp(pConfig));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority)
{
    return __setLastError(cudaDeviceGetStreamPriorityRange_cpp(leastPriority, greatestPriority));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device)
{
    return __setLastError(cudaDeviceGetTexture1DLinearMaxWidth_cpp(maxWidthInElements, fmtDesc, device));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceReset(void)
{
    return __setLastError(cudaDeviceReset_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
    return __setLastError(cudaDeviceSetCacheConfig_cpp(cacheConfig));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    return __setLastError(cudaDeviceSetLimit_cpp(limit, value));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config)
{
    return __setLastError(cudaDeviceSetSharedMemConfig_cpp(config));
}

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    return __setLastError(cudaGetDevice_cpp(device));
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    return __setLastError(cudaGetDeviceCount_cpp(count));
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceFlags(unsigned int *flags)
{
    return __setLastError(cudaGetDeviceFlags_cpp(flags));
}

extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    return __setLastError(cudaGetDeviceProperties_cpp(prop, device));
}

extern "C" cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags)
{
    return __setLastError(cudaSetDeviceFlags_cpp(flags));
}

extern "C" cudaError_t CUDARTAPI cudaSetValidDevices(int *deviceArr, int len)
{
    return __setLastError(cudaSetValidDevices_cpp(deviceArr, len));
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
    return __setLastError(cudaPeekAtLastError_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *pStream)
{
    return __setLastError(cudaStreamCreate_cpp(pStream));
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    return __setLastError(cudaStreamCreateWithFlags_cpp(pStream, flags));
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags, int priority)
{
    return __setLastError(cudaStreamCreateWithPriority_cpp(pStream, flags, priority));
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    return __setLastError(cudaStreamDestroy_cpp(stream));
}

extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph)
{
    return __setLastError(cudaStreamEndCapture_cpp(stream, pGraph));
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags)
{
    return __setLastError(cudaStreamGetFlags_cpp(stream, flags));
}

extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int *priority)
{
    return __setLastError(cudaStreamGetPriority_cpp(stream, priority));
}

extern "C" cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    return __setLastError(cudaStreamIsCapturing_cpp(stream, pCaptureStatus));
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
    return __setLastError(cudaStreamQuery_cpp(stream));
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    return __setLastError(cudaStreamSynchronize_cpp(stream));
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return __setLastError(cudaStreamWaitEvent_cpp(stream, event, flags));
}

extern "C" cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode *mode)
{
    return __setLastError(cudaThreadExchangeStreamCaptureMode_cpp(mode));
}

extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    return __setLastError(cudaStreamBeginCapture_cpp(stream, mode));
}

extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
{
    return __setLastError(cudaEventCreate_cpp(event));
}

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    return __setLastError(cudaEventCreateWithFlags_cpp(event, flags));
}

extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
    return __setLastError(cudaEventDestroy_cpp(event));
}

extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    return __setLastError(cudaEventElapsedTime_cpp(ms, start, end));
}

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
{
    return __setLastError(cudaEventQuery_cpp(event));
}

extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return __setLastError(cudaEventRecord_cpp(event, stream));
}

extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    return __setLastError(cudaEventRecordWithFlags_cpp(event, stream, flags));
}

extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
    return __setLastError(cudaEventSynchronize_cpp(event));
}

extern "C" cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    return __setLastError(cudaFuncGetAttributes_cpp(attr, func));
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetAttributes(const void *func, int attr, int value)
{
    return __setLastError(cudaFuncSetAttributes_cpp(func, attr, value));
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)
{
    return __setLastError(cudaFuncSetCacheConfig_cpp(func, cacheConfig));
}

extern "C" cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig(const void *func, enum cudaSharedMemConfig config)
{
    return __setLastError(cudaFuncSetSharedMemConfig_cpp(func, config));
}

extern "C" cudaError_t CUDARTAPI cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return __setLastError(cudaLaunchCooperativeKernel_cpp(func, gridDim, blockDim, args, sharedMem, stream));
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return __setLastError(cudaLaunchKernel_cpp(func, gridDim, blockDim, args, sharedMem, stream));
}

extern "C" cudaError_t CUDARTAPI cudaArrayGetInfo(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array)
{
    return __setLastError(cudaArrayGetInfo_cpp(desc, extent, flags, array));
}

extern "C" cudaError_t CUDARTAPI cudaArrayGetSparseProperties(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array)
{
    return __setLastError(cudaArrayGetSparseProperties_cpp(sparseProperties, array));
}

extern "C" cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    return __setLastError(cudaFree_cpp(devPtr));
}

extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array)
{
    return __setLastError(cudaFreeArray_cpp(array));
}

extern "C" cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    return __setLastError(cudaFreeHost_cpp(ptr));
}

extern "C" cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
    return __setLastError(cudaGetSymbolAddress_cpp(devPtr, symbol));
}

extern "C" cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol)
{
    return __setLastError(cudaGetSymbolSize_cpp(size, symbol));
}

extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size, unsigned int flags)
{
    return __setLastError(cudaHostAlloc_cpp(ptr, size, flags));
}

extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    return __setLastError(cudaHostGetDevicePointer_cpp(pDevice, pHost, flags));
}

extern "C" cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    return __setLastError(cudaHostGetFlags_cpp(pFlags, pHost));
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    return __setLastError(cudaMalloc_cpp(devPtr, size));
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent)
{
    return __setLastError(cudaMalloc3D_cpp(pitchedDevPtr, extent));
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags)
{
    return __setLastError(cudaMalloc3DArray_cpp(array, desc, extent, flags));
}

extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags)
{
    return __setLastError(cudaMallocArray_cpp(array, desc, width, height, flags));
}

extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    return __setLastError(cudaMallocPitch_cpp(devPtr, pitch, width, height));
}

extern "C" cudaError_t CUDARTAPI cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    return __setLastError(cudaMemAdvise_cpp(devPtr, count, advice, device));
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(size_t *free, size_t *total)
{
    return __setLastError(cudaMemGetInfo_cpp(free, total));
}

extern "C" cudaError_t CUDARTAPI cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    return __setLastError(cudaMemPrefetchAsync_cpp(devPtr, count, dstDevice, stream));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return __setLastError(cudaMemcpy_cpp(dst, src, count, kind));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyHtoD(void *dst, const void *src, size_t count)
{
    return __setLastError(cudaMemcpyHtoD_cpp(dst, src, count));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyDtoD(void *dst, const void *src, size_t count)
{
    return __setLastError(cudaMemcpyDtoD_cpp(dst, src, count));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyDtoH(void *dst, const void *src, size_t count)
{
    return __setLastError(cudaMemcpyDtoH_cpp(dst, src, count));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return __setLastError(cudaMemcpyToSymbol_cpp(symbol, src, count, offset, kind));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyToSymbolShm(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return __setLastError(cudaMemcpyToSymbolShm_cpp(symbol, src, count, offset, kind));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyShm(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return __setLastError(cudaMemcpyShm_cpp(dst, src, count, kind));
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyIB(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return __setLastError(cudaMemcpyIB_cpp(dst, src, count, kind));
}

extern "C" cudaError_t CUDARTAPI cudaMemset(void *devPtr, int value, size_t count)
{
    return __setLastError(cudaMemset_cpp(devPtr, value, count));
}

extern "C" cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return __setLastError(cudaMemset2D_cpp(devPtr, pitch, value, width, height));
}

extern "C" cudaError_t CUDARTAPI cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return __setLastError(cudaMemset2DAsync_cpp(devPtr, pitch, value, width, height, stream));
}

extern "C" cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    return __setLastError(cudaMemset3D_cpp(pitchedDevPtr, value, extent));
}

extern "C" cudaError_t CUDARTAPI cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return __setLastError(cudaMemset3DAsync_cpp(pitchedDevPtr, value, extent, stream));
}

extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return __setLastError(cudaMemsetAsync_cpp(devPtr, value, count, stream));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice)
{
    return __setLastError(cudaDeviceDisablePeerAccess_cpp(peerDevice));
}

extern "C" cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return __setLastError(cudaDeviceEnablePeerAccess_cpp(peerDevice, flags));
}

extern "C" cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
    return __setLastError(cudaDriverGetVersion_cpp(driverVersion));
}

extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
    return __setLastError(cudaRuntimeGetVersion_cpp(runtimeVersion));
}

extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void)
{
    return __setLastError(cudaProfilerStart_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void)
{
    return __setLastError(cudaProfilerStop_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache(void)
{
    return __setLastError(cudaCtxResetPersistingL2Cache_cpp());
}

extern "C" cudaError_t CUDARTAPI cudaStreamCopyAttributes(cudaStream_t dst, cudaStream_t src)
{
    return __setLastError(cudaStreamCopyAttributes_cpp(dst, src));
}

extern "C" cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags)
{
    return __setLastError(cudaGraphInstantiate_cpp(pGraphExec, graph, flags));
}

extern "C" cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes)
{
    return __setLastError(cudaGraphGetNodes_cpp(graph, nodes, numNodes));
}

