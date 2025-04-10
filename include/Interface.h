#ifndef INTERFACE_H
#define INTERFACE_H

#include <cstring>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDAContext.h>
#include <EmulatedCUDADevice.h>
#include <CUDAModule.h>
#include <CUDAFunction.h>
#include <CUDAEvent.h>
#include <log.h>

// CUDA Types
struct fatDeviceText {
    uint32_t magic;
    uint32_t version;
    void*    fatbin;
    uint64_t data; 
};

const uint32_t FATTEXT_MAGIC = 0x466243b1; 

// CUDA Driver API
// driver management
CUresult CUDAAPI cuInit_cpp(unsigned int Flags);
CUresult CUDAAPI cuDeviceGetCount_cpp(int *count);
CUresult CUDAAPI cuDriverGetVersion_cpp(int *driverVersion);

// device management
CUresult CUDAAPI cuDeviceGet_cpp(CUdevice *device, int ordinal);
CUresult CUDAAPI cuDeviceGetName_cpp(char *name, int len, CUdevice dev);
CUresult CUDAAPI cuDeviceComputeCapability_cpp(int *major, int *minor, CUdevice dev);
CUresult CUDAAPI cuDeviceTotalMem_cpp(size_t *bytes, CUdevice dev);
CUresult CUDAAPI cuDeviceGetUuid_cpp(CUuuid *uuid, CUdevice dev);
CUresult CUDAAPI cuDeviceGetAttribute_cpp(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxRetain_cpp(CUcontext *pctx, CUdevice dev);
CUresult CUDAAPI cuDevicePrimaryCtxGetState_cpp( CUdevice dev, unsigned int* flags, int* active );
CUresult CUDAAPI cuDeviceGetP2PAttribute_cpp(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice);


// context management
CUresult CUDAAPI cuCtxCreate_cpp(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult CUDAAPI cuCtxCreate_v3_cpp(CUcontext* pctx, CUexecAffinityParam* paramsArray, int numParams, unsigned int flags, CUdevice dev); 
CUresult CUDAAPI cuCtxSetCurrent_cpp(CUcontext ctx);
CUresult CUDAAPI cuCtxGetCurrent_cpp(CUcontext *pctx);
CUresult CUDAAPI cuCtxGetDevice_cpp(CUdevice *device);
CUresult CUDAAPI cuCtxDestroy_cpp(CUcontext ctx);
CUresult CUDAAPI cuCtxSynchronize_cpp();
CUresult CUDAAPI cuCtxGetApiVersion_cpp ( CUcontext ctx, unsigned int* version );

// module management
CUresult CUDAAPI cuModuleLoad_cpp(CUmodule *module, const char *fname);
CUresult CUDAAPI cuModuleLoadData_cpp(CUmodule *module, const void *image);
CUresult CUDAAPI cuModuleLoadDataEx_cpp(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult CUDAAPI cuModuleUnload_cpp(CUmodule hmod);
CUresult CUDAAPI cuModuleGetFunction_cpp(CUfunction *hfunc, CUmodule hmod, const char *name);
CUresult CUDAAPI cuModuleGetGlobal_cpp(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);

// driver memory managment
CUresult CUDAAPI cuMemAlloc_cpp(CUdeviceptr* dptr, size_t bytesize);
CUresult CUDAAPI cuMemFree_cpp(CUdeviceptr dptr);
CUresult CUDAAPI cuMemcpyHtoD_cpp(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult CUDAAPI cuMemcpyDtoH_cpp(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
CUresult CUDAAPI cuMemsetD32_cpp (CUdeviceptr dstDevice, unsigned int ui, size_t N);
CUresult CUDAAPI cuMemGetInfo_cpp(size_t *free, size_t *total);
CUresult CUDAAPI cuMemGetAllocationGranularity_cpp(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option);


// helper functions
CUresult cuGetErrorName_cpp( CUresult error, const char** pStr );
CUresult cuGetErrorString_cpp(CUresult error, const char **pStr);

// kernel launch
CUresult CUDAAPI cuLaunchKernel_cpp(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra);

CUresult cuFuncGetAttribute_cpp( int* pi, CUfunction_attribute attrib, CUfunction hfunc );
CUresult cuFuncSetAttribute_cpp( CUfunction hfunc, CUfunction_attribute attrib, int  value );
CUresult cuFuncSetCacheConfig_cpp( CUfunction hfunc, CUfunc_cache config );
CUresult cuFuncSetSharedMemConfig_cpp( CUfunction hfunc, CUsharedconfig config );

// CUDA Runtime API
// internel initialization
void __cudaRegisterVar_cpp(void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant,
    int global);
void **__cudaRegisterFatBinary_cpp(void *fatCubin);  
void __cudaRegisterFunction_cpp(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName,
        int thread_limit, uint3 *tid, uint3 *bid,
        dim3 *bDim, dim3 *gDim, int *wSize);
void __cudaUnregisterFatBinary_cpp(void **fatCubinHandle);
// internel kernel launch configuration
unsigned __cudaPushCallConfiguration_cpp(dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    struct CUstream_st *stream = 0);

unsigned __cudaPopCallConfiguration_cpp(void* param1, void* param2, void* param3, void* param4);
  

cudaError_t CUDARTAPI cudaDeviceCanAccessPeer_cpp(int *canAccessPeer, int device, int peerDevice);
cudaError_t CUDARTAPI cudaSetDevice_cpp(int device);

cudaError_t CUDARTAPI cudaDeviceSynchronize_cpp(void);
cudaError_t CUDARTAPI cudaChooseDevice_cpp(int *device, const struct cudaDeviceProp *prop);
cudaError_t CUDARTAPI cudaDeviceGetAttribute_cpp(int *value, enum cudaDeviceAttr attr, int device);
cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId_cpp(int *device, const char *pciBusId);
cudaError_t CUDARTAPI cudaDeviceGetCacheConfig_cpp(enum cudaFuncCache *pCacheConfig);
cudaError_t CUDARTAPI cudaDeviceGetLimit_cpp(size_t *pValue, enum cudaLimit limit);
cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute_cpp(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice);
cudaError_t CUDARTAPI cudaDeviceGetPCIBusId_cpp(char *pciBusId, int len, int device);
cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig_cpp(enum cudaSharedMemConfig *pConfig);
cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange_cpp(int *leastPriority, int *greatestPriority);
cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth_cpp(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device);
cudaError_t CUDARTAPI cudaDeviceReset_cpp(void);
cudaError_t CUDARTAPI cudaDeviceSetCacheConfig_cpp(enum cudaFuncCache cacheConfig);
cudaError_t CUDARTAPI cudaDeviceSetLimit_cpp(enum cudaLimit limit, size_t value);
cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig_cpp(enum cudaSharedMemConfig config);
cudaError_t CUDARTAPI cudaGetDevice_cpp(int *device);
cudaError_t CUDARTAPI cudaGetDeviceCount_cpp(int *count);
cudaError_t CUDARTAPI cudaGetDeviceFlags_cpp(unsigned int *flags);
cudaError_t CUDARTAPI cudaGetDeviceProperties_cpp(struct cudaDeviceProp *prop, int device);
cudaError_t CUDARTAPI cudaSetDeviceFlags_cpp(unsigned int flags);
cudaError_t CUDARTAPI cudaSetValidDevices_cpp(int *deviceArr, int len);
const char* CUDARTAPI cudaGetErrorName_cpp(cudaError_t error);
const char* CUDARTAPI cudaGetErrorString_cpp(cudaError_t error);
cudaError_t CUDARTAPI cudaGetLastError_cpp(void);
cudaError_t CUDARTAPI cudaPeekAtLastError_cpp(void);
cudaError_t CUDARTAPI cudaStreamCreate_cpp(cudaStream_t *pStream);
cudaError_t CUDARTAPI cudaStreamCreateWithFlags_cpp(cudaStream_t *pStream, unsigned int flags);
cudaError_t CUDARTAPI cudaStreamCreateWithPriority_cpp(cudaStream_t *pStream, unsigned int flags, int priority);
cudaError_t CUDARTAPI cudaStreamDestroy_cpp(cudaStream_t stream);
cudaError_t CUDARTAPI cudaStreamEndCapture_cpp(cudaStream_t stream, cudaGraph_t *pGraph);
cudaError_t CUDARTAPI cudaStreamGetFlags_cpp(cudaStream_t stream, unsigned int *flags);
cudaError_t CUDARTAPI cudaStreamGetPriority_cpp(cudaStream_t stream, int *priority);
cudaError_t CUDARTAPI cudaStreamIsCapturing_cpp(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus);
cudaError_t CUDARTAPI cudaStreamQuery_cpp(cudaStream_t stream);
cudaError_t CUDARTAPI cudaStreamSynchronize_cpp(cudaStream_t stream);
cudaError_t CUDARTAPI cudaStreamWaitEvent_cpp(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode_cpp(cudaStreamCaptureMode *mode);
cudaError_t CUDARTAPI cudaStreamBeginCapture_cpp(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t CUDARTAPI cudaEventCreate_cpp(cudaEvent_t *event);
cudaError_t CUDARTAPI cudaEventCreateWithFlags_cpp(cudaEvent_t *event, unsigned int flags);
cudaError_t CUDARTAPI cudaEventDestroy_cpp(cudaEvent_t event);
cudaError_t CUDARTAPI cudaEventElapsedTime_cpp(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t CUDARTAPI cudaEventQuery_cpp(cudaEvent_t event);
cudaError_t CUDARTAPI cudaEventRecord_cpp(cudaEvent_t event, cudaStream_t stream);
cudaError_t CUDARTAPI cudaEventRecordWithFlags_cpp(cudaEvent_t event, cudaStream_t stream, unsigned int flags);
cudaError_t CUDARTAPI cudaEventSynchronize_cpp(cudaEvent_t event);
cudaError_t CUDARTAPI cudaFuncGetAttributes_cpp(struct cudaFuncAttributes *attr, const void *func);
cudaError_t CUDARTAPI cudaFuncSetAttributes_cpp(const void *func, int attr, int value);
cudaError_t CUDARTAPI cudaFuncSetCacheConfig_cpp(const void *func, enum cudaFuncCache cacheConfig);
cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig_cpp(const void *func, enum cudaSharedMemConfig config);
cudaError_t CUDARTAPI cudaLaunchCooperativeKernel_cpp(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t CUDARTAPI cudaLaunchKernel_cpp(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream);
cudaError_t CUDARTAPI cudaArrayGetInfo_cpp(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array);
cudaError_t CUDARTAPI cudaArrayGetSparseProperties_cpp(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array);
cudaError_t CUDARTAPI cudaFree_cpp(void *devPtr);
cudaError_t CUDARTAPI cudaFreeArray_cpp(cudaArray_t array);
cudaError_t CUDARTAPI cudaFreeHost_cpp(void *ptr);
cudaError_t CUDARTAPI cudaGetSymbolAddress_cpp(void **devPtr, const void *symbol);
cudaError_t CUDARTAPI cudaGetSymbolSize_cpp(size_t *size, const void *symbol);
cudaError_t CUDARTAPI cudaHostAlloc_cpp(void **ptr, size_t size, unsigned int flags);
cudaError_t CUDARTAPI cudaHostGetDevicePointer_cpp(void **pDevice, void *pHost, unsigned int flags);
cudaError_t CUDARTAPI cudaHostGetFlags_cpp(unsigned int *pFlags, void *pHost);
cudaError_t CUDARTAPI cudaMalloc_cpp(void **devPtr, size_t size);
cudaError_t CUDARTAPI cudaMalloc3D_cpp(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent);
cudaError_t CUDARTAPI cudaMalloc3DArray_cpp(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags);
cudaError_t CUDARTAPI cudaMallocArray_cpp(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags);
cudaError_t CUDARTAPI cudaMallocPitch_cpp(void **devPtr, size_t *pitch, size_t width, size_t height);
cudaError_t CUDARTAPI cudaMemAdvise_cpp(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
cudaError_t CUDARTAPI cudaMemGetInfo_cpp(size_t *free, size_t *total);
cudaError_t CUDARTAPI cudaMemPrefetchAsync_cpp(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream);
cudaError_t CUDARTAPI cudaMemcpy_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t CUDARTAPI cudaMemcpyHtoD_cpp(void *dst, const void *src, size_t count);
cudaError_t CUDARTAPI cudaMemcpyDtoD_cpp(void *dst, const void *src, size_t count);
cudaError_t CUDARTAPI cudaMemcpyDtoH_cpp(void *dst, const void *src, size_t count);
cudaError_t CUDARTAPI cudaMemcpyToSymbol_cpp(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t CUDARTAPI cudaMemcpyToSymbolShm_cpp(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind);
cudaError_t CUDARTAPI cudaMemcpyShm_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t CUDARTAPI cudaMemcpyIB_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t CUDARTAPI cudaMemset_cpp(void *devPtr, int value, size_t count);
cudaError_t CUDARTAPI cudaMemset2D_cpp(void *devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t CUDARTAPI cudaMemset2DAsync_cpp(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream);
cudaError_t CUDARTAPI cudaMemset3D_cpp(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent);
cudaError_t CUDARTAPI cudaMemset3DAsync_cpp(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream);
cudaError_t CUDARTAPI cudaMemsetAsync_cpp(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess_cpp(int peerDevice);
cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess_cpp(int peerDevice, unsigned int flags);
cudaError_t CUDARTAPI cudaDriverGetVersion_cpp(int *driverVersion);
cudaError_t CUDARTAPI cudaRuntimeGetVersion_cpp(int *runtimeVersion);
cudaError_t CUDARTAPI cudaProfilerStart_cpp(void);
cudaError_t CUDARTAPI cudaProfilerStop_cpp(void);
cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache_cpp(void);
cudaError_t CUDARTAPI cudaStreamCopyAttributes_cpp(cudaStream_t dst, cudaStream_t src);

cudaError_t CUDARTAPI cudaGraphInstantiate_cpp ( cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags );

cudaError_t CUDARTAPI cudaGraphGetNodes_cpp ( cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes );

#endif