#include "EmulatedCUDADevice.h"
#include "cuda.h"
#include <Interface.h>

CUresult CUDAAPI cuDeviceGetCount_cpp(int *count) {
    // Implementation to get the number of CUDA devices
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer
    // For the translator, we always return 1 device (the emulated one)
    *count = driver::MAX_DEVICES; 
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuInit_cpp(unsigned int Flags) {
    // Implementation to initialize the CUDA driver
    // This is a placeholder implementation
    if (driver::driverInitialized) return CUDA_SUCCESS; // do nothing
    for (int i = 0; i < driver::MAX_DEVICES; i++) {
        driver::devices[i] = new driver::EmulatedCUDADevice();
        driver::devices[i]->initialize(Flags);
    }
    driver::driverInitialized = true; // Mark the driver as initialized
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDriverGetVersion_cpp(int *driverVersion) {
    // Implementation to get the version of the CUDA driver
    if (driverVersion == NULL) return CUDA_ERROR_INVALID_VALUE; // Check for null pointer (invalid argument)
    *driverVersion = driver::DRIVER_VERSION; // Version PTX 5.0. CUDA 8.0
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGet_cpp(CUdevice *device, int ordinal) {
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

CUresult CUDAAPI cuDeviceGetName_cpp(char *name, int len, CUdevice dev) {
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

CUresult CUDAAPI cuDeviceTotalMem_cpp(size_t *bytes, CUdevice dev) {
    // Implementation to get the total memory of the device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    *bytes = driver::devices[dev]->totalMemBytes;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetUuid_cpp(CUuuid *uuid, CUdevice dev) {
    // Implementation to get the UUID of the device
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (uuid == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    memcpy(uuid, &driver::devices[dev]->uuid, 16);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetAttribute_cpp(int *pi, CUdevice_attribute attrib, CUdevice dev) {
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

CUresult CUDAAPI cuDevicePrimaryCtxRetain_cpp(CUcontext *pctx, CUdevice dev) {
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

CUresult CUDAAPI cuCtxCreate_cpp(CUcontext *pctx, unsigned int flags, CUdevice dev) {
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
CUresult CUDAAPI cuCtxSetCurrent_cpp(CUcontext ctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ctx == nullptr) return CUDA_ERROR_INVALID_CONTEXT; // Check for valid context handle
    auto context = driver::contextStack.top();
    context->setCtx(ctx);
    return CUDA_SUCCESS;
}
// Note that we havn't implemented the context stack management yet, so cuCtxGetCurrent will not work as expected.
CUresult CUDAAPI cuCtxGetCurrent_cpp(CUcontext *pctx) {
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

CUresult CUDAAPI cuCtxGetDevice_cpp(CUdevice *device) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    *device = context->getContext()->device;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy_cpp(CUcontext ctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid context
    ctx->destroyed = 1;
    ctx->valid = 0;
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuModuleLoad_cpp(CUmodule *module, const char *fname) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (module == nullptr || fname == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        struct CUmod_st *mod = new struct CUmod_st();
        auto inner_module = new driver::CUDAModule();
        if (!(inner_module->load(fname))) {
            delete inner_module;
            delete mod;
            return CUDA_ERROR_FILE_NOT_FOUND;
        }
        mod->module = inner_module;
        *module = mod;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }
}

CUresult CUDAAPI cuModuleUnload_cpp(CUmodule hmod) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (hmod == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        driver::CUDAModule *inner_module = static_cast<driver::CUDAModule*>(hmod->module);
        inner_module->unload();
        delete inner_module;
        delete hmod;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CUDAAPI cuModuleGetFunction_cpp(CUfunction *hfunc, CUmodule hmod, const char *name) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (hmod == nullptr || hfunc == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        driver::CUDAModule *inner_module = static_cast<driver::CUDAModule*>(hmod->module);
        *hfunc = inner_module->getFunction(name);
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CUDAAPI cuModuleLoadData_cpp(CUmodule *module, const void *image) {
    // Placeholder implementation
    return CUDA_ERROR_NOT_SUPPORTED;
}

// CUDA Runtime API
cudaError_t CUDARTAPI cudaDeviceCanAccessPeer_cpp(int *canAccessPeer, int device, int peerDevice) 
{
    // placeholder implementation
    // set canAccessPeer to 0, indicating no peer access
    if (canAccessPeer == nullptr) {
        return cudaErrorInvalidValue;
    }
    *canAccessPeer = 0;
    return cudaSuccess;  // return success for the placeholder implementation
}

cudaError_t CUDARTAPI cudaSetDevice_cpp(int device)
{
    CUresult r;
    // init driver
    r = cuInit_cpp(0);
    if (r != CUDA_SUCCESS) {
        return cudaErrorInitializationError;
    }
    // get device
    CUdevice cuDevice;
    r = cuDeviceGet_cpp(&cuDevice, device);
    if (r != CUDA_SUCCESS) {
        return cudaErrorInvalidDevice;
    }
    // create context
    CUcontext ctx;
    r = cuCtxCreate_cpp(&ctx, 0, cuDevice);
    if (r != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    // set device primary context
    driver::devices[device]->context = static_cast<driver::CUDAContext*>(ctx->outerContex);
    return cudaSuccess;  // return success for the placeholder implementation
}

// Virtual implementation
cudaError_t CUDARTAPI cudaDeviceSynchronize_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaChooseDevice_cpp(int *device, const struct cudaDeviceProp *prop)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetAttribute_cpp(int *value, enum cudaDeviceAttr attr, int device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetByPCIBusId_cpp(int *device, const char *pciBusId)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetCacheConfig_cpp(enum cudaFuncCache *pCacheConfig)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetLimit_cpp(size_t *pValue, enum cudaLimit limit)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute_cpp(int *value, enum cudaDeviceP2PAttr attr, int srcDevice, int dstDevice)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId_cpp(char *pciBusId, int len, int device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetSharedMemConfig_cpp(enum cudaSharedMemConfig *pConfig)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange_cpp(int *leastPriority, int *greatestPriority)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceGetTexture1DLinearMaxWidth_cpp(size_t *maxWidthInElements, const struct cudaChannelFormatDesc *fmtDesc, int device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceReset_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceSetCacheConfig_cpp(enum cudaFuncCache cacheConfig)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceSetLimit_cpp(enum cudaLimit limit, size_t value)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceSetSharedMemConfig_cpp(enum cudaSharedMemConfig config)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetDevice_cpp(int *device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetDeviceCount_cpp(int *count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetDeviceFlags_cpp(unsigned int *flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetDeviceProperties_cpp(struct cudaDeviceProp *prop, int device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags_cpp(unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetValidDevices_cpp(int *deviceArr, int len)
{
    return cudaErrorUnknown;
}

const char* CUDARTAPI cudaGetErrorName_cpp(cudaError_t error)
{
    return "cudaErrorUnknown";
}

const char* CUDARTAPI cudaGetErrorString_cpp(cudaError_t error)
{
    return "Unknown error";
}

cudaError_t CUDARTAPI cudaGetLastError_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaPeekAtLastError_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamCreate_cpp(cudaStream_t *pStream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamCreateWithFlags_cpp(cudaStream_t *pStream, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamCreateWithPriority_cpp(cudaStream_t *pStream, unsigned int flags, int priority)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamDestroy_cpp(cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamEndCapture_cpp(cudaStream_t stream, cudaGraph_t *pGraph)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamGetFlags_cpp(cudaStream_t stream, unsigned int *flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamGetPriority_cpp(cudaStream_t stream, int *priority)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamIsCapturing_cpp(cudaStream_t stream, cudaStreamCaptureStatus *pCaptureStatus)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamQuery_cpp(cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamSynchronize_cpp(cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamWaitEvent_cpp(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode_cpp(cudaStreamCaptureMode *mode)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamBeginCapture_cpp(cudaStream_t stream, cudaStreamCaptureMode mode)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventCreate_cpp(cudaEvent_t *event)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags_cpp(cudaEvent_t *event, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventDestroy_cpp(cudaEvent_t event)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventElapsedTime_cpp(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventQuery_cpp(cudaEvent_t event)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventRecord_cpp(cudaEvent_t event, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventRecordWithFlags_cpp(cudaEvent_t event, cudaStream_t stream, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventSynchronize_cpp(cudaEvent_t event)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncGetAttributes_cpp(struct cudaFuncAttributes *attr, const void *func)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncSetAttributes_cpp(const void *func, int attr, int value)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncSetCacheConfig_cpp(const void *func, enum cudaFuncCache cacheConfig)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncSetSharedMemConfig_cpp(const void *func, enum cudaSharedMemConfig config)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaLaunchCooperativeKernel_cpp(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaLaunchKernel_cpp(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaArrayGetInfo_cpp(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaArrayGetSparseProperties_cpp(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFree_cpp(void *devPtr)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFreeArray_cpp(cudaArray_t array)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFreeHost_cpp(void *ptr)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetSymbolAddress_cpp(void **devPtr, const void *symbol)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaGetSymbolSize_cpp(size_t *size, const void *symbol)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaHostAlloc_cpp(void **ptr, size_t size, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer_cpp(void **pDevice, void *pHost, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaHostGetFlags_cpp(unsigned int *pFlags, void *pHost)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMalloc_cpp(void **devPtr, size_t size)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMalloc3D_cpp(struct cudaPitchedPtr *pitchedDevPtr, struct cudaExtent extent)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMalloc3DArray_cpp(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, struct cudaExtent extent, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMallocArray_cpp(cudaArray_t *array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMallocPitch_cpp(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemAdvise_cpp(const void *devPtr, size_t count, enum cudaMemoryAdvise advice, int device)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemGetInfo_cpp(size_t *free, size_t *total)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemPrefetchAsync_cpp(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpy_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyHtoD_cpp(void *dst, const void *src, size_t count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyDtoD_cpp(void *dst, const void *src, size_t count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyDtoH_cpp(void *dst, const void *src, size_t count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyToSymbol_cpp(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyToSymbolShm_cpp(const void *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyShm_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyIB_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemset_cpp(void *devPtr, int value, size_t count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemset2D_cpp(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemset2DAsync_cpp(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemset3D_cpp(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemset3DAsync_cpp(struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemsetAsync_cpp(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceDisablePeerAccess_cpp(int peerDevice)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess_cpp(int peerDevice, unsigned int flags)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDriverGetVersion_cpp(int *driverVersion)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion_cpp(int *runtimeVersion)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaProfilerStart_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaProfilerStop_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaCtxResetPersistingL2Cache_cpp(void)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaStreamCopyAttributes_cpp(cudaStream_t dst, cudaStream_t src)
{
    return cudaErrorUnknown;
}