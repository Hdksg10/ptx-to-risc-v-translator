#include "EmulatedCUDADevice.h"
#include "cuda.h"
#include "driver_types.h"
#include <Interface.h>
#include <cstdint>


static cudaError_t __driverErrorToRuntime(CUresult err) {
    switch (err) {
        case CUDA_SUCCESS: return cudaSuccess;
        case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
        case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
        case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError ;
        case CUDA_ERROR_DEINITIALIZED: return cudaErrorInitializationError ;
        case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
        case CUDA_ERROR_INVALID_CONTEXT: return cudaErrorIncompatibleDriverContext;
        default: return cudaErrorUnknown;
    }
}



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
    std::cout << "===== Hello from RISC-V CUDA Driver Library! =====" << std::endl;
    // Implementation to initialize the CUDA driver
    // This is a placeholder implementation
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Initializing CUDA driver with flags: %u", Flags);
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
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Retaining primary context for device %d", dev);
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    
    // Check if the primary context is already retained
    if (driver::devices[dev]->context != nullptr) {
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Primary context %p already retained for device %d", driver::devices[dev]->context, dev);
        *pctx = driver::devices[dev]->context->getContext();
        return CUDA_SUCCESS;
    }
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Creating new primary context for device %d", dev);
    // Create a new primary context for the device
    driver::devices[dev]->context = new driver::CUDAContext();
    driver::devices[dev]->context->create(dev, 0); // Assuming default flags for primary context
    *pctx = driver::devices[dev]->context->getContext();
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Primary context %p created and retained for device %d", driver::devices[dev]->context, dev);
    driver::contextStack.push(driver::devices[dev]->context);
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuDeviceGetP2PAttribute_cpp(int *value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    if (srcDevice == dstDevice || srcDevice >= driver::MAX_DEVICES || dstDevice >= driver::MAX_DEVICES) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    // Not implemented yet
    switch (attrib) {
        case CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK:
           *value = 0;
            break;
        case CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED:
            *value = 0;
            break;
        case CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED:
            *value = 0;
            break;
        default:
            return CUDA_ERROR_INVALID_VALUE;
    }
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

CUresult CUDAAPI cuCtxCreate_v3_cpp(CUcontext* pctx, CUexecAffinityParam* paramsArray, int  numParams, unsigned int  flags, CUdevice dev) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dev < 0 || dev >= driver::MAX_DEVICES) return CUDA_ERROR_INVALID_DEVICE; // Check for valid device handle
    if (pctx == nullptr) return CUDA_ERROR_INVALID_VALUE; // Check for valid pointer
    if (numParams < 0) return CUDA_ERROR_INVALID_VALUE; // Check for valid number of parameters
    // Not implemented yet
    *pctx = nullptr;
    return CUDA_ERROR_UNKNOWN;
}

// Note that we havn't implemented the context stack management yet, so cuCtxSetCurrent will not work as expected.
CUresult CUDAAPI cuCtxSetCurrent_cpp(CUcontext ctx) {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (ctx == nullptr) return CUDA_ERROR_INVALID_CONTEXT; // Check for valid context handle
    auto context = static_cast<driver::CUDAContext*>(ctx->outerContext);
    driver::contextStack.push(context);
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
CUresult CUDAAPI cuCtxSynchronize_cpp() {
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (!context->valid()) return CUDA_ERROR_INVALID_CONTEXT;

    // We have synchronized current context in Ocelot translation-execution module

    // Do nothing here

    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuModuleLoad_cpp(CUmodule *module, const char *fname) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (module == nullptr || fname == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
        auto context = driver::contextStack.top();
        if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT; // Check for valid context
        struct CUmod_st *mod = new struct CUmod_st();
        auto inner_module = new driver::CUDAModule();
        std::string fname_str(fname); // Convert const char* to std::string 
        if (!(inner_module->load(fname_str))) {
            delete inner_module;
            delete mod;
            return CUDA_ERROR_FILE_NOT_FOUND;
        }
        driver::devices[context->getContext()->device]->load(inner_module);
        mod->module = inner_module;
        mod->context = context;
        *module = mod;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        return CUDA_ERROR_UNSUPPORTED_PTX_VERSION;
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
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Getting function from module %p with name: %s", hmod, name);
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

CUresult CUDAAPI cuModuleGetGlobal_cpp(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Getting global variable from module %p with name: %s", hmod, name);
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (hmod == nullptr || dptr == nullptr || bytes == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        driver::CUDAModule *inner_module = static_cast<driver::CUDAModule*>(hmod->module);
        auto [global, size]  = inner_module->getGlobal(name);
        *dptr = global;
        *bytes = size;
        return CUDA_SUCCESS;
    }
    catch (const std::exception& e) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CUDAAPI cuModuleLoadData_cpp(CUmodule *module, const void *image) {
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Loading module data from image. image ptr: %p", image);
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (module == nullptr || image == nullptr) return CUDA_ERROR_INVALID_VALUE;
    try {
        if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
        auto context = driver::contextStack.top();
        if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT; // Check for valid context
        struct CUmod_st *mod = new struct CUmod_st();
        auto inner_module = new driver::CUDAModule();
        if (!(inner_module->load(image))) {
            delete inner_module;
            delete mod;
            return CUDA_ERROR_FILE_NOT_FOUND;
        }
        driver::devices[context->getContext()->device]->load(inner_module);
        mod->module = inner_module;
        mod->context = context;
        *module = mod;
        return CUDA_SUCCESS;
    } catch (const std::exception& e) {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Failed to load module: %s", e.what());
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult cuMemAlloc_cpp(CUdeviceptr* dptr, size_t bytesize)
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dptr == nullptr || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;
    if (!context->allocate(dptr, bytesize)) return CUDA_ERROR_OUT_OF_MEMORY;
    return CUDA_SUCCESS;
}

CUresult cuMemFree_cpp(CUdeviceptr dptr) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dptr == 0) return CUDA_ERROR_INVALID_VALUE;

    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;
    if (!context->free(dptr)) return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}
CUresult cuMemcpyHtoD_cpp(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dstDevice == 0 || srcHost == nullptr || ByteCount == 0) return CUDA_ERROR_INVALID_VALUE;

    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;

    // check device memory allocation
    if (!context->getAllocatedSize(dstDevice)) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice)), srcHost, ByteCount);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyDtoH_cpp(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (srcDevice == 0 || dstHost == nullptr || ByteCount == 0) return CUDA_ERROR_INVALID_VALUE;

    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;

    // check device memory allocation
    if (!context->getAllocatedSize(srcDevice)) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(dstHost, reinterpret_cast<void*>(static_cast<uintptr_t>(srcDevice)), ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32_cpp(CUdeviceptr dstDevice, unsigned int ui, size_t N) 
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (dstDevice == 0) return CUDA_ERROR_INVALID_VALUE;

    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;

    // check device memory allocation
    if (!context->getAllocatedSize(dstDevice)) return CUDA_ERROR_INVALID_VALUE;
    std::memset(reinterpret_cast<void*>(static_cast<uintptr_t>(dstDevice)), ui, N * sizeof(unsigned int)); // memset device memory
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuMemGetInfo_cpp(size_t *free, size_t *total)
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;
    if (free == nullptr || total == nullptr) return CUDA_ERROR_INVALID_VALUE;
    // Not implemented yet
    return CUDA_ERROR_UNKNOWN;
}
CUresult CUDAAPI cuMemGetAllocationGranularity_cpp(size_t* granularity, const CUmemAllocationProp* prop, CUmemAllocationGranularity_flags option)
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;
    if (granularity == nullptr || prop == nullptr) return CUDA_ERROR_INVALID_VALUE;
    // Not implemented yet
    return CUDA_ERROR_UNKNOWN;
}

CUresult CUDAAPI cuLaunchKernel_cpp(CUfunction f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream hStream,
    void **kernelParams,
    void **extra)
{
    if (driver::driverDeinitialized) return CUDA_ERROR_DEINITIALIZED;
    if (!driver::driverInitialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (f == nullptr) return CUDA_ERROR_INVALID_VALUE;

    // get current context;
    if (driver::contextStack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return CUDA_ERROR_INVALID_CONTEXT;
    
    // launch kernel
    driver::devices[context->getContext()->device]->launchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
    return CUDA_SUCCESS;
}

// CUDA Runtime API



__attribute__((constructor)) static void __cudaRuntimeInit()
{
    LOG(LOG_LEVEL_INFO, "INFO", "Init CUDA Runtime");
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Init driver global variables");
    CUresult r;
    // init driver
    r = cuInit_cpp(0);
    if (r != CUDA_SUCCESS) {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Failed to initialize CUDA driver");
        return ;
    }
    for (int i = 0; i < 1; i++) {
        // cudaSetDevice will init device 0 in device tables and set current context to device 0
        cudaSetDevice_cpp(i);
    }
}
// Internal registration functions
// TODO: Internel registration functions should be isolated from runtime context/module management.
//       In current implementation, __cudaRegisterFatBinary_cpp will call cudaSetDevice_cpp to init device 0, and register fatbinary and functions in the primary context of device 0.
void __cudaRegisterVar_cpp(void **fatCubinHandle, char *hostVar, char *deviceAddress,
    const char *deviceName, int ext, size_t size, int constant,
    int global) {
        return;
    }

void **__cudaRegisterFatBinary_cpp(void *fatCubin)
{   
    // init cuda runtime
    // __cudaRuntimeInit();
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "__cudaRegisterFatBinary_cpp(fatcubin: %p)", fatCubin);
    fatDeviceText * fatCubinHandle = reinterpret_cast<fatDeviceText *>(fatCubin);
    if (fatCubinHandle->magic != FATTEXT_MAGIC) {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Invalid fatCubin magic number: %x", fatCubinHandle->magic);
        return nullptr;
    }
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Invalid CUDA context");
        return nullptr;
    }
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Registering fatbinary with context: %p, fatbin version: %d, fatbin address: %p", context, fatCubinHandle->version, fatCubinHandle->fatbin);
    if (context->registerFatbinary(fatCubinHandle->fatbin)) {
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Fatbinary registered successfully");
        // fatDeviceText * fatCubinHandle = static_cast<fatDeviceText *>(calloc(1, 24));
        // fatCubinHandle->magic = FATTEXT_MAGIC;
        // fatCubinHandle->version = 0x1;
        // fatCubinHandle->fatbin = fatCubin;
        // fatCubinHandle->data = 0;
        return reinterpret_cast<void **>(fatCubinHandle);  // Cast to void** as expected by the CUDA runtime API.
    }
    else {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Failed to register fatbinary");
        return nullptr;  // Return nullptr if registration fails.
    }
}

void __cudaRegisterFunction_cpp(void **fatCubinHandle, const char *hostFun,
    char *deviceFun, const char *deviceName,
    int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize)
{
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Registering function %p with device function %s at address %p in fatbinary handler %p.", hostFun, deviceName, deviceFun, fatCubinHandle);
    auto context = driver::contextStack.top();
    fatDeviceText * fatCubin = reinterpret_cast<fatDeviceText *>(fatCubinHandle);
    if (fatCubin->magic != FATTEXT_MAGIC) {
        return;
    }
    bool r = context->registerFunction(fatCubin->fatbin, hostFun, deviceFun, deviceName);
    if (!r) {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Failed to register function %p with device function %s at address %p in fatbinary handler %p.", hostFun, deviceName, deviceFun, fatCubinHandle);
    }
    else {
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Successfully registered function %p with device function %s at address %p in fatbinary handler %p.", hostFun, deviceName, deviceFun, fatCubinHandle);
    }
}    

void __cudaUnregisterFatBinary_cpp(void **fatCubinHandle)
{
    // It's ok to do nothing here as all context and resources will be released when the context is destroyed. 
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Unregistering fat binary handler %p.", fatCubinHandle);
}

// Kernel launch configuration
// TODO: Implement the actual configuration logic
unsigned __cudaPushCallConfiguration_cpp(dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    struct CUstream_st *stream) 
{ 
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Pushing call configuration with gridDim (%d, %d, %d), blockDim (%d, %d, %d), sharedMem %zu, stream %p", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream);
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Grid address: %p, Block address: %p", (void*)&gridDim, (void*)&blockDim);
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) 
    {
        LOG(LOG_LEVEL_ERROR, "ERROR", "__cudaPushCallConfiguration failed: context is null or invalid.");
        return cudaErrorLaunchFailure;
    }
    context->pushLaunchConfig(gridDim, blockDim, sharedMem, stream);
    return 0;
}

unsigned __cudaPopCallConfiguration_cpp(void* gridDim, void* blockDim, void* sharedMem, void* stream) {
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Popping call configuration with params %p, %p, %p, %p", gridDim, blockDim, sharedMem, stream);
    dim3* gridDimPtr = static_cast<dim3*>(gridDim);
    dim3* blockDimPtr = static_cast<dim3*>(blockDim);
    size_t* sharedMemPtr = static_cast<size_t*>(sharedMem);
    CUstream_st** streamPtr = static_cast<CUstream_st**>(stream);
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) 
    {
        LOG(LOG_LEVEL_ERROR, "ERROR", "__cudaPopCallConfiguration failed: context is null or invalid.");
        return cudaErrorLaunchFailure;
    }
    launchConfiguration config = context->popLaunchConfig();
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "GridDim set to (%d, %d, %d), BlockDim set to (%d, %d, %d)", config.blockDim.x, config.blockDim.y, config.blockDim.z, config.gridDim.x, config.gridDim.y, config.gridDim.z);

    gridDimPtr->x = config.gridDim.x;
    gridDimPtr->y = config.gridDim.y;
    gridDimPtr->z = config.gridDim.z;
    blockDimPtr->x = config.blockDim.x;
    blockDimPtr->y = config.blockDim.y;
    blockDimPtr->z = config.blockDim.z;
    *sharedMemPtr = config.sharedMemBytes;
    *streamPtr = config.hStream;
    return 0;
}

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
    LOG(LOG_LEVEL_DEBUG, "INFO", "Setting CUDA device %d", device);
    CUresult r;
    // get device
    CUdevice cuDevice;
    r = cuDeviceGet_cpp(&cuDevice, device);
    if (r != CUDA_SUCCESS) {
        return cudaErrorInvalidDevice;
    }
    // create context
    CUcontext pctx;
    // r = cuCtxCreate_cpp(&ctx, 0, cuDevice);
    // if (r != CUDA_SUCCESS) {
    //     return cudaErrorUnknown;
    // }
    // set device primary context
    // cuDevicePrimaryCtxRetain will push the context onto the context stack
    r = cuDevicePrimaryCtxRetain_cpp(&pctx, cuDevice);
    if (r != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;  // return success for the placeholder implementation
}

cudaError_t CUDARTAPI cudaLaunchKernel_cpp(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream)
{
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Launching kernel: %p, gridDim: (%d, %d, %d), blockDim: (%d, %d, %d), args: %p, sharedMem: %zu, stream: %p", func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, args, sharedMem, stream);
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Grid address: %p, Block address: %p", (void*)&gridDim, (void*)&blockDim);
    auto context = driver::contextStack.top();
    if (context == nullptr || !(context->valid())) return cudaErrorLaunchFailure;
    auto kernel = context->getKernel(func);
    if (kernel == nullptr) return cudaErrorInvalidDeviceFunction;
    // Launch the kernel
    CUresult r = cuLaunchKernel_cpp(kernel, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream, args, nullptr);
    if (r != CUDA_SUCCESS) return cudaErrorLaunchOutOfResources;
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMalloc_cpp(void **devPtr, size_t size)
{
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Malloc called with size: %zu", size);
    CUdeviceptr dptr;
    auto r = cuMemAlloc_cpp(&dptr, size);
    if (r == CUDA_SUCCESS) {
        *devPtr = reinterpret_cast<void*>(dptr);
        LOG(LOG_LEVEL_DEBUG, "DEBUG", "Malloc successful, allocated %zu bytes at address %p", size, *devPtr);
    }
    else {
        LOG(LOG_LEVEL_ERROR, "ERROR", "Malloc failed with error code %d", r);
    }
    return __driverErrorToRuntime(r);
}

cudaError_t CUDARTAPI cudaMemcpy_cpp(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Copy %zu bytes from %p to %p with kind %d", count, src, dst, kind);
    switch (kind) {
        case cudaMemcpyHostToDevice:
            return cudaMemcpyHtoD_cpp(dst, src, count);
        case cudaMemcpyDeviceToHost:
            return cudaMemcpyDtoH_cpp(dst, src, count);
        default:
            return cudaErrorInvalidValue;
    }
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyHtoD_cpp(void *dst, const void *src, size_t count)
{
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(dst);
    auto r = cuMemcpyHtoD_cpp(dptr, src, count);
    return __driverErrorToRuntime(r);
}

cudaError_t CUDARTAPI cudaMemcpyDtoD_cpp(void *dst, const void *src, size_t count)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaMemcpyDtoH_cpp(void *dst, const void *src, size_t count)
{
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(src);
    auto r = cuMemcpyDtoH_cpp(dst, dptr, count);
    return __driverErrorToRuntime(r);
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
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(devPtr);
    CUresult r = cuMemsetD32_cpp(dptr, static_cast<unsigned int>(value), count / sizeof(unsigned int) + 1);
    return __driverErrorToRuntime(r);
}

cudaError_t CUDARTAPI cudaFree_cpp(void *devPtr)
{
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Freeing memory at %p", devPtr);
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(devPtr);
    CUresult r = cuMemFree_cpp(dptr);
    LOG(LOG_LEVEL_DEBUG, "DEBUG", "Memory freed at %p", devPtr);
    return __driverErrorToRuntime(r);
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

cudaError_t CUDARTAPI cudaArrayGetInfo_cpp(struct cudaChannelFormatDesc *desc, struct cudaExtent *extent, unsigned int *flags, cudaArray_t array)
{
    return cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaArrayGetSparseProperties_cpp(struct cudaArraySparseProperties *sparseProperties, cudaArray_t array)
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