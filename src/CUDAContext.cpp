#include "cuda.h"
#include <Interface.h>
#include <CUDAContext.h>
#include <log.h>
using namespace driver;

std::stack<CUDAContext*> driver::contextStack; // Initialize the stack to manage contexts

CUDAContext::CUDAContext() {
    context.launch_id = 0;
    context.minor = 0;
    context.flags = 0;
    context.destroyed = 0;
    context.outerContext = this;
    
    context.valid = 0;
}

CUDAContext::~CUDAContext() {
    // destroy the context if it hasn't been destroyed yet by cuCtxDestroy
    if (!context.destroyed && context.valid) {
        destroy();
    }
}

void CUDAContext::create(CUdevice device, unsigned int flags) {
    context.flags = flags;
    context.valid = 1;
    context.device = device; // Assign the device id to the context
}

void CUDAContext::destroy() {
    if (context.valid) {
        context.valid = 0;
        context.destroyed = 1;
    }
}

CUcontext CUDAContext::getContext() {
    return (&context);
}

void CUDAContext::setCtx(CUcontext ctx) {
    context = *static_cast<CUctx_st*>(ctx);

}
bool CUDAContext::registerFatbinary(void* fatbinary) {
    CUmodule module;
    if (cuModuleLoadData_cpp(&module, fatbinary) == CUDA_SUCCESS) {
        fatbins[fatbinary] = module;
        return true;
    }
    return false;
}

bool CUDAContext::registerFunction(void* fatbinary, const void * hostFunc, char* deviceFunc, const char* name) {
    auto it = fatbins.find(fatbinary);
    if (it != fatbins.end()) {
        // Fatbinary already loaded, get the module
        CUmodule module = it->second;
        CUfunction f;
        CUresult r = cuModuleGetFunction_cpp(&f, module, name);
        if (r != CUDA_SUCCESS) {
            LOG(LOG_LEVEL_ERROR, "ERROR", "Failed to get function %s from module %p, error code %d", name, module, r);
            return false;
        }
        kernels[hostFunc] = f;
        return true; 
    }
    else {
        return false;
    }
}

CUfunction CUDAContext::getKernel(const void * hostFunc){
    auto it = kernels.find(hostFunc);
    if (it == kernels.end()) {
        return nullptr;
    }
    else {
        return it->second;
    }
}

bool CUDAContext::allocate(CUdeviceptr* dptr, size_t size) {
    void * ptr_h = malloc(size);
    CUdeviceptr ptr = (reinterpret_cast<uintptr_t>(ptr_h));
    if (ptr) {
        allocations[ptr_h] = size;
        *dptr = ptr;
        return true;
    }
    else {
        return false;
    }
}

bool CUDAContext::free(CUdeviceptr dptr) {
    for (auto it = allocations.begin(); it != allocations.end(); ++it) {
        if (reinterpret_cast<uintptr_t>(it->first) == dptr) {  
            void* ptr_h = it->first;
            LOG(LOG_LEVEL_DEBUG, "DEBUG", "Found allocation for %#llx of size %zu bytes", dptr, it->second);
            // std::free(ptr_h);
            allocations.erase(it);
            return true;
        }
    }
    return false;
}

size_t CUDAContext::getAllocatedSize(CUdeviceptr ptr) const {
    void* ptr_h = reinterpret_cast<void*>((ptr));
    if (allocations.find(ptr_h) != allocations.end()) {
        return allocations.at(ptr_h);
    }
    else {
        return 0;
    }
}

bool CUDAContext::valid() const {
    return context.valid && !context.destroyed; 
}

void CUDAContext::pushLaunchConfig(const dim3& gridDim, const dim3& blockDim, size_t sharedMemBytes, CUstream_st* hStream) {
    launchConfiguration config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.sharedMemBytes = sharedMemBytes;
    config.hStream = hStream;
    launchConfigurations.push(config);
}

launchConfiguration CUDAContext::popLaunchConfig() {
    if (!launchConfigurations.empty()) {
        launchConfiguration config = launchConfigurations.top();
        launchConfigurations.pop();
        return config;
    }
    else {
        // Handle empty stack case, possibly throw an exception or return a default configuration
        return launchConfiguration();
    }
}