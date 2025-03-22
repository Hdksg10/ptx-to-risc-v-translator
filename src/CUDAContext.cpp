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
    // Assuming a function cuCtxCreate exists in CUDA driver API
    // cuCtxCreate(&context, flags, device);
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
        CUfunction* f = nullptr;
        cuModuleGetFunction_cpp(f, module, name);
        kernels[hostFunc] = *f;
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
    CUdeviceptr ptr = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ptr_h));
    if (ptr) {
        allocations[ptr] = size;
        *dptr = ptr;
        return true;
    }
    else {
        return false;
    }
}

bool CUDAContext::free(CUdeviceptr dptr) {
    if (allocations.find(dptr) != allocations.end()) {
        std::free(reinterpret_cast<void*>(static_cast<uintptr_t>(dptr)));
        allocations.erase(dptr);
        return true;
    }
    return false;
}

size_t CUDAContext::getAllocatedSize(CUdeviceptr ptr) const {
    if (allocations.find(ptr) != allocations.end()) {
        return allocations.at(ptr);
    }
    else {
        return 0;
    }
}

bool CUDAContext::valid() const {
    return context.valid && !context.destroyed; // Check if the context is valid and not destroyed
}