#include <CUDAContext.h>

using namespace driver;

std::stack<CUDAContext*> driver::contextStack; // Initialize the stack to manage contexts

CUDAContext::CUDAContext() {
    context.launch_id = 0;
    context.minor = 0;
    context.flags = 0;
    context.destroyed = 0;
    context.outerContex = this;
    
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