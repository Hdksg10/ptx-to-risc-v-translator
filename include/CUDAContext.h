/*
*/
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include <cuda.h>
#include <stack>
#include <ocelot/ir/Module.h>

namespace driver{
    class CUDAContext;
}

extern "C"{
    // Implementation details of the CUDA CUcontext structure
    // Use extern "C" to prevent name mangling for compatibility
    // CUctx_st should never be allocated directly in code, since we assume that the outerContex has the longer lifetime
    struct CUctx_st {
        int launch_id;
        int minor;
        unsigned int flags;
        int destroyed;
        int valid;
        CUdevice device; // device id
        void* outerContex; // pointer to outer CUDAContext instance (always!)
    }; // struct CUctx_st
}

namespace driver {
    class CUDAContext {
    public:
        CUDAContext();
        ~CUDAContext();
        void create(CUdevice device, unsigned int flags);
        void destroy();
        void setCtx(CUcontext ctx); // Set the inner context (used in cuCtxSetCurrent)
        CUcontext getContext();
    private:
        CUctx_st context;
    };

    extern std::stack<CUDAContext*> contextStack; // Stack to manage contexts
} // namespace driver

#endif