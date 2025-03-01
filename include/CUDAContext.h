/*
*/
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include <cuda.h>

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
        void push();
        void pop();
        void setCurrent();
        void setCurrent(CUcontext context);
        CUcontext getContext();
    private:
        CUctx_st context;
    };
} // namespace driver

#endif