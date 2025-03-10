/*
*/
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include <cuda.h>
#include <stack>
#include <unordered_map>
#include <cstdlib>

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
        void* outerContext; // pointer to outer CUDAContext instance (always!)
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
        bool allocate(CUdeviceptr* dptr, size_t size); // Allocate device memory and return a pointer to it
        bool free(CUdeviceptr dptr); // Free device memory
        size_t getAllocatedSize(CUdeviceptr ptr) const; // Get the size of an allocated device memory block
        bool valid() const; // Check if the context is valid
    private:
        CUctx_st context;
        std::unordered_map<CUdeviceptr, size_t> allocations; // Map to track device memory allocations and their sizes
    };

    extern std::stack<CUDAContext*> contextStack; // Stack to manage contexts
} // namespace driver

#endif