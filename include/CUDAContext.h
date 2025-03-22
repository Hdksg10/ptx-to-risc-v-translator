/*
*/
#ifndef CUDA_CONTEXT_H
#define CUDA_CONTEXT_H

#include "CUDAFunction.h"
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
        bool registerFatbinary(void* fatbinary);
        bool registerFunction(void* fatbinary, const void * hostFunc, char* deviceFunc, const char* name);
        // get registered kernels
        CUfunction getKernel(const void * hostFunc);
    private:
        CUctx_st context;
        // Allocated device memory pointers and their sizes
        std::unordered_map<CUdeviceptr, size_t> allocations;
        // Registered kernels
        std::unordered_map<const void*, CUfunction> kernels;
        // Registered fatbinary modules
        std::unordered_map<void*, CUmodule> fatbins; 
    };

    extern std::stack<CUDAContext*> contextStack; // Stack to manage contexts
} // namespace driver

#endif