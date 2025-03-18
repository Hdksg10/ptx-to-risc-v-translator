#ifndef CUDAMODULE_H
#define CUDAMODULE_H
#include <cuda.h>
#include <string>
#include <unordered_map>
#include <log.h>
#include <CUDAFunction.h>

#include <fatbin-decompress.h>

#include <ocelot/ir/Module.h>

namespace driver {
    class CUDAModule;
}

extern "C" {
    struct CUmod_st {
        void* module; // Pointer to the internal module representation(CUDA module instance)
        void* context; // Pointer to the context(CUDAContext instance) associated with this module
    };
}

namespace driver {
    class CUDAModule {
    public:
        bool load(const std::string& path);
        // load module from cubin or fatbin as output by nvcc, or a NULL-terminated PTX, either as output by nvcc or hand-written.
        bool load(const void* image);
        void unload();
        ir::Module* getModule();
        CUfunction getFunction(const char* name); // Retrieve a function by name

    private:
        std::unordered_map<std::string, CUDAFunction> functions; // Map of function names to CUDAFunction objects
        ir::Module module;
    };
} // namespace driver



#endif