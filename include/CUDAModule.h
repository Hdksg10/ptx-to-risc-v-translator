#ifndef CUDAMODULE_H
#define CUDAMODULE_H
#include <cuda.h>
#include <string>
#include <unordered_map>
#include <CUDAFunction.h>

#include <ocelot/ir/Module.h>

namespace driver {
    class CUDAModule;
}

extern "C" {
    struct CUmod_st {
        void* module; // Pointer to the internal module representation(CUDA module instance)
    };
}

namespace driver {
    class CUDAModule {
    public:
        bool load(const std::string& path);
        void unload();
        ir::Module* getModule();
        CUfunction getFunction(const char* name); // Retrieve a function by name

    private:
        std::unordered_map<std::string, CUDAFunction> functions; // Map of function names to CUDAFunction objects
        ir::Module module;
    };
} // namespace driver



#endif