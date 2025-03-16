#ifndef CUDAFUNCTION_H
#define CUDAFUNCTION_H

#include <cuda.h>
#include <ocelot/ir/Module.h>
#include <ocelot/ir/PTXKernel.h>

namespace driver {
    class CUDAFunction;
    class CUDAModule;
}

struct CUfunc_st {
    void* outerKernel; // pointer to outer CUDAFunction instance (always!)
};

namespace driver {
    class CUDAFunction {
    public:
        CUDAFunction(const CUDAFunction& other);
        CUDAFunction(const std::string& name, CUDAModule* module);
        // CUDAFunction(ir::PTXKernel* kernel);
        CUfunction getFunctionPointer();
        std::string getName() const;
        CUDAModule* getModule() const;
        // ir::PTXKernel* getKernel();
    private:
        struct CUfunc_st function;
    private:
        std::string name;
        CUDAModule* module;
    };
} // namespace driver

#endif // CUDAFUNCTION_H