#ifndef CUDAFUNCTION_H
#define CUDAFUNCTION_H
#include <cuda.h>

#include <ocelot/ir/Module.h>
#include <ocelot/ir/PTXKernel.h>

namespace driver {
    class CUDAFunction;
}

struct CUfunc_st {
    void* outerKernel; // pointer to outer CUDAFunction instance (always!)
};

namespace driver {
    class CUDAFunction {
    public:
        CUDAFunction(ir::PTXKernel* kernel);
        CUfunction getFunctionPointer();
        ir::PTXKernel* getKernel();
    private:
        struct CUfunc_st function;
    private:
        ir::PTXKernel* kernel;
    };
} // namespace driver

#endif // CUDAFUNCTION_H