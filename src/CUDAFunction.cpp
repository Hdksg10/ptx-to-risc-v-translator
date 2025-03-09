#include <CUDAFunction.h>

using namespace driver;

CUDAFunction::CUDAFunction(const CUDAFunction& other) : kernel(other.kernel) {
    function.outerKernel = this; // Ensure the outerKernel pointer points to this instance.
}

CUDAFunction::CUDAFunction(ir::PTXKernel* kernel) : kernel(kernel) {
    function.outerKernel = this; // Set the outerKernel pointer to this instance. This is always done.
}

ir::PTXKernel* CUDAFunction::getKernel() {
    return kernel;
}

CUfunction CUDAFunction::getFunctionPointer() {
    return &function;
}