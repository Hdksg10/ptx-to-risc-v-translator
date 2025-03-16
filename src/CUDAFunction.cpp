#include <CUDAFunction.h>

using namespace driver;

CUDAFunction::CUDAFunction(const CUDAFunction& other) : name(other.name), module(other.module) {
    function.outerKernel = this; // Ensure the outerKernel pointer points to this instance.
}
CUDAFunction::CUDAFunction(const std::string& _name, CUDAModule* _module) : name(_name), module(_module) {
    function.outerKernel = this; // Ensure the outerKernel pointer points to this instance.
}

// CUDAFunction::CUDAFunction(ir::PTXKernel* kernel) : kernel(kernel) {
//     function.outerKernel = this; // Set the outerKernel pointer to this instance. This is always done.
// }

// ir::PTXKernel* CUDAFunction::getKernel() {
//     return kernel;
// }

CUfunction CUDAFunction::getFunctionPointer() {
    return &function;
}

std::string CUDAFunction::getName() const {
    return name;
}

CUDAModule* CUDAFunction::getModule() const {
    return module;
}