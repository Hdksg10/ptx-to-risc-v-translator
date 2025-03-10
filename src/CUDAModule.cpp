#include <CUDAModule.h>

using namespace driver;

bool CUDAModule::load(const std::string& path) {
    // Load the module from the given path
    return module.load(path);
}

void CUDAModule::unload() {
    // Unload the module
    module.unload();
}

ir::Module* CUDAModule::getModule() {
    return &module;
}

CUfunction CUDAModule::getFunction(const char* name) {
    // Check if the function is already loaded
    auto it = functions.find(name);
    if (it != functions.end()) {
        return it->second.getFunctionPointer();
    }

    auto func = module.getKernel(name);
    if (func == nullptr) {
        return nullptr;
    }
    CUDAFunction cudaFunc(func);
    functions.emplace(name, CUDAFunction(func)); 
    return functions.at(name).getFunctionPointer();
}