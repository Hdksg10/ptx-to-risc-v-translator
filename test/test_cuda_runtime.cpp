#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cuda.h>
using namespace std;


#define CUDA_CHECK(f, msg) \
	if ((r = f) != CUDA_SUCCESS) { \
		status << msg << r;  \
		return false;            \
	}   



bool doTest(std::stringstream& status)
{
    CUresult r;
    // Initialize the CUDA driver API
    CUDA_CHECK(cuInit(0), "Failed to initialize CUDA driver API: ");
    // Get the number of CUDA-capable devices
    int deviceCount;
    CUDA_CHECK(cuDeviceGetCount(&deviceCount), "Failed to get device count: ");
    if (deviceCount == 0) {
        status << "No CUDA-capable devices found.";
        return false;
    }
    // Create a context for the first device
    CUdevice device;
    CUDA_CHECK(cuDeviceGet(&device, 0), "Failed to get device: ");
    CUcontext context;
    CUDA_CHECK(cuCtxCreate(&context, 0, device), "Failed to create context: ");
    // Set the current context
    CUDA_CHECK(cuCtxSetCurrent(context), "Failed to set current context: ");

    // Load fatbin file into bytes array
    std::ifstream fatbinFile("test_runtime.fatbin", std::ios::binary);
    if (!fatbinFile) {
        std::cerr << "Failed to open file!" << std::endl;
        return false;
    }
    std::vector<char> fatbinBytes((std::istreambuf_iterator<char>(fatbinFile)), std::istreambuf_iterator<char>());
    fatbinFile.close();
    // try print first 4 bytes of fatbinBytes (magic number)
    uint32_t magicNumer = *reinterpret_cast<uint32_t *>(fatbinBytes.data());
    std::cout << "Magic number: 0x" << std::hex << magicNumer << std::endl;
    // Load the module from the fatbin bytes
    CUmodule module;
    CUDA_CHECK(cuModuleLoadData(&module, fatbinBytes.data()), "Failed to load module from fatbin data: ");

    std::string kernelName = "_Z14EwiseAddKernelPKfS0_Pfm";

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, kernelName.c_str()), "Failed to get function: ");
    if (kernel == nullptr) {
        status << "Failed to retrieve kernel: " << kernelName << std::endl;
        return false;
    }

    // Allocate memory for the kernel argument
    CUdeviceptr d_arg;
    CUDA_CHECK(cuMemAlloc(&d_arg, sizeof(float)), "Failed to allocate memory for kernel argument: ");

    CUdeviceptr a_arg;
    CUDA_CHECK(cuMemAlloc(&a_arg, sizeof(float)), "Failed to allocate memory for kernel argument: ");
    
    CUdeviceptr b_arg;
    CUDA_CHECK(cuMemAlloc(&b_arg, sizeof(float)), "Failed to allocate memory for kernel argument: ");

    float h_a = 3.14;
    float h_b = 2.13;

    CUDA_CHECK(cuMemcpyHtoD(a_arg, &h_a, sizeof(float)), "Failed to copy memory from host to device: ");
    std::cout << a_arg << std::endl;

    CUDA_CHECK(cuMemcpyHtoD(b_arg, &h_b, sizeof(float)), "Failed to copy memory from host to device: ");

    int N = 1;

    std::vector<void*> kernelParams;
    kernelParams.push_back(&a_arg);
    kernelParams.push_back(&b_arg);
    kernelParams.push_back(&d_arg);
    kernelParams.push_back(&N);
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, kernelParams.data(), 0), "Failed to launch kernel: ");

    // Copy result back to host
    float h_arg;
    CUDA_CHECK(cuMemcpyDtoH(&h_arg, d_arg, sizeof(float)), "Failed to copy memory from device to host: ");

    std::cout << h_arg << std::endl;

    // Clean up
    CUDA_CHECK(cuCtxDestroy(context), "Failed to destroy context: ");

    return true;
}

int main() {
    std::stringstream status;
    if (!doTest(status)) {
        std::cerr << "Test failed: " << status.str() << std::endl;
        return 1;
    }
    return 0;
}