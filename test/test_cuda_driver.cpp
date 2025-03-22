#include <cuda.h>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <vector>
#define CUDA_CHECK(f, msg) \
	if ((r = f) != CUDA_SUCCESS) { \
		status << msg << r;  \
		return false;            \
	}   

bool test(std::stringstream& status) {
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
    // Try to allocate some memory on the device
    CUdeviceptr d_ptr;
    size_t size = 1024; // Allocate 1024 bytes
    CUDA_CHECK(cuMemAlloc(&d_ptr, size), "Failed to allocate memory on device: ");
    
    // Test Memory Copy
    int h_data[16]; // Host data
    for (int i = 0; i < 16; ++i) {
        h_data[i] = i;
    }
    CUDA_CHECK(cuMemcpyHtoD(d_ptr, h_data, sizeof(h_data)), "Failed to copy data from host to device: ");
    // Allocate host memory for the result
    int h_result[16];
    CUDA_CHECK(cuMemcpyDtoH(h_result, d_ptr, sizeof(h_result)), "Failed to copy data from device to host: ");
    // Verify the data
    // Check address
    std::cout << std::hex << "Device pointer: 0x" << d_ptr << std::endl;
    std::cout << std::hex << "Host pointer 1: " << h_data << std::endl;
    std::cout << std::hex << "Host pointer 2: " << h_result << std::endl;
    for (int i = 0; i < 16; ++i) {
        std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
        if (h_result[i] != h_data[i]) {
            status << "Data mismatch at index " << i << ": expected " << h_data[i] << ", got " << h_result[i] << std::endl; // Use stringstream for error messages
            return false;
        }
    }
    // Free the allocated memory
    CUDA_CHECK(cuMemFree(d_ptr), "Failed to free memory on device: ");
    // Try free again to check for errors
    CUresult error;
    error = cuMemFree(d_ptr);
    if (error != CUDA_SUCCESS) {
        std::cout << "Attempt to free already freed memory failed as expected: " << error << std::endl;
    } else {
        status << "Unexpected success in freeing already freed memory" << std::endl;
        return false;
    }

    // Test Module Management
    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, "test_add.ptx"), "Failed to load module: ");

    std::string kernelName = "_Z14EwiseAddKernelPKfS0_Pfm";

    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, kernelName.c_str()), "Failed to get function: ");
    if (kernel == nullptr) {
        status << "Failed to retrieve kernel: " << kernelName << std::endl;
        return false;
    }

    // Allocate memory for the kernel argument
    CUdeviceptr d_arg;
    CUDA_CHECK(cuMemAlloc(&d_arg, sizeof(uint64_t)), "Failed to allocate memory for kernel argument: ");

    std::vector<void*> kernelParams;
    kernelParams.push_back(&d_arg);
    
    CUDA_CHECK(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, kernelParams.data(), 0), "Failed to launch kernel: ");

    // Copy result back to host
    uint64_t h_arg;
    CUDA_CHECK(cuMemcpyDtoH(&h_arg, d_arg, sizeof(uint64_t)), "Failed to copy memory from device to host: ");

    std::cout << h_arg << std::endl;

    // Clean up
    CUDA_CHECK(cuCtxDestroy(context), "Failed to destroy context: ");
    return true;
}

int main() {
    std::stringstream status;
    if (!test(status)) {
        std::cerr << "Test failed: " << status.str() << std::endl;
        return 1;
    }
    std::cout << "Test passed!" << std::endl;
    return 0;
}