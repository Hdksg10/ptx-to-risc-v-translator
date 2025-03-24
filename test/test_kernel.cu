
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#define BASE_THREAD_NUM 8

typedef int scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaDims {
  dim3 block, grid;
};

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) {
      std::cerr << err << std::endl;
      throw std::runtime_error("Failed to allocate memory on GPU");
    }
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

int main() {
  std::cout << "log1" << std::endl;
    // init test array
    const size_t M = 128;
    scalar_t a[M], b[M], out[M];
    for (size_t i = 0; i < M; i++) {
        a[i] = i + 1;
        b[i] = i + 1;
    }
    memset(out, 0, M * ELEM_SIZE);

    // copy to device
    CudaArray device_a(M), device_b(M), device_out(M);
    cudaMemcpy(device_a.ptr, a, M * ELEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b.ptr, b, M * ELEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemset(device_out.ptr, 0, M * ELEM_SIZE);

    // call kernel
    CudaDims dims = CudaOneDim(M);
    EwiseAddKernel<<<dims.grid, dims.block>>>(device_a.ptr, device_b.ptr, device_out.ptr, M);
    
    // copy back to host
    cudaMemcpy(out, device_out.ptr, M * ELEM_SIZE, cudaMemcpyDeviceToHost);

    // print result
    for (size_t i = 0; i < M; i++) {
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}