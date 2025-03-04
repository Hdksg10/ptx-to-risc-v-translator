#include <cuda_runtime.h>

#include <Interface.h>

extern "C" cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    return cudaDeviceCanAccessPeer_cpp(canAccessPeer, device, peerDevice);
} 