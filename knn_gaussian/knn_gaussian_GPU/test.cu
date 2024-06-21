//test cuda device
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.h>

__global__ void testKernel()
{
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }

    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        if (cudaGetLastError() != cudaSuccess)
        {
            std::cerr << "Error getting device properties for device " << device << std::endl;
            return 1;
        }

        if (deviceProp.major >= 1)
        {
            break;
        }
    }

    if (device == deviceCount)
    {
        std::cerr << "No CUDA devices with at least compute capability 1.0 found" << std::endl;
        return 1;
    }

    cudaSetDevice(device);
    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Error setting CUDA device" << std::endl;
        return 1;
    }

    testKernel<<<2, 4>>>();
    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Error launching kernel" << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess)
    {
        std::cerr << "Error synchronizing after kernel launch" << std::endl;
        return 1;
    }

    return 0;
}