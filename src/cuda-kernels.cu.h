// This contains the parallel kernels
#ifndef _CUDA_KERNELS
#define _CUDA_KERNELS

__global__ void parralel(int n, int* N){
    N[0] = n;
}

#endif
