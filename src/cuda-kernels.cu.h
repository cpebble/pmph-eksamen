// This contains the parallel kernels
#ifndef _CUDA_KERNELS
#define _CUDA_KERNELS

/// HELPERS
// Yanked from w3 t3
template <class ElTp, int T> 
__global__ void matTransposeTiledKer(ElTp* A, ElTp* B, int heightA, int widthA) {
    extern __shared__ char sh_mem1[];
    volatile ElTp* tile = (volatile ElTp*)sh_mem1;
    //__shared__ float tile[T][T+1];

    int x = blockIdx.x * T + threadIdx.x;
    int y = blockIdx.y * T + threadIdx.y;

    if( x < widthA && y < heightA )
        tile[threadIdx.y*(T+1) + threadIdx.x] = A[y*widthA + x];

    __syncthreads();

    x = blockIdx.y * T + threadIdx.x; 
    y = blockIdx.x * T + threadIdx.y;

    if( x < heightA && y < widthA )
        B[y*heightA + x] = tile[threadIdx.x*(T+1) + threadIdx.y];
}

// Kernels
// --- Kernel 1 ---
// Assumes 2d block dimensions
__global__ void gpu_mkX(int K, int N, float f, int* mappingIndices, float* X_out){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check
    if (i >= K || j >= N)
        // Just quit
        return;

    if (i == 0)
        X_out[I2(i,j,N)] = 1;
    else if (i == 1)
        X_out[I2(i,j,N)] = (float)mappingIndices[j];
    else{
        float i_ = (float)(i / 2);
        float j_ = ((float)mappingIndices[j]);
        float F = (2.0f*M_PI*i_*j_) / f;
        X_out[I2(i,j,N)] = (i % 2 == 0) ? sinf(F) : cosf(F);
    }
}

int catchthis(int n){

    return n*n;
}
#endif
