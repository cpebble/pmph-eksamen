// This contains the parallel kernels
#ifndef _CUDA_KERNELS
#define _CUDA_KERNELS
#define lgWARP 5
#define WARP ( 1<<lgWARP )

/// HELPERS

__global__ void sliceMatrix(float* X_in, float* X_out, int r_slice, int c_slice, int width){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r_slice && j < c_slice)
        X_out[I2(i, j, c_slice)] = X_in[I2(i, j, width)];
}
//
//
// Yanked from w3 t3
template <int T> 
__global__ void matTransposeTiledKer(float* A, float* B, int heightA, int widthA) {
    extern __shared__ char sh_mem1[];
    volatile float* tile = (volatile float*)sh_mem1;
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
// Yanked from w2
template <class Tp>
__device__ inline Tp scanIncWarp( volatile Tp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for(int d = 0; d < lgWARP; d++){
        int h = 1 << d;
        if (lane >= h){
            ptr[idx] = (ptr[idx-h] + ptr[idx]);
        }
    }
    return (ptr[idx]);
}

template <class Tp> 
__device__ inline Tp
scanIncBlock(volatile Tp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;


    // 1. perform scan at warp level
    Tp res = scanIncWarp<Tp>(ptr,idx);
    __syncthreads();

    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (WARP-1) && warpid < 31) { 
        ptr[warpid] = (Tp)(ptr[idx]);
    }
    __syncthreads();
    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<Tp>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = (ptr[warpid-1] + res);
    }
    return res;
}

__device__ int filterPadWithKeys(float* arr, float* Rs, int* Ks, int n){
    // Can't handle more than 1024 "n" values. 
    // To fix this we need to allocate shared memory dynamically, which is :'(
    __shared__ int tfs[512];
    __shared__ int isT[512];
    __shared__ int inds[512];
    // Get this into local thread so we can access it uncoalesced
    //
    int i = threadIdx.x;
    if (i >= n)
        return -1;
    __shared__ float arr_shr[512];
    arr_shr[i] = arr[i];
    __syncthreads();
    tfs[i] = !(isnan(arr_shr[i]));
    __syncthreads();
    isT[i] = scanIncBlock<int>(tfs, i);
    __syncthreads();
    int i_ = isT[n-1];
    inds[i] = (!isnan(arr_shr[i])) ? isT[i]-1 : -1;
    __syncthreads();
    // Now scatter using calculated values
    int c = inds[i];
    Ks[i] = 0;
    Rs[i] = NAN;
    if(c != -1){
        Ks[c] = i;
        Rs[c] = arr_shr[i];
    }
    return i_;
}


// Kernels
// --- Kernel 1 ---
// Assumes 2d block dimensions
__global__ void gpu_mkX(int K, int N, float f, int* mappingIndices, float* X_out){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check
    if (i < K && j < N){

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
}
// --- Kernel 2 ---
// Naive version
__global__ void gpu_mmMulFilt_naive(float* X, float* X_t, float* y,
        float* M, int pixels, int n, int p, int m){
    int pix = blockIdx.z * blockDim.z + threadIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if ( pix < pixels && i < n && j < m){
        float accum = 0.0f;
        for (int k = 0; k < p; k++){
            float a = X[I2(i,k,p)];
            float b = X_t[I2(k,j,m)];
            if (!isnan(y[I2(pix, k, p)]))
                accum += a*b;
        }
        M[I3(pix, i, j, n, m)] = accum;
    }
}

// --- Kernel 3 ---
// Naive mat-inversion
template <int K> 
__global__ void gpu_batchMatInv(float* A, float* A_inv, int m){
    int pix = blockIdx.x;
    // We only use threadIdx here since we get a block size of 16*8
    int k1 = threadIdx.y;
    int k2 = threadIdx.x;
    // Fill shared array
    __shared__ float Ash[K*2*K];
    Ash[I2(k1, k2, 2*K)] = (k2 < K) ? A[I3(pix, k1, k2, K, K)] : (k2 == K + k1);
    __syncthreads();

    // This is the one from the futhark code
    for(int i = 0; i < K; i++){
        // K, j = row, col = k1, k2
        float v1 = Ash[i];
        float tmp = 0.0f;
        if (v1 == 0.0){
            tmp = Ash[I2(k1,k2,K*2)];
        } else {
            float x = (Ash[k2] / v1);
            if( k1 < K-1 ){
                tmp = Ash[I2(k1+1,k2,2*K)] - Ash[I2(k1+1, i, 2*K)] * x;
            }
            else{
                tmp = x;
            }
        }
        __syncthreads();
        Ash[I2(k1, k2, 2*K)] = tmp;
    }
    if (k2 < K)
        A_inv[I3(pix, k1, k2, K, K)] = Ash[I2(k1, k2 + K, 2*K)];

}
// The simple unoptimized from cpu
//
template <int K> 
__global__ void gpu_batchMatInv_naive(float* A, float* A_inv, int m){
    int pix = blockIdx.x;
    // We only use threadIdx here since we get a block size of 16*8
    int k1 = threadIdx.y;
    int k2 = threadIdx.x;
    int width = 2*K;
    // Fill shared array
    __shared__ float Ash[K*2*K];
    Ash[I2(k1, k2, 2*K)] = (k2 < K) ? A[I3(pix, k1, k2, K, K)] : (k2 == K + k1);
    __syncthreads();
    for(int i = 0; i < K; i++){
        float curEl = Ash[I2(i, i, 2*K)];
        float tmp = 0.0f;
        if (Ash[I2(i, i, width)] == 0.0)
            continue;
        
        if (i != k1){
            float ratio = Ash[I2(k1, i, width)] / Ash[I2(i, i, width)];
            tmp = Ash[I2(k1, k2, width)] - ratio*Ash[I2(i, k2, width)];
        }
        __syncthreads();
        if (i != k1)
            Ash[I2(k1, k2, width)] = tmp;
        Ash[I2(i, k2, width)] /= curEl;
        __syncthreads();
    }

    if (k2 < K)
        A_inv[I3(pix, k1, k2, K, K)] = Ash[I2(k1, k2 + K, 2*K)];
    
}
//
// Batched mat-inversion
// Requires a grid of ds->m blocks, with y = K, and x = 2K
template <int K> 
__global__ void gpu_batchMatInv_old(float* A, float* A_inv, int m){
    int pix = blockIdx.x;
    // We only use threadIdx here since we get a block size of 16*8
    int k1 = threadIdx.y;
    int k2 = threadIdx.x;
    // Fill shared array
    __shared__ float Ash[K*2*K];
    Ash[I2(k1, k2, 2*K)] = (k2 < K) ? A[I3(pix, k1, k2, K, K)] : (k2 == K + k1);
    __syncthreads();
    // Do the transformation / elimination
    for (int q = 0; q < K; q++){
        float vq = Ash[I2(0,q,2*K)];
        float tmp = 0.0f;
        if (fabs(vq) <= 0.001f)
            tmp = Ash[I2(k1, k2, 2*K)];
        else{
            float x = Ash[I2(0, k2, 2*K)] / vq;
            if (k1 == K-1){
                tmp = x;
            } else{
                tmp = Ash[I2(k1 + 1, k2, 2*K)] - Ash[I2(k1 + 1, q, 2*K)] * x;
            }
        }
        __syncthreads();
        Ash[I2(k1, k2, 2*K)] = tmp;
        __syncthreads();

    }
    // Copy back to global memory
    if (k2 < K)
        A_inv[I3(pix, k1, k2, K, K)] = Ash[I2(k1, k2 + K, 2*K)];
    
}
// --- Kernel 4
// Matrix-Vector multiplication
__global__ void gpu_mvMulFilt(float* X, float* y, float* y_out, int pixels, int height, int width){
    //int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    if ( i < pixels && j < height ){
        float accum = 0.0f;
        for(int k = 0; k < width; k++){
            float a = X[I2(j, k, width)];
            float b = y[I2(i, k, width)];
            if (!isnan(b))
                accum += a*b;
        }
        y_out[I2(i, j, height)] = accum;
    }
}
__global__ void gpu_mvMul(float* X, float* y, float* y_out, int pixels, int height, int width){
    int i = blockIdx.x;
    int j = threadIdx.x;
    if ( i < pixels && j < height ){
        float accum = 0.0f;
        for(int k = 0; k < width; k++){
            float a = X[I3(i, j, k, width, width)];
            float b = y[I2(i, k, width)];
            accum += a*b;
        }
        y_out[I2(i, j, height)] = accum;
    }
}

// --- Kernel 5
// Y Error Prediction
// Requires blockDim.x  = N
// Requires numBlocks.x = m
__global__ void gpu_YErrorCalculation(float* Y, float* Ypred, float* R, int* K, int* Ns, int m, int N){
    float y_err_tmp[512];
    int pix = blockIdx.x;
    int i = threadIdx.x;
    float ye = Y[I2(pix, i, N)];
    float yep= Ypred[I2(pix, i, N)];
    y_err_tmp[i] = (isnan(ye)) ? ye : ye-yep;
    __syncthreads();
    int n = filterPadWithKeys(y_err_tmp, &R[I2(pix, 0, N)], &K[I2(pix, 0, N)], N);
    if (threadIdx.x == 0){
        Ns[pix] = n;
    }
}

// --- Kernel 6
// Inner parallel size is n, for each m pixels. We do map -> scan, map -> scan
// to avoid writing a warp optimized reduce
__global__ void gpu_NSSigma(float* Y_errors, float* Y_historic, 
         float* sigmas, int* hs, int* nss, int N, int n, int m, int k, float hfrac){
    int pix = blockIdx.x;
    int t   = threadIdx.x;
    // Calculate ns
    __shared__ int nss_tmp[512];
    __shared__ int nss_tmp_scanned[512];

    // it holds t < n since we set block size explicit
    if(t >= n){
        nss_tmp[t] = 0;
    } else {
        nss_tmp[t] = 1 - (isnan(Y_historic[I2(pix, t, n)]));
    }
    nss_tmp_scanned[t] = scanIncBlock<int>(nss_tmp, t);
    __syncthreads();
    int ns = nss_tmp_scanned[n - 1];
    // Now calc sigma
    __shared__ float y_error_filtered[512];
    y_error_filtered[t] = (t < ns) ? Y_errors[I2(pix, t, N)] : 0.0f;
    y_error_filtered[t] = y_error_filtered[t] * y_error_filtered[t];
    __shared__ float y_error_scanned[512];
    y_error_scanned[t] = scanIncBlock<float>(y_error_filtered, t);
    __syncthreads();
    // We really only need one thread to calculate and return values
    if (t == 0){
        float sigma = y_error_scanned[ns-1];
        //printf("Got sigma %f\n", sigma);
        sigmas[pix] = sqrtf(sigma / (float)(ns - k));
        hs[pix] = (int) ( ((float) ns) * hfrac );
        nss[pix] = ns;
    }
}

// --- Kernel 7
__global__ void gpu_msFst(float* y_error, int* ns, int* hs, float* MO_fsts, int m, int N, int hMax){
    int pix = blockIdx.x;
    int i = threadIdx.x;
    __shared__ float shr_arr[512];
    if (i < hs[pix]){
        shr_arr[i] = y_error[I2(pix, i+ns[pix]-hs[pix]+1, N)];   
        // Dirty arrayless reduce
        float tmp = scanIncBlock<float>(shr_arr, i);
        if (i == hs[pix] - 1){
            MO_fsts[pix] = tmp;
        }
    }
}
__global__ void gpu_calcBound(int* mappingIndices, float* BOUND, float lam, int N, int n, int mIm1){
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N - n)
        return;
    int t = q + n;
    int time = mappingIndices[t];
    float tmp = logplus_dev( ((float)time) / ((float) mIm1));
    BOUND[q] = lam * (sqrtf(tmp));
}

// --- Kernel 8
// Outer parall size: m
// Inner parallel size: N
__global__ void gpu_mosum(int* Nss, int* nss, float* sigmas, int* hs, float* MO_fsts,
        float* y_errors, int* val_inds, float* BOUND, int* breaks, int N, int n, int m){
    // Calc MO
    int pix = blockIdx.x;
    int j = threadIdx.x;
    __shared__ float MO[512];
    __shared__ float MO_[512];
    if(j < N - n){
        if (j >= Nss[pix] - nss[pix]){
            MO[j] = 0.0f;
        } else if( j == 0){
            MO[j] = MO_fsts[pix];
        } else {
            MO[j] = (-y_errors[I2(pix, nss[pix]-hs[pix]+j, N)] +
                     y_errors[I2(pix, nss[pix]+j, N)]);
        }
        __syncthreads();
        MO_[j] = scanIncBlock<float>(MO, j);
        __syncthreads();
        MO_[j] = MO_[j] / (sigmas[pix] * (sqrtf((float)nss[pix])));
        // Now with Mo calculated, just get us a list of breaks
    }
    // Could be done in parallel with reduce, but this is way simpler implemented
    if (j == 0){
        breaks[pix] = -1;
        for(int i = 0; i < N - n; i++){
            if (i < (Nss[pix] - nss[pix]) && !(isnan(MO_[i]))){
                if(fabs(MO_[i]) > BOUND[i]){
                    breaks[pix] = val_inds[i];
                    break;
                }
            }
        }
    }

}
// Dummy func to ease compiler errors
int catchthis(int n){

    return n*n;
}
#endif
