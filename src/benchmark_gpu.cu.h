// Returns the average break difference

void gpu_run(dataset* ds){
    // KERNEL 1
    // Make interpolation matrix
    int k2p2 = 2*ds->k + 2;
    const int k2p2_ = (ds->trend > 0) ? k2p2 : k2p2-1;
    // Get mappingIndices and images to the GPU
    int* MI_dev; cudaMalloc((void**)&MI_dev, ds->N * sizeof(int));
    float* images_dev; cudaMalloc((void**)&images_dev, ds->m * ds->N * sizeof(float));
    cudaMemcpy(MI_dev, ds->mappingIndices, ds->N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(images_dev, ds->images, ds->m * ds->N*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // Perform sequential validation run

    // Now CudaMalloc and do the same thing on the GPU
    float* X_dev; cudaMalloc((void**)&X_dev, k2p2_*ds->N*sizeof(float));
    // Threads per block(kind of arbitrary for the naive version)
    {
        dim3 threadsPerBlock(8, 8);
        dim3 numBlocks((ds->N / threadsPerBlock.x) + 1, (k2p2_ / threadsPerBlock.y));
        gpu_mkX<<<numBlocks, threadsPerBlock>>>(k2p2_, ds->N, ds->freq, MI_dev, X_dev);
    }

    // Cuda mem
    float* Xh_dev;  cudaMalloc((void**)&Xh_dev,  k2p2_ * ds->n * sizeof(float));
    float* Xth_dev; cudaMalloc((void**)&Xth_dev, k2p2_ * ds->n * sizeof(float));
    float* Yh_dev;  cudaMalloc((void**)&Yh_dev, ds->m * ds->n * sizeof(float));
    // And gpu slicing
    {
        dim3 threadsPerBlock(8,128);
        dim3 numBlocks((ds->n / threadsPerBlock.x) + 1, (k2p2_ / threadsPerBlock.y) + 1);
        sliceMatrix<<<numBlocks, threadsPerBlock>>>(X_dev, Xh_dev, k2p2_, ds->n, ds->N);
        dim3 y_numBlocks((ds->n / threadsPerBlock.x) + 1, (ds->m / threadsPerBlock.y) + 1);
        sliceMatrix<<<y_numBlocks, threadsPerBlock>>>(images_dev, Yh_dev, ds->m, ds->n, ds->N);
    }
    // Transpose X
    {
        const int T = 32; int height = k2p2_; int width = ds->n;
        // 1. setup block and grid parameters
        unsigned int sh_mem_size = T * (T+1) * sizeof(float); 
        int  dimy = (height+T-1) / T; 
        int  dimx = (width +T-1) / T;
        dim3 block(T, T, 1);
        dim3 grid (dimx, dimy, 1);

        matTransposeTiledKer<T><<<grid, block, sh_mem_size>>>
            (Xh_dev, Xth_dev, height, width);
        cudaDeviceSynchronize();
        
    }
    // KERNEL 2
    // Cuda
    float* Xsqr_dev; cudaMalloc((void**)&Xsqr_dev, ds->m * k2p2_ * k2p2_ * sizeof(float));

    {
        dim3 threadsPerBlock(4, 8, 8);
        dim3 numBlocks(
                (k2p2_ / threadsPerBlock.x) + 1,
                (k2p2_ / threadsPerBlock.y) + 1, 
                (ds->m / threadsPerBlock.z) + 1
                );
        gpu_mmMulFilt_naive<<<numBlocks, threadsPerBlock>>>
            (Xh_dev, Xth_dev, Yh_dev, Xsqr_dev, ds->m, k2p2_, ds->n, k2p2_ );
    }
    // KERNEL 3
    float* Xinv_dev; cudaMalloc((void**)&Xinv_dev, ds->m * k2p2_ * k2p2_ * sizeof(float));
    {
        // Every block handles one inversion. Since k2p2 is less than 23 we have 2*k^2 < 1024
        dim3 threadsPerBlock(2*k2p2_,k2p2_);
        dim3 numBlocks(ds->m);
        // Hardcoded magic number 8 = K because i need to
        gpu_batchMatInv_naive<8><<<numBlocks, threadsPerBlock>>>(Xsqr_dev, Xinv_dev, ds->m);
    }

    // Kernel 4
    // Xh  is kxn
    // Yh  is mxn
    // out is mxk 
    float* beta0_dev; cudaMalloc((void**)&beta0_dev, ds->m * k2p2_ * sizeof(float));
    {
        //dim3 threadsPerBlock(16, 16);
        int threadsPerBlock = ds->n;
        int numBlocks = ds->m;
        gpu_mvMulFilt<<<numBlocks, threadsPerBlock>>>
            (Xh_dev, Yh_dev, beta0_dev, ds->m, k2p2_, ds->n);
    }

    float* beta_dev; cudaMalloc((void**)&beta_dev, ds->m * k2p2_ * sizeof(float));
    // Xinv is a mxKxK matrix
    // Bea0 is a mxK matrix
    // Output is a mxK matrix
    {
        dim3 threadsPerBlock(8);
        dim3 numBlocks((ds->m) + 1);
        gpu_mvMul<<<numBlocks, threadsPerBlock>>>
            (Xinv_dev, beta0_dev, beta_dev, ds->m, k2p2_, k2p2_);
    }

    // Xt     is a NxK matrix
    // beta   is a mxK matrix
    // Output is a mxN matrix
    // Cuda mem
    float* y_preds_dev; cudaMalloc((void**)&y_preds_dev, ds->m * ds->N * sizeof(float));
    float* Xt_dev; cudaMalloc((void**)&Xt_dev, k2p2_ * ds->N * sizeof(float));
    // GPU Transpose
    {
        const int T = 32; int height = k2p2_; int width = ds->N;
        // 1. setup block and grid parameters
        unsigned int sh_mem_size = T * (T+1) * sizeof(float); 
        int  dimy = (height+T-1) / T; 
        int  dimx = (width +T-1) / T;
        dim3 block(T, T, 1);
        dim3 grid (dimx, dimy, 1);

        matTransposeTiledKer<T><<<grid, block, sh_mem_size>>>
            (X_dev, Xt_dev, height, width);
        cudaDeviceSynchronize();
        
    }
    // GPU mvMul
    {
        int threadsPerBlock = ds->N;
        int numBlocks = ds->m;
        gpu_mvMulFilt<<<numBlocks, threadsPerBlock>>>
            (Xt_dev, beta_dev, y_preds_dev, ds->m, ds->N, k2p2_);
    }
    // Kernel 5
    // CPU
    // GPU
    float* r_dev; cudaMalloc((void**)&r_dev, ds->m * ds->N * sizeof(float));
    int* k_dev; cudaMalloc((void**)&k_dev, ds->m * ds->N * sizeof(int));
    int* Ns_dev; cudaMalloc((void**)&Ns_dev, ds->m*sizeof(int));

    {
        dim3 threadsPerBlock(ds->N);
        dim3 numBlocks(ds->m);
        gpu_YErrorCalculation<<<numBlocks, threadsPerBlock>>>
            (images_dev, y_preds_dev, r_dev, k_dev, Ns_dev, ds->m, ds->N);
    }
    
    // Kernel 6
    // GPU Allocations
    float* sigmas_dev; cudaMalloc((void**)&sigmas_dev, ds->m * sizeof(float));
    int* ns_dev; cudaMalloc((void**)&ns_dev, ds->m * sizeof(int));
    int* hs_dev; cudaMalloc((void**)&hs_dev, ds->m * sizeof(int));

    {
        dim3 threadsPerBlock(ds->N);
        dim3 numBlocks(ds->m);
        gpu_NSSigma<<<numBlocks, threadsPerBlock>>>
            (r_dev, Yh_dev, sigmas_dev, hs_dev, ns_dev, ds->N, ds->n, ds->m, k2p2_, ds->hfrac);
    }
    // Kernel 7
    // TODO: Fix this 4 gpu speed
    int* hmax_ptr; cudaMalloc((void**)&hmax_ptr, 1*sizeof(int));
    hmaxCalc<<<1,1>>>(hs_dev, ds->m, hmax_ptr);
    int* hmax_ptr_host = (int*)malloc(sizeof(int)) ;
    cudaMemcpy(hmax_ptr_host, hmax_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    int hmax = hmax_ptr_host[0];

    float* MOfst_dev; cudaMalloc((void**)&MOfst_dev, ds->m * sizeof(float*));
    float* BOUND_dev; cudaMalloc((void**)&BOUND_dev, (ds->N - ds->n) * sizeof(float*));
    {
        dim3 threadsPerBlock(ds->N - ds->n);
        dim3 numBlocks(ds->m);
        gpu_msFst<<<numBlocks, threadsPerBlock>>>
            (r_dev, ns_dev, hs_dev, MOfst_dev, ds->m, ds->N, hmax);
    }
    {
        dim3 threadsPerBlock(ds->N - ds->n);
        dim3 numBlocks(1);
        gpu_calcBound<<<numBlocks, threadsPerBlock>>>
            (MI_dev, BOUND_dev, ds->lam, ds->N, ds->n, ds->mappingIndices[ds->N-1]);
    }

    int* breaks_dev; cudaMalloc((void**)&breaks_dev, ds->m * sizeof(int*));
    
    // Moving sums
    {
        dim3 threadsPerBlock(ds->N - ds->n);
        dim3 numBlocks(ds->m);
        gpu_mosum<<<numBlocks, threadsPerBlock>>>(Ns_dev, ns_dev, sigmas_dev, hs_dev,
                MOfst_dev, r_dev, k_dev, BOUND_dev, breaks_dev, ds->N, ds->n, ds->m);
    }
    cudaFree(X_dev);
    cudaFree(Xh_dev);
    cudaFree(Xth_dev);
    cudaFree(Yh_dev);
    cudaFree(Xsqr_dev);
    cudaFree(Xinv_dev);
    cudaFree(beta0_dev);
    cudaFree(beta_dev);
    cudaFree(y_preds_dev);
    cudaFree(Xt_dev);
    cudaFree(r_dev);
    cudaFree(k_dev);
    cudaFree(Ns_dev);
    cudaFree(sigmas_dev);
    cudaFree(ns_dev);
    cudaFree(hs_dev);
    cudaFree(MOfst_dev);
    cudaFree(BOUND_dev);
    cudaFree(breaks_dev);
    return;
}

void benchmark_gpu(dataset* ds, int runs){
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    for(int i = 0; i < runs; i++){
        gpu_run(ds);
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs; 
    printf("GPU Ran %d runs in avg. %lu microsecs\n", runs, elapsed);
    return ;
}
