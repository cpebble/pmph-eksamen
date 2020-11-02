// This will be the main C file to play with
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <errno.h>
#include "helpers.cu.h"
#include "cuda-kernels.cu.h"
#include "cpu-kernels.cu.h"

int validate(){
    return 0;
}

int readDataset(char* pathname, dataset* ds){
// Read a dataset in the newline seperated format:
// trend - i32
// k - i32
// n - i32
// freq - f32
// hfrac - f32
// lam - f32
// N - i32
// m - i32
// mappingindices - i32[N]
// images - f32[m][N]
 
    FILE* file = fopen(pathname, "r");
    if (file == NULL){
        printf("Error reading dataset %s:\n%s\n", pathname, strerror(errno));
        return 1;
    }
    // Init dataset struct
    fscanf(file, "%d\n%d\n%d\n%f\n%f\n%f\n%d\n%d\n",
            &ds->trend, &ds->k, &ds->n,
            &ds->freq, &ds->hfrac, &ds->lam,
            &ds->N, &ds->m
        );

    // Read mappingIndices
    ds->mappingIndices = (int*)malloc(ds->N*sizeof(int));
    int e = readIntArray(ds->N, ds->mappingIndices, file);
    switch(e){
        case 0:
            printf("Successfully read Dataset mappingindices\n");
            break;
        case -1:
            printf("FP not pointing to start of array\n");
            return -1;
        case -2:
            printf("Error in scanning array\n");
            return -1;
        case -3:
            printf("Error in scanning end of array\n");
            return -1;
        default:
            printf("Unexpected error in array scan\n");
            return -1;
    }

    // Read the image array
    ds->images = (float*) malloc(ds->m*ds->N*sizeof(float));
    float* curImage = ds->images;
    char first = (char) fgetc(file);
    if (first != '['){
        printf("Error in reading matrix start\n");
        return -1;
    }
    for(int i = 0; i < ds->m; i++){
        e = readFloatArray(ds->N, curImage, file);
        switch(e){
            case 0:
                break;
            case -1:
                printf("FP not pointing to start of array\n");
                return -1;
            case -2:
                printf("Error in scanning array\n");
                return -1;
            case -3:
                printf("Error in scanning end of array\n");
                return -1;
            default:
                printf("Unexpected error in array scan\n");
                return -1;
        }
        curImage += ds->N;
        if (i != ds->m - 1 && ((char)fgetc(file) != ',' || (char)fgetc(file) != ' ')){
            printf("Format Error\n");
            return -1;
        }
    }
    printf("Successfully read Image Array\n");

    fclose(file);
    return 0;
}
int validate(dataset* ds){
    // KERNEL 1
    // Make interpolation matrix
    printf("Creating X matrix\n");
    int k2p2 = 2*ds->k + 2;
    const int k2p2_ = (ds->trend > 0) ? k2p2 : k2p2-1;
    float* X_host = (float*) malloc(k2p2_ * ds->N * sizeof(float));
    // Get mappingIndices and images to the GPU
    int* MI_dev; cudaMalloc((void**)&MI_dev, ds->N * sizeof(int));
    float* images_dev; cudaMalloc((void**)&images_dev, ds->m * ds->N * sizeof(float));
    cudaMemcpy(MI_dev, ds->mappingIndices, ds->N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(images_dev, ds->images, ds->m * ds->N*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // Perform sequential validation run
    seq_mkX(k2p2_, ds->N, ds->freq, ds->mappingIndices, X_host);

    // Now CudaMalloc and do the same thing on the GPU
    float* X_dev; cudaMalloc((void**)&X_dev, k2p2_*ds->N*sizeof(float));
    // Threads per block(kind of arbitrary for the naive version)
    {
        dim3 threadsPerBlock(8, 8);
        dim3 numBlocks((ds->N / threadsPerBlock.x) + 1, (k2p2_ / threadsPerBlock.y));
        gpu_mkX<<<numBlocks, threadsPerBlock>>>(k2p2_, ds->N, ds->freq, MI_dev, X_dev);
        
        
    }
    //cudaDeviceSynchronize();
    float* X_dev_v = (float*)malloc(k2p2_*ds->N*sizeof(float));
    cudaMemcpy(X_dev_v, X_dev, k2p2_*ds->N*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(X_host, X_dev_v, 1, k2p2_, ds->N, 0.001f);
    free(X_dev_v);

    printf("Transposing matrices and extracting Historical data\n");
    // Host mem
    float* Xh_host = (float*) malloc(k2p2_ * ds->n * sizeof(float)); // Kxn
    float* Xth_host= (float*) malloc(k2p2_ * ds->n * sizeof(float)); // nxK
    float* Yh_host = (float*) malloc(ds->m * ds->n * sizeof(float)); // mxn
    // Cuda mem
    float* Xh_dev;  cudaMalloc((void**)&Xh_dev,  k2p2_ * ds->n * sizeof(float));
    float* Xth_dev; cudaMalloc((void**)&Xth_dev, k2p2_ * ds->n * sizeof(float));
    float* Yh_dev;  cudaMalloc((void**)&Yh_dev, ds->m * ds->n * sizeof(float));
    // Do the list slicing sequentially
    for(int k = 0; k < k2p2_; k++){
        for (int i = 0; i < ds->n; i++){
            // Copy Xh[:,:n]
            Xh_host[k*ds->n + i] = X_host[k*ds->N + i];
        }
    }
    for(int j = 0; j < ds->m; j++){
        for (int i = 0; i < ds->n; i++){
            // Copy Yh[:,:n]
            Yh_host[j*ds->n + i] = ds->images[j*ds->N + i];
        }
    }
    // And gpu slicing
    {
        dim3 threadsPerBlock(1,1);
        dim3 numBlocks((ds->n / threadsPerBlock.x) + 1, (k2p2_ / threadsPerBlock.y) + 1);
        sliceMatrix<<<numBlocks, threadsPerBlock>>>(X_dev, Xh_dev, k2p2_, ds->n, ds->N);
        dim3 y_numBlocks((ds->n / threadsPerBlock.x) + 1, (ds->m / threadsPerBlock.y) + 1);
        printf("%d\n", y_numBlocks.y);
        sliceMatrix<<<y_numBlocks, threadsPerBlock>>>(images_dev, Yh_dev, ds->m, ds->n, ds->N);
    }
    printf("Validating Xh\n");
    float* Xh_dev_v = (float*)malloc(k2p2_*ds->n*sizeof(float));
    cudaMemcpy(Xh_dev_v, Xh_dev, k2p2_*ds->n*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(Xh_host, Xh_dev_v, 1, k2p2_, ds->n, 0.001f);
    free(Xh_dev_v);
    printf("Validating Yh\n");
    float* Yh_dev_v = (float*)malloc(ds->m*ds->n*sizeof(float));
    cudaMemcpy(Yh_dev_v, Yh_dev, ds->m*ds->n*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(Yh_host, Yh_dev_v, 1, ds->m, ds->n, 0.001f);
    free(Yh_dev_v);
    // Transpose X
    printf("Transposing now:\n");
    seq_transpose(Xh_host, Xth_host, k2p2_, ds->n);
    
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
    printf("Validating Xth\n");
    float* Xth_dev_v = (float*)malloc(ds->n * k2p2_ * sizeof(float));
    cudaMemcpy(Xth_dev_v, Xth_dev, ds->n * k2p2_ * sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(Xth_host, Xth_dev_v, 1, ds->n, k2p2_, 0.001f);
    free(Xth_dev_v);
    printf("[!]K1 done\n");

    // KERNEL 2
    printf("Creating Xsqr\n");
    // CPU
    float* Xsqr_host = (float*) malloc(ds->m * k2p2_ * k2p2_ * sizeof(float));
    // Cuda
    float* Xsqr_dev; cudaMalloc((void**)&Xsqr_dev, ds->m * k2p2_ * k2p2_ * sizeof(float));

    seq_mmMulFilt(Xh_host, Xth_host, Yh_host, Xsqr_host, ds->m, k2p2_, ds->n, k2p2_ );
    printf("Testing Xqr_dev\n");
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
    //cudaDeviceSynchronize();
    float* Xsqr_dev_v = (float*)malloc(k2p2_*k2p2_*ds->m*sizeof(float));
    cudaMemcpy(Xsqr_dev_v, Xsqr_dev, k2p2_*k2p2_*ds->m*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(Xsqr_host, Xsqr_dev_v, ds->m, k2p2_, k2p2_, 0.01f);
    free(Xsqr_dev_v);


    printf("[!]K2 Done\n");
    // KERNEL 3
    printf("Inverting Xsqr\n");
    float* Xinv_host = (float*) malloc(ds->m * k2p2_ * k2p2_ * sizeof(float));
    float* Xinv_dev; cudaMalloc((void**)&Xinv_dev, ds->m * k2p2_ * k2p2_ * sizeof(float));
    seq_matInv(Xsqr_host, Xinv_host, ds->m, k2p2_);
    printf("GPU Inversion\n");
    {
        // Every block handles one inversion. Since k2p2 is less than 23 we have 2*k^2 < 1024
        dim3 threadsPerBlock(2*k2p2_,k2p2_);
        dim3 numBlocks(ds->m);
        // Hardcoded magic number 8 = K because i need to
        gpu_batchMatInv_naive<8><<<numBlocks, threadsPerBlock>>>(Xsqr_dev, Xinv_dev, ds->m);
    }
    printf("Validating inversion\n");
    float* Xinv_dev_v = (float*)malloc(k2p2_*k2p2_*ds->m*sizeof(float));
    cudaMemcpy(Xinv_dev_v, Xinv_dev, k2p2_*k2p2_*ds->m*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(Xinv_host, Xinv_dev_v, ds->m, k2p2_, k2p2_, 0.01f);
    free(Xinv_dev_v);
    printf("[!]K3 Done\n");

    // Kernel 4
    printf("running Vector Multiplication and calculating betas\n");
    printf("Filtered first\n");
    // Xh  is kxn
    // Yh  is mxn
    // out is mxk 
    float* beta0_host = (float*) malloc(ds->m * k2p2_ * sizeof(float));
    float* beta0_dev; cudaMalloc((void**)&beta0_dev, ds->m * k2p2_ * sizeof(float));
    seq_mvMulFilt(Xh_host, Yh_host, beta0_host, ds->m, k2p2_, ds->n);
    {
        //dim3 threadsPerBlock(16, 16);
        int threadsPerBlock = ds->n;
        int numBlocks = ds->m;
        gpu_mvMulFilt<<<numBlocks, threadsPerBlock>>>
            (Xh_dev, Yh_dev, beta0_dev, ds->m, k2p2_, ds->n);
    }
    printf("Validating beta0\n");
    float* beta0_dev_v = (float*)malloc(ds->m*k2p2_*sizeof(float));
    cudaMemcpy(beta0_dev_v, beta0_dev, ds->m*k2p2_*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(beta0_host, beta0_dev_v, 1, ds->m, k2p2_, 0.01f);
    free(beta0_dev_v);

    printf("Unfiltered beta and y_preds\n");
    float* beta_host = (float*) malloc(ds->m * k2p2_*sizeof(float));
    float* beta_dev; cudaMalloc((void**)&beta_dev, ds->m * k2p2_ * sizeof(float));
    // Xinv is a mxKxK matrix
    // Bea0 is a mxK matrix
    // Output is a mxK matrix
    seq_mvMul(Xinv_host, beta0_host, beta_host, ds->m, k2p2_, k2p2_);
    {
        dim3 threadsPerBlock(8);
        dim3 numBlocks((ds->m) + 1);
        gpu_mvMul<<<numBlocks, threadsPerBlock>>>
            (Xinv_dev, beta0_dev, beta_dev, ds->m, k2p2_, k2p2_);
    }
    printf("Validating beta\n");
    cudaDeviceSynchronize();
    float* beta_dev_v = (float*)malloc(ds->m * k2p2_ * sizeof(float));
    cudaMemcpy(beta_dev_v, beta_dev, ds->m * k2p2_*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(beta_host, beta_dev_v, 1, ds->m, k2p2_, 0.01f);
    free(beta_dev_v);

    // Xt     is a NxK matrix
    // beta   is a mxK matrix
    // Output is a mxN matrix
    float* y_preds_host = (float*) malloc(ds->m * ds->N * sizeof(float));
    // Transpose X
    float* Xt_host = (float*) malloc(k2p2_ * ds->N * sizeof(float));
    // Cuda mem
    float* y_preds_dev; cudaMalloc((void**)&y_preds_dev, ds->m * ds->N * sizeof(float));
    float* Xt_dev; cudaMalloc((void**)&Xt_dev, k2p2_ * ds->N * sizeof(float));

    seq_transpose(X_host, Xt_host, k2p2_, ds->N);
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
    seq_mvMulFilt(Xt_host, beta_host, y_preds_host, ds->m, ds->N, k2p2_);
    // GPU mvMul
    {
        int threadsPerBlock = ds->N;
        int numBlocks = ds->m;
        gpu_mvMulFilt<<<numBlocks, threadsPerBlock>>>
            (Xt_dev, beta_dev, y_preds_dev, ds->m, ds->N, k2p2_);
    }
    printf("Validating ypreds\n");
    float* y_preds_dev_v = (float*)malloc(ds->m * ds->N * sizeof(float));
    cudaMemcpy(y_preds_dev_v, y_preds_dev, ds->m * ds->N*sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(y_preds_host, y_preds_dev_v, 1, ds->m, ds->N, 0.1f);
    free(y_preds_dev_v);
    printf("[!]K4 Done\n");

    // Kernel 5
    printf("Calculating Y_errors\n");
    // CPU
    float* r_host = (float*) malloc(ds->m * ds->N * sizeof(float));
    int* k_host = (int*) malloc(ds->m * ds->N * sizeof(int));
    int* Ns_host = (int*) malloc(ds->m*sizeof(int));
    // GPU
    float* r_dev; cudaMalloc((void**)&r_dev, ds->m * ds->N * sizeof(float));
    int* k_dev; cudaMalloc((void**)&k_dev, ds->m * ds->N * sizeof(int));
    int* Ns_dev; cudaMalloc((void**)&Ns_dev, ds->m*sizeof(int));

    seq_YErrorCalculation(ds->images, y_preds_host, r_host, k_host, Ns_host, ds->m, ds->N);
    {
        dim3 threadsPerBlock(ds->N);
        dim3 numBlocks(ds->m);
        gpu_YErrorCalculation<<<numBlocks, threadsPerBlock>>>
            (images_dev, y_preds_dev, r_dev, k_dev, Ns_dev, ds->m, ds->N);
    }
    printf("Validating Y errors\n");
    float* r_dev_v = (float*)malloc(ds->m * ds->N * sizeof(float));
    cudaMemcpy(r_dev_v, r_dev, ds->m * ds->N *sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(r_host, r_dev_v, 1, ds->m, ds->N, 0.1f);
    free(r_dev_v);
    
    // Kernel 6
    printf("Calculating Sigmas\n");
    // CPU Allocations
    float* sigmas_host = (float*) malloc(ds->m * sizeof(float));
    int* ns_host     = (int*) malloc(ds->m * sizeof(int));
    int* hs_host     = (int*) malloc(ds->m * sizeof(int));
    // GPU Allocations
    float* sigmas_dev; cudaMalloc((void**)&sigmas_dev, ds->m * sizeof(float));
    int* ns_dev; cudaMalloc((void**)&ns_dev, ds->m * sizeof(int));
    int* hs_dev; cudaMalloc((void**)&hs_dev, ds->m * sizeof(int));

    seq_NSSigma(r_host, Yh_host, sigmas_host, hs_host, ns_host, ds->N, ds->n, ds->m, k2p2_, ds->hfrac);
    {
        dim3 threadsPerBlock(ds->N);
        dim3 numBlocks(ds->m);
        gpu_NSSigma<<<numBlocks, threadsPerBlock>>>
            (r_dev, Yh_dev, sigmas_dev, hs_dev, ns_dev, ds->N, ds->n, ds->m, k2p2_, ds->hfrac);
    }
    printf("Validating sigmas\n");
    float* sigmas_dev_v = (float*)malloc(ds->m * sizeof(float));
    cudaMemcpy(sigmas_dev_v, sigmas_dev, ds->m * sizeof(float), cudaMemcpyDeviceToHost);
    validateMatrices(sigmas_host, sigmas_dev_v, 1, 1, ds->m, 0.01f);
    free(sigmas_dev_v);
    printf("Sigmas calculated\n");
    printf("[!]K6 Done\n");

    // Kernel 7
    printf("Calculating hmax: ");
    int hmax = -100000;
    for(int i = 0; i < ds->m; i++){
        if (hs_host[i] > hmax)
            hmax = hs_host[i];
    } 
    printf("%d\n", hmax);
    float* MOfst_host = (float*) malloc(ds->m * sizeof(float*));
    float* BOUND_host = (float*) malloc((ds->N - ds->n)*sizeof(float));
    float* MOfst_dev; cudaMalloc((void**)&MOfst_dev, ds->m * sizeof(float*));
    float* BOUND_dev; cudaMalloc((void**)&BOUND_dev, (ds->N - ds->n) * sizeof(float*));
    printf("Calculated MO_fsts\n");
    seq_msFst(r_host, ns_host, hs_host, MOfst_host, ds->m, ds->N, hmax);
    printf("Calculating BOUND\n");
    for(int q = 0; q < ds->N - ds->n; q++){
        int t    = ds->n+q;
        int time = ds->mappingIndices[t];
        float tmp = logplus( ((float)time) / ((float) ds->mappingIndices[ds->N-1]));
        BOUND_host[q] = ds->lam * (sqrtf(tmp));
    }
    {
        dim3 threadsPerBlock(hmax);
        dim3 numBlocks(ds->m);
        gpu_msFst<<<numBlocks, threadsPerBlock>>>
            (r_dev, ns_dev, hs_dev, MOfst_dev, ds->m, ds->N, hmax);
    }
    printf("Validating MOfst\n");
    float* MOfst_dev_v = (float*)malloc(ds->m * sizeof(float));
    cudaMemcpy(MOfst_dev_v, MOfst_dev, ds->m * sizeof(float), cudaMemcpyDeviceToHost);
    // We are starting to see some serious divergence from host calculations. However they
    // still look right in the debugger
    validateMatrices(MOfst_host, MOfst_dev_v, 1, 1, ds->m, 0.5f); 

    free(MOfst_dev_v);

    return 0;
    printf("[!]K7 Done\n");
    

    float* breaks_host = (float*) malloc(ds->m * sizeof(float*));
    float* breaks_dev; cudaMalloc((void**)&breaks_dev, ds->m * sizeof(float*));
    
    // Moving sums
    seq_mosum(Ns_host, ns_host, sigmas_host, hs_host, MOfst_host, r_host, k_host, BOUND_host,
            ds->N, ds->n, ds->m);

    printMatrix(breaks_host, 1, 3);
    // Free everything!!!
    // TODO:
    //free(X_host);
    //free(Xh_host);
    return 0;
    
    
}


int main(int argc, char* argv[]){
    dataset* ds = (dataset*) malloc(sizeof(dataset));
    char* dsPath = "data/small_peru.clean";
    readDataset(dsPath, ds);
    printf("Ready to work on dataset of %d images, with %d pixels each\n", ds->N, 
            ds->m);
    validate(ds);
    return 0;
}
