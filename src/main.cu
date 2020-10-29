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
    int k2p2_ = (ds->trend > 0) ? k2p2 : k2p2-1;
    float* X_host = (float*) malloc(k2p2_ * ds->N * sizeof(float));
    seq_mkX(k2p2_, ds->N, ds->freq, ds->mappingIndices, X_host);

    printf("Transposing matrices and extracting Historical data\n");
    float* Xh_host = (float*) malloc(k2p2_ * ds->n * sizeof(float));
    float* Xth_host= (float*) malloc(k2p2_ * ds->n * sizeof(float));
    float* Yh_host = (float*) malloc(ds->m * ds->n * sizeof(float));
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
    // Transpose X
    seq_transpose(Xh_host, Xth_host, k2p2_, ds->n);
    printMatrix(Xth_host,  ds-> n, k2p2_);
    printf("[!]K1 done\n");

    // KERNEL 2
    printf("Creating Xsqr\n");
    float* Xsqr_host = (float*) malloc(k2p2_ * k2p2_ * sizeof(float));
    seq_mmMulFilt(Xh_host, Xth_host, Yh_host, Xsqr_host, ds->n, k2p2_, k2p2);
    printf("[!]K2 Done\n");
    // KERNEL 3
    printf("Inverting Xsqr\n");
    float* Xinv_host = (float*) malloc(k2p2_ * k2p2_ * sizeof(float));
    seq_matInv(Xsqr_host, Xinv_host, k2p2_);
    printf("[!]K3 Done\n");

    // Kernel 4
    printf("running Vector Multiplication and calculating betas\n");
    printf("Filtered first\n");
    float* beta0_host = (float*) malloc(k2p2_ * ds->m * sizeof(float));
    for(int row = 0; row < ds->m; row++){
        seq_mvMulFilt(Xh_host, (Yh_host + (row*ds->n)), beta0_host+(row*k2p2_), k2p2, ds->n);
    }

    printf("Unfiltered beta and y_preds\n");
    float* beta_host = (float*) malloc(k2p2_ * ds->m * sizeof(float));
    for(int row = 0; row < ds->m; row++){
        seq_mvMul(Xinv_host, (beta0_host + (row*k2p2_)), beta_host+(row*k2p2_), k2p2_, k2p2_);
    }
    float* ypreds_host = (float*) malloc(ds->N * ds->m * sizeof(float));
    // Transpose X
    float* Xt_host = (float*) malloc(ds->N * k2p2_ * sizeof(float));
    seq_transpose(X_host, Xt_host, k2p2_, ds->N);
    
    for(int row = 0; row < ds->m; row++){
        seq_mvMul(Xt_host, beta_host + (row*k2p2_), ypreds_host + (row*ds->N), ds->N, k2p2_);
    }
    printf("[!]K4 Done\n");
    // Kernel 5
    printf("Calculating Y_errors\n");
    float* r_host = (float*) malloc(ds->m * ds->N * sizeof(float));
    seq_YErrorCalculation(ds->images, ypreds_host, r_host, ds->m, ds->N);
    printf("[!]K5 Done\n");
    
    // Kernel 6
    printf("Calculating Sigmas\n");
    float* sigmas_host = (float*) malloc(ds->m * sizeof(float));
    float* ns_host     = (float*) malloc(ds->m * sizeof(float));
    int* hs_host     = (int*) malloc(ds->m * sizeof(float));
    seq_NSSigma(r_host, Yh_host, sigmas_host, hs_host, ns_host, ds->N, ds->m, k2p2_, ds->hfrac);
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
    seq_msFst(hmax, r_host, hs_host, ns_host, MOfst_host, BOUND_host, ds->N, ds->n);
    printf("Calculated MO_fsts and Bounds\n");
    printf("[!]K7 Done\n");

    float* breaks_host = (float*) malloc(ds->m * sizeof(float*));
    float* means_host  = (float*) malloc(ds->m * sizeof(float*));
    
    // Moving sums
    for(int pixel = 0; pixel < ds->m; pixel++){
        float* _pixel = ds->images + (pixel*ds->m);
        float* MO = (float*) malloc((ds->N - ds->n) * sizeof(float));
        // Calculate SUM^t_
        float sigSqrN = sigmas_host[pixel] * sqrtf(ns_host[pixel]);
        for(int i = 0; i < ds->N - ds->n; i++){
            if (i == 0){ // Put in mo_fst
                MO[i] = MOfst_host[pixel];
            }
            // I'm not sure this needs to be there, since we allow nan in MO
            //if(isnan(_pixel[i]))
                //MO[i] = MO[i-1]; // If Nan, assume no forest cut down in the meantime
                //continue;
            int h = hs_host[pixel];
            // r[n + t] - r[n + t - h]
            float tmp = (r_host[pixel*ds->m + ds->n + i] - r_host[pixel*ds->m + (ds->n + i - h)]);
            MO[i] = tmp / sigSqrN;

        }
        // Now MO is calculated
        int firstBreak = -1;
        for (int j = 0; j < ds->N - ds->n; j++){
            float b = BOUND_host[j];
            float mo= MO[j];
            if (!isnan(mo)){
                if (fabsf(mo) > b){
                    firstBreak = j;
                    break;
                }
            }
        }
        breaks_host[pixel] = firstBreak;
    }

    // Free everything!!!
    // TODO:
    free(X_host);
    free(Xh_host);
    return 0;
    
}


int main(int argc, char* argv[]){
    float* a = (float*)malloc(8*sizeof(float));
    float* b = (float*)malloc(8*sizeof(float));
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 2; j++){
            a[i*2+j] = (float)(i+j);
        }
    }
    printMatrix(a, 4, 2);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 4; j++){
            b[i*4+j] = (float)(i+j);
        }
    }
    printMatrix(b, 2, 4);
    float* c = (float*)malloc(4*4*sizeof(float));
    float* idk=(float*)malloc(1024*sizeof(float));
    seq_mmMulFilt(a, b, idk, c, 4, 2, 4);
    printMatrix(c, 4, 4);

    return 0;
    dataset* ds = (dataset*) malloc(sizeof(dataset));
    char* dsPath = "data/small_peru.clean";
    readDataset(dsPath, ds);
    printf("Ready to work on dataset of %d images, with %d pixels each\n", ds->N, 
            ds->m);
    validate(ds);
    return 0;
}
