#ifndef _CPU_BENCH
#define _CPU_BENCH


float cpu_run(dataset* ds){
    // KERNEL 1
    // Make interpolation matrix
    int k2p2 = 2*ds->k + 2;
    const int k2p2_ = (ds->trend > 0) ? k2p2 : k2p2-1;
    float* X_host = (float*) malloc(k2p2_ * ds->N * sizeof(float));
    // Get mappingIndices and images to the GPU
    // Perform sequential validation run
    seq_mkX(k2p2_, ds->N, ds->freq, ds->mappingIndices, X_host);

    // Host mem
    float* Xh_host = (float*) malloc(k2p2_ * ds->n * sizeof(float)); // Kxn
    float* Xth_host= (float*) malloc(k2p2_ * ds->n * sizeof(float)); // nxK
    float* Yh_host = (float*) malloc(ds->m * ds->n * sizeof(float)); // mxn
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
    // KERNEL 2
    float* Xsqr_host = (float*) malloc(ds->m * k2p2_ * k2p2_ * sizeof(float));
    seq_mmMulFilt(Xh_host, Xth_host, Yh_host, Xsqr_host, ds->m, k2p2_, ds->n, k2p2_ );

    // KERNEL 3
    float* Xinv_host = (float*) malloc(ds->m * k2p2_ * k2p2_ * sizeof(float));
    seq_matInv(Xsqr_host, Xinv_host, ds->m, k2p2_);

    // Kernel 4
    float* beta0_host = (float*) malloc(ds->m * k2p2_ * sizeof(float));
    seq_mvMulFilt(Xh_host, Yh_host, beta0_host, ds->m, k2p2_, ds->n);

    float* beta_host = (float*) malloc(ds->m * k2p2_*sizeof(float));
    seq_mvMul(Xinv_host, beta0_host, beta_host, ds->m, k2p2_, k2p2_);
    float* y_preds_host = (float*) malloc(ds->m * ds->N * sizeof(float));
    float* Xt_host = (float*) malloc(k2p2_ * ds->N * sizeof(float));

    seq_transpose(X_host, Xt_host, k2p2_, ds->N);
    seq_mvMulFilt(Xt_host, beta_host, y_preds_host, ds->m, ds->N, k2p2_);

    // Kernel 5
    // CPU
    float* r_host = (float*) malloc(ds->m * ds->N * sizeof(float));
    int* k_host = (int*) malloc(ds->m * ds->N * sizeof(int));
    int* Ns_host = (int*) malloc(ds->m*sizeof(int));
    seq_YErrorCalculation(ds->images, y_preds_host, r_host, k_host, Ns_host, ds->m, ds->N);
    
    // Kernel 6
    // CPU Allocations
    float* sigmas_host = (float*) malloc(ds->m * sizeof(float));
    int* ns_host     = (int*) malloc(ds->m * sizeof(int));
    int* hs_host     = (int*) malloc(ds->m * sizeof(int));

    seq_NSSigma(r_host, Yh_host, sigmas_host, hs_host, ns_host, ds->N, ds->n, ds->m, k2p2_, ds->hfrac);

    // Kernel 7
    int hmax = -100000;
    for(int i = 0; i < ds->m; i++){
        if (hs_host[i] > hmax)
            hmax = hs_host[i];
    } 
    float* MOfst_host = (float*) malloc(ds->m * sizeof(float*));
    float* BOUND_host = (float*) malloc((ds->N - ds->n)*sizeof(float));
    seq_msFst(r_host, ns_host, hs_host, MOfst_host, ds->m, ds->N, hmax);
    for(int q = 0; q < ds->N - ds->n; q++){
        int t    = ds->n+q;
        int time = ds->mappingIndices[t];
        float tmp = logplus( ((float)time) / ((float) ds->mappingIndices[ds->N-1]));
        BOUND_host[q] = ds->lam * (sqrtf(tmp));
    }

    

    int* breaks_host = (int*) malloc(ds->m * sizeof(int*));
    
    // Moving sums
    seq_mosum(Ns_host, ns_host, sigmas_host, hs_host, MOfst_host, r_host, k_host, BOUND_host,
            breaks_host, ds->N, ds->n, ds->m);

    // Free all the things
    free(X_host);
    free(Xh_host);
    free(Xth_host);
    free(Yh_host);
    free(Xsqr_host);
    free(Xinv_host);
    free(beta0_host);
    free(beta_host);
    free(y_preds_host);
    free(Xt_host);
    free(r_host);
    free(k_host);
    free(Ns_host);
    free(sigmas_host);
    free(ns_host);
    free(hs_host);
    free(MOfst_host);
    free(BOUND_host);
    free(breaks_host);
    return 0.0f;
}
// Returns the average break difference
float benchmark_cpu(dataset* ds, int runs){
    
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    for(int i = 0; i < runs; i++){
        cpu_run(ds);
    }
    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs; 
    printf("CPU Ran %d runs in avg. %lu microsecs\n", runs, elapsed);
    return 0.0f;
}
#endif
