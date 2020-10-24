// This contains the sequential C versions to validate
#ifndef _CPU_KERNELS
#define _CPU_KERNELS
#include <math.h>
#include <cmath>

// Makes interpolation matrix 
//K is the number of rows, K = 2*k + 2 
//N is the number of cols
void seq_mkX (int K, int N, float f, float* X_out){
        for (int i = 0; i < K; i++)
        {
                for (int ind = 0; ind < N; ind++)
                {
                        //Calculates the current index for X_out
                        int index = K*i + ind; 
                        if(i == 0){
                                X_out[index] = 1; //1f32
                        }
                        else if(i == 1){
                                X_out[index] = i;//r32 ind 
                        }else{
                                float f_t = (2*M_PI*ind*i)/f; //Ft(j)=2πjt/f.
                                if (i % 2 == 0){
                                       X_out[index] = sin(f_t);  
                                }else{
                                    X_out[index] = cos(f_t);
                                }
                        }
                }
                
        }
        
}

// Filtered Matrix - Matrix Multiplication
// X is assumed to be a nxm matrix
void seq_mmMulFilt(float* X, float* y, float* X_sqr, int n, int m){
        // Calculates X * X^T
        for (int i = 0; i < n; i++)
        {
                for(int j = 0; j < n; j ++){
                        int index_X_sqr = i*n + j; 
                        X_sqr[index_X_sqr] = 0; 
                        for(int k = 0; k < m; k ++){
                                int index_X = i*n + k; 
                                int index_X_T = k*m + i;
                                X_sqr[index_X_sqr] += X[index_X] * X[index_X_T];
                        }
                }
        }
        //Maps y to every row in X_sqr
        for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < n; j++)
                {
                        int index = i*n + j; 
                        X_sqr[index] = X_sqr[index] * y[j]; 
                }       
        }
}
                

// We need to invert here
// Allocate the [K][2K] array in here
// X_inv is the output
void seq_matInv (float* X_sqr, float* X_inv, int n, int m){
        for (int row = 0; row < row; row++)
        {
                for (int col = 0; col < m; col ++)
                {
                        float val = X_sqr[row*n + col];
                        X_inv[col*m + row] = val;                                               
                }
        }
}


bool isnan(float y){

}

// Filtered Matrix * Vector multiplication
void seq_mvMulFilt(float* X, float* y, float* Beta_zero, int n, int m){
        for (int i = 0; i < n; i++)
        {
                Beta_zero[i] = 0; 
                for (int j = 0; j < m; j++)
                {
                        int index_X = n*i + j;
                        float cur_X = X[index_X]; 
                        float cur_y = y[j]; 
                        float xy = cur_X * cur_y * (1.0 - isnan(cur_y)); 
                        Beta_zero[i] += xy; //er ikke sikker på at den skal indsættes der                         
                }                
        }       
}

// UnFiltered matrix * vector multiplication
// Used for calculating Beta, and y_preds
void seq_mvMult(float* X, float* y, float* X_out);

// Calculates Y - Y_pred
//NB: Has to check if nan and filter those out
// Takes Y and Y_hat
// Outputs:
/// Nbar: # valid keys - int
/// r: # prediction errors - [N]float
/// I: Indices of valid keys - [Nbar]int
//
// Also remember that we pad the resulting r array to size N.
void seq_YErrorCalculation(
        float* Y, float* Ypred, int N,
        int Nbar, float* r, int* I
        );

void seq_NSSigma(
        float* Y_errors, float* Y_historic,
        float* sigma, float* h, float* ns
        );

void seq_msFst(
        int hMax, float* Y_error, int N, int n,
        float* msFst, float* bounds
        );

void seq_mosum(
        float* Nss, float* nss, float* sigmas, float* hs,
        float* msFsts, float* Y_error, float* val_inds, 
        float* breaks, float* means
        );

#endif
