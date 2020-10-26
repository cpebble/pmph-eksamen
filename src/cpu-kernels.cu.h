// This contains the sequential C versions to validate
#ifndef _CPU_KERNELS
#define _CPU_KERNELS
#include <math.h>
#include <cmath>

//-- Creates the interpolation matrix ---
// K is the number of rows
// N is the number of columns 
// f is the frequency of the observations
// X_out is the resulting interpolation matrix, which has size KxN
//      ecach column, x_t, will correspond to the data generated for a spefic date t
void seq_mkX(int K, int N, float f, float* X_out){
        //Loops through the rows
        for (int i = 0; i < K; i++)
        {
                
                //Loops through the indices in each row and creates the elements
                for(int t = 0; t < N; t++){
                        //Calculates the current index for X_out
                        int cur_ind = K*i + t;
                        if(i == 0){
                                //This correspond to the first index of each x_t, which should always be 1
                                X_out[cur_ind] = 1;
                        }else if (i == 1)
                        {
                                //This correspond the second index of each x_t, which should be the date, t
                                X_out[cur_ind] = t; 
                        }else
                        {
                                //Calculates Ft(j)=2πjt/f
                                float F = (2*M_PI*i*t)/f;  
                                if(i% 2 == 0){
                                        X_out[cur_ind] = sin(F);
                                }else{
                                        X_out[cur_ind] = cos(F);
                                } 
                        }                       
                }
        }
        
}

// Filtered Matrix - Matrix Multiplication
// X is assumed to be a nxm matrix
//Output is nxn matrix
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
                
void seq_scatter(float* X, int* I, float* D, int n){

}

void seq_gauss_jordan(float* A, int n, int m){
        for (int i = 0; i < n; i++)
        {
                int v1 = A[i]; 
                float _A[m]; 
                for (int ind = 0; ind < m; ind++)
                {
                        int k = ind/m; 
                        int j = ind % m; 
                        if(v1 == 0.0){
                                _A[ind] = A[k*m+j]; //er ikke sikker på om det skal være det
                        }else
                        {
                                float x = A[j]/v1; 
                                if(k < n -1){
                                        _A[ind] = A[(k+1)*m+j] - A[(k+1)*m+i] * x; 
                                }else
                                {
                                        _A[ind] = x; 
                                }
                                
                        }
                        
                }
                
        }
        //in  scatter A (iota nm) A'
        
}
// We need to invert here
// Allocate the [K][2K] array in here
// X_sqr is a nxn matrix 
// X_inv is the output
void seq_matInv (float* X_sqr, float* X_inv, int n){
        int m = 2*n; 
        int nm = n*m; 
        float Ap[nm]; 
        for (int ind = 0; ind < nm; ind++)
        {
                int i = ind/m; 
                int j = ind % m; 
                if(j < n){
                        int index = i* n+ j; //ved ikke om det er rigtigt
                        Ap[ind] = X_sqr[index];
                }else if (j == n + i)
                {
                        Ap[ind] = 1.0; 
                }else
                {
                        Ap[ind] = 0.0
                }    
        }

        //create a gauss_jordan on Ap
        
    
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
