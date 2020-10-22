// This contains the sequential C versions to validate
#ifndef _CPU_KERNELS
#define _CPU_KERNELS

// Makes interpolation matrix
void seq_mkX (int K, float f, float* X_out);

// Filtered Matrix - Matrix Multiplication
// X * X^T
void seq_mmMulFilt(float* X, float* y, float* X_sqr);

// We need to invert here
// Allocate the [K][2K] array in here
// X_inv is the output
void seq_matInv (float* X_sqr, float X_inv, int K);

// Filtered Matrix * Vector multiplication
void seq_mvMulFilt(float* X, float* y, float Beta_zero);

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
