// This contains the sequential C versions to validate
#ifndef _CPU_KERNELS
#define _CPU_KERNELS
#include <math.h>

//--- Creates the interpolation matrix ---
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


// --- Filtered Matrix - Matrix Multiplication ---
// Should multiply X*X^t and filter out the rows, where y is NAN
// X is a KxN matrix
// X_t is a NxK matrix
// y is a vector of size N 
// Output is KxK matrix
void seq_mmMulFilt(float* X, float* X_t, float* y, float* X_sqr, int n, int K){
        //Loops through the rows of X
        /*for (int i = 0; i < K; i++)
        {
                //Loops through the columns of X_t
                for (int j = 0; j < K; j++)
                {
                        float acc = 0; 
                        //Calculates each element in X_sqr
                        for(int p = 0; p < n; p++){
                                int index_X = i*K + p; 
                                int index_Xt = p*N + j; 
                                float a = X[index_X] * X_t[index_Xt] *(1.0 - isnan(y[p]));
                                acc += a; 
                        }
                        int index_Xsqr = K*i + j; 
                        X_sqr[index_Xsqr] = acc; 
                }               
        }*/
}


// --- Invert a matrix --- 
// Allocate the [K][2K] array in here
// X_sqr is a KxK matrix 
// X_inv is the output and also a KxK matrix
void seq_matInv (float* X_sqr, float* X_inv, int K){
      /*  float t1, t2; 

        //skal inverte X_sqr, der er en kxk matrise 
        //først laver vi en unit-matrix af samme størrelse som X_sqr 

        // Creates a matrix containing both X_sqr and a KxK identity matrix
        float A[K][2*K]; 
        for (int i = 0; i < K; i++)
        {
                for (int j = 0; i < K; i++)
                {
                        // X_sqr will be placed at the left side of A
                        if(j < K){
                                int index_Xsqr = i*K + j; 
                                A[i][j] = X_sqr[index_Xsqr]; 
                        }// The identity matrix will be placed to the rigth side of A
                        else{
                                if(i == j){
                                        A[i][j] = 1.0; 
                                }else{
                                        A[i][j] = 0.0; 
                                }
                        }
                }            
        }
        // Now we Gauss Jordan is performed 
        for (int i = 0; i < K; i++)
        {
                t1 = A[i][i]; 
                for (int j = 0; j < K; j++)
                {
                        A[i][j] = A[i][j]/t1; 
                        A[i][j+K] = A[i][j+K]/t1; 
                }
                for(int p = 0; p < K; p++){
                        t2 = A[p][i]; 
                        for(int j = 0; j < K; j++){
                                if(p == i){
                                        break;
                                }else{
                                        A[p][j] = A[p][j] - A[i][j]*t2; 
                                        A[p][j+K] = A[p][j+K] - A[i][j+K]*t2; 
                                }
                        }
                }        
        }
        // Now we copy the elements from A to X_inv
        for (int i = 0; i < K; i++)
        {
                for (int j = 0; i < K; i++)
                {
                        int index_Xinv = i*K+ j; 
                        X_inv[index_Xinv] = A[i][j]; 
                }              
        }*/
         
        
    
}

// --- Filtered Matrix * Vector multiplication ---
// Calculate X*y
// X is assumed to a KxN matrix
// Y is also supposed to be a vector of size Nx1
// Output is y_out and will be vector of size Nx1
void seq_mvMulFilt(float* X, float* y, float* y_out int K, int N){
      /*  for (int i = 0; i < K; i++)
        {
             y_out[i] = 0; 
             for (int j = 0; j < N; j++)
             {
                     int index_X = i*K + j; 
                     y_out[i] += X[index_X] * y[j]; 
             }
                
        }    */    
}


// --- UnFiltered matrix * vector multiplication - calculates the prediction --- 
// Used for calculating Beta, and y_preds
// X_sqr is an KxK matrix
// y is an vector of size Kx1
// Ouput will be B_out, which an Kx1 vector
void seq_mvMult(float* X_sqr, float* y, float* B_out, int K, int N){
        //her skal vi gange X_sqr, så vi får en vector med størrelsen K
     /*   for (int i = 0; i < K; i++)
        {
                B_out[i] = 0; 
                for (int j = 0; j < K; j++)
                {
                        int index_Xsqr = i*K +j; 
                        B_out[i] += X_sqr[index_Xsqr] * y[j]; 
                }                
        }*/
}

// --- Calculates Y - Y_pred --- 
// Y is the real targets, which has size N
// Ypred is the predictions, which has size N
// Out will be R, which is the error
void seq_YErrorCalculation(float* Y, float* Ypred, float* R, int N, int M){
      /*  for(int i = 0; i < M; i++){
                for (int j = 0; j < N; j++)
                {
                        int index = i*M + j; 
                        R[index] = Ypred[index] - Y[index]; 
                }        
        }*/
}




// --- Kernel 6 ---
// Creates the lists hs, nss and sigmas, whih will be used in later calculatations
// Y_historic is an matrix containing, where each row is a time-serie for a pixel 
// n is the number of rows in Y_historic - which is the number of pixels
// m is the number of cols in Y_historic  
void seq_NSSigma(float* Y_errors, float* Y_historic, 
     /*   float* sigmas, float* hs, float* nss, int n, int m, float hfrac){

        // Loops through each row of Y_historic and Y_errors
        // For each iteration, we creates values for nss, hs and sigmas
        for (int i = 0; i < n; i++)
        {
                float val_ns = 0; 
                float val_h = 0; 
                float val_sigma = 0; 

                // Loops through the element in a row in Y_historic 
                for (int j = 0; j < m; j++)
                {
                        // Current value of Y_historic
                        int val_Yh = Y_historic[i *n + j];
                        if(isnan(val_Yh)){
                                val_ns += 1;
                        }
                }

                //vi skal ligger værdien 
                

        }
        





        //for hver række i y udregner vi et sigma, ns og hs 
        for(int yh = 0; yh < N; yh++){
                //Calculates the ns value for the current pixel
                ns[yh] = 0; 
                for (int ye = 0; ye < M; ye++)
                {
                        float cur_ye = Y_historic[yh*N + ye]; 
                        if(!isnan(cur_ye)){
                                ns[yh] += 1;         
                        }
                }
                //Calculates the sigma value for the current pixel
                sigma[yh] = 0;
                for (int i = 0; i < N; i++)
                {
                        float cur = 0  
                        if(i < ns[yh]){
                                cur = Y_errors[yh*N + ye]; 
                        }
                        cur = cur*cur; 
                        sigma[yh] += cur;
                }
                h[yh] = ns[yh] * hfrac;      
        }*/
}

// --- Kernel 7 ---
// 
void seq_msFst(int hMax, float* Y_error, float* hss, float* nss, float* msFst, float* bounds, int N, int n){

        //looper hen over rækkerne i de tre matriser, Y_errors, nss og hss 

        //looper hen over hvert element i hver række for de tre matriser
        // for hver af de tre elementer 

}

void seq_mosum(
        float* Nss, float* nss, float* sigmas, float* hs,
        float* msFsts, float* Y_error, float* val_inds, 
        float* breaks, float* means
        );

#endif
