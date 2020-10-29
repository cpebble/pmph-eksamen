// This contains the sequential C versions to validate
#ifndef _CPU_KERNELS
#define _CPU_KERNELS
#include <math.h>

//--- Creates the interpolation matrix ---
// K is the number of rows
// N is the number of columns 
// f is the frequency of the observations
// X_out is the resulting interpolation matrix, which has size KxN
//  ecach column, x_t, will correspond to the data generated for a spefic date t
void seq_mkX(int K, int N, float f, int* mappingIndices, float* X_out){
    //Loops through the rows
    for (int i = 0; i < K; i++)
    {         
        //Loops through the indices in each row and creates the elements
        for(int t = 0; t < N; t++){
            //Calculates the current index for X_out
            int cur_ind = N*i + t;
            if(i == 0){
                //This correspond to the first index of each x_t, which should always be 1
                X_out[cur_ind] = 1;
            } else if (i == 1) {
                //This correspond the second index of each x_t
                X_out[cur_ind] = (float)mappingIndices[t]; 
            } else {
                //Calculates Ft(j)
                float i_ = (float)(i / 2);
                float j_ = ((float)mappingIndices[t]);
                float F = (2.0f*M_PI*i_*j_)/f;  
                if(i% 2 == 0){
                    X_out[cur_ind] = sinf(F);
                }else{
                    X_out[cur_ind] = cosf(F);
                }
            }
        }
    }
}


float seq_dotprodFilt(float* x, float* y, float* vec, int n){
    float sum = 0.0f;
    for(int i = 0; i < n; i++){
        if (!isnan(vec[i]))
            sum += x[i] * y[i];
    }
    return 0.0f;
}
float seq_dotprod(float* x, float* y, int n){
    float sum = 0.0f;
    for(int i = 0; i < n; i++){
        sum += x[i] * y[i];
    }
    return sum;
}

// --- Filtered Matrix - Matrix Multiplication ---
// Should multiply X*X^t and filter out the rows, where y is NAN
// X is a KxN matrix
// X_t is a NxK matrix
// y is a matrix of size m*N
// Output is mxKxK matrix
void seq_mmMulFilt(float* X, float* X_t, float* y,
        float* M, int pixels, int n, int p, int m){
    for(int pix = 0; pix < pixels; pix++){
        for(int i = 0; i < n; i++){
            for (int j = 0; j < m; j++){
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
    }
}


// --- Invert a matrix --- 
// Allocate the [K][2K] array in here
// X_sqr is a mxKxK matrix 
// A is the output and also a mxKxK matrix. 768b, 192 floats
void seq_matInv (float* X_sqr, float* A, int matrices, int height){
    int width = 2*height;
    // And will be shared in the grid blocks
    // 384 float
    float* Ash = (float*)malloc(height*width*sizeof(float));// This will be array we gauss-jordan
    for(int i = 0; i < matrices; i++){
        for(int k1 = 0; k1 < height; k1++){
            for(int k2 = 0; k2 < width; k2++){
                // Fill up the tmp array
                //Ash[I2(k1, k2, width)] = 0.0f;
                int ind = I2(k1, k2, width);
                int ind3= I3(i, k1, k2, height, height);
                Ash[ind] = (k2 < height) ?
                    X_sqr[ind3] : 
                    (k2 == height + k1);
            }
        }
        printf("Filled Ash %d times, which is less than %d\n",i, matrices);
        for(int i_=0;i_<height;i_++)
        {
            float curEl = Ash[I2(i_, i_, width)];
            if(Ash[I2(i_,i_,width)] == 0.0)
            {
                printf("Mathematical Error!");
                continue;
            }
            for(int j=0;j<height;j++)
            {
                if(i_!=j)
                {
                     float ratio = Ash[I2(j,i_,width)]/Ash[I2(i_,i_,width)];
                     for(int k=0;k<width;k++)
                     {
                         Ash[I2(j,k,width)] = Ash[I2(j,k,width)] - ratio*Ash[I2(i_,k,width)];
                     }
                }
            }
            for(int c = 0; c < width; c++){
                Ash[I2(i_, c, width)] /= curEl;
            }
            printf("More %f\n", curEl);
            print3dMatrix(Ash, 1, height, 16);
        }
        for(int k1 = 0; k1 < height; k1++){
            for(int k2 = 0; k2 < height; k2++){
                A[I3(i, k1, k2, height, height)] = Ash[I2(k1, height+k2, width)];
            }
        }
    }
}

// --- Filtered Matrix * Vector multiplication ---
// Calculate X*y
// X is assumed to a KxN matrix
// Y is also supposed to be a vector of size Nx1
// Output is y_out and will be vector of size Nx1
void seq_mvMulFilt(float* X, float* y, float* y_out, int K, int N){ 
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
void seq_mvMul(float* X_sqr, float* y, float* B_out, int K, int N){
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
         float* sigmas, int* hs, float* nss, int n, int m, int k, float hfrac){
}

// --- Kernel 7 ---
// 
void seq_msFst(int hMax, float* Y_error, int* hss, float* nss, float* msFst, float* bounds, int N, int n){

    //looper hen over rækkerne i de tre matriser, Y_errors, nss og hss 

    //looper hen over hvert element i hver række for de tre matriser
    // for hver af de tre elementer 

}

void seq_mosum(
    float* Nss, float* nss, float* sigmas, float* hs,
    float* msFsts, float* Y_error, float* val_inds, 
    float* breaks, float* means
    ){

}

// HELPERS
void seq_transpose(float* X_in, float* X_out, int rows, int cols){
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            // X_out[c][r] = X_in[r][c]
            X_out[c * rows + r] = X_in[r*cols + c];
        }
    }
}

#endif



