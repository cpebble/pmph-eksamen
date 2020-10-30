#ifndef _HELPERS 
#define _HELPERS
#include <sys/time.h>
#include <time.h> 
#define I2(r,c,w) (r*w+c)
#define I3(z,y,x,h,w) (h*w*z+w*y+x)

// Will include helper scripts for time values, validations, etc
float logplus(float x){
    if (x > expf(1) )
        return logf(x);
    return 1;
}
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}
int testme(int n){
    return n*n;
}
void printMatrix(float* mat, int rows, int cols){
    printf("[");
    for (int r = 0; r < rows; r++){
        printf("\n\t[");
        for(int c = 0; c < cols; c++){
            printf("%.3f, ", mat[cols*r + c]);
        }
        printf("], ");
    }
    printf("]\n");
}
void print3dMatrix(float* mat, int mats, int rows, int cols){
    printf("[");
    for (int m = 0; m < mats; m++){
        printf("\n[");
        for (int r = 0; r < rows; r++){
            printf("\n\t[");
            for(int c = 0; c < cols; c++){
                printf("%.3f, ", mat[I3(m,r,c,rows,cols)]);
            }
            printf("], ");
        }
        printf("\n]");
    }
    printf("]\n");
}
struct dataset {
    int trend;
    int k;
    int n;
    float freq;
    float hfrac;
    float lam;
    int N;
    int m;
    int* mappingIndices;
    float* images;
} ;

// Read an array of ints in futhark format
int readIntArray(int N, int* arr, FILE* fp){
    char f = (char) fgetc(fp);
    if (f != '['){
        // FP isn't pointing at the array
        return -1;
    }
    int e;
    for(int i = 0; i < N - 1; i++){
        e = fscanf(fp, "%d, ", &arr[i]);
        if (e != 1){
            return -2;
        }
    }
    // And read the last value
    e = fscanf(fp, "%d]\n", &arr[N-1]);
    if (e != 1){
        return -3;
    }
    return 0;
}

// Read an array of floats in futhark format
int readFloatArray(int N, float* arr, FILE* fp){
    char f = (char) fgetc(fp);
    if (f != '['){
        // FP isn't pointing at the array
        return -1;
    }
    int e;
    for(int i = 0; i < N - 1; i++){
        e = fscanf(fp, "%f, ", &arr[i]);
        if (e != 1){
            return -2;
        }
    }
    // And read the last value
    e = fscanf(fp, "%f]\n", &arr[N-1]);
    if (e != 1){
        return -3;
    }
    return 0;
}

#endif
