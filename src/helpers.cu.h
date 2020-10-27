#include <sys/time.h>
#include <time.h> 
// Will include helper scripts for time values, validations, etc

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
