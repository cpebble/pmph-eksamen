// This will be the main C file to play with
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <strings.h>
#include <errno.h>
#include "helpers.cu.h"
#include "cuda-kernels.cu.h"

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

int main(int argc, char* argv[]){
    dataset* ds = (dataset*) malloc(sizeof(dataset));
    char* dsPath = "data/peru.clean";
    readDataset(dsPath, ds);
    printf("%d\n%d\n%d\n%ff\n%ff\n%ff\n%d\n%d\n",
            ds->trend, ds->k, ds->n,
            ds->freq, ds->hfrac, ds->lam,
            ds->N, ds->m
        );
    return 0;
}

