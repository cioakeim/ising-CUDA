#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV3.h"
#include "isingV2.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

int main(){
  // Grids for V3.
  char **G0=NULL;
  char **G=NULL;
  // Grids for V2.
  char **G02=NULL;
  char **G2=NULL;
  // Iteration parameters.
  int n_min=20;
  int n_max=2000;
  int n_step=20;
  int k_min=20;
  int k_max=40;
  int k_step=20;
  // CUDA variables:
  dim3 blockSizeV3,gridSizeV3;
  dim3 blockSizeV2,gridSizeV2;
  cudaError_t cudaError;
  // initBlockLength is for the initial state only.
  int initBlockLength=16; 
  int threadBlockSize=2;
  // For each size.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids for algorithms.
    // V2. 
    gridAllocateV2(&G2,n);
    gridAllocateV2(&G02,n);
    getDimensionsV2(n,blockSizeV2,gridSizeV2,threadBlockSize);
    // V3.
    gridAllocateV3(&G,n);
    gridAllocateV3(&G0,n);
    getDimensionsV3(n,blockSizeV3,gridSizeV3,initBlockLength);
    // For each iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      initRandomV3<<<gridSizeV3,blockSizeV3>>>(G0,n,initBlockLength);
      cudaDeviceSynchronize();
      // Error check for initRandom..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initRandomV3: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to V2 initial state:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          G02[i][j]=G0[i][j];
        }
      }
      // Run both algorithms.
      // Grid dimensions update for the second half of the algorithm.
      getDimensionsV3(n,blockSizeV3,gridSizeV3,1);
      isingV3(G,G0,n,k,blockSizeV3,gridSizeV3);
      cudaDeviceSynchronize();
      isingV2(G2,G02,n,k,blockSizeV2,gridSizeV2,threadBlockSize);
      cudaDeviceSynchronize();
      // Compare results and exit if there is an error..
      int errCount=0;
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(G[i][j]!=G2[i][j]){
            errCount++;
          }
        }
      }
      if(errCount>0){
        printf("Results don't match for n:%d and k:%d, error count:%d\n",n,k,errCount);
        exit(1);
      }
    }
    printf("Size %d done\n",n);
    freeGridV2(G2);
    freeGridV2(G02);
    freeGridV3(G);
    freeGridV3(G0);
  }
  printf("Testing successful.\n");
  return 0;

}
