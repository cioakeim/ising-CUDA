#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV1.h"
#include "isingV2.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

int main(){
  // Grids for V2.
  char **G0=NULL;
  char **G=NULL;
  // Grids for V1.
  char **G01=NULL;
  char **G1=NULL;
  // Iteration parameters.
  int n_min=20;
  int n_max=2000;
  int n_step=20;
  int k_min=20;
  int k_max=40;
  int k_step=20;
  // CUDA variables:
  dim3 blockSizeV1,gridSizeV1;
  dim3 blockSizeV2,gridSizeV2;
  cudaError_t cudaError;
  // This needs tweeking:
  int threadBlockSize=4;
  
  // For each size.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids for algorithms.
    // V1. 
    gridAllocateV1(&G1,n);
    gridAllocateV1(&G01,n);
    getDimensionsV1(n,blockSizeV1,gridSizeV1);
    // V2.
    gridAllocateV2(&G,n);
    gridAllocateV2(&G0,n);
    getDimensionsV2(n,blockSizeV2,gridSizeV2,threadBlockSize);
    // For each iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      initRandomV2<<<gridSizeV2,blockSizeV2>>>(G0,n,threadBlockSize);
      cudaDeviceSynchronize();
      // Error check for initRandom..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initRandomV2: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to V1 initial state:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          G01[i][j]=G0[i][j];
        }
      }
      // Run both algorithms.
      isingV1(G1,G01,n,k,blockSizeV1,gridSizeV1);
      cudaDeviceSynchronize();
      isingV2(G,G0,n,k,blockSizeV1,gridSizeV1,threadBlockSize);
      // Compare results and exit if there is an error..
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(G[i][j]!=G1[i][j]){
            printf("Results don't match for n:%d and k:%d\n",n,k);
            exit(1);
          }
        }
      }
    }
    printf("Size %d done\n",n);
    freeGridV1(G1);
    freeGridV1(G01);
    freeGridV2(G);
    freeGridV2(G0);
  }
  printf("Testing successful.\n");
  return 0;
}
