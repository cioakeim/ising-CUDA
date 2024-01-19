#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV1.h"
#include "isingV0.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

int main(){
  // Grids used for V1.
  char **G0=NULL;
  char **G=NULL;
  // Grids used for V0.
  char **G0seq=NULL;
  char **Gseq=NULL;
  // Define iteration parameters:
  int n_min=20;
  int n_max=2000;
  int n_step=200;
  int k_min=10;
  int k_max=40;
  int k_step=10;
  // CUDA variables:
  dim3 blockSize,gridSize;
  cudaError_t cudaError;

  // For each test size..
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids for both algorithms.
    // v0 
    gridAllocateV0(&Gseq,n);
    gridAllocateV0(&G0seq,n);
    // v1
    gridAllocateV1(&G0,n);
    gridAllocateV1(&G,n);
    getDimensionsV1(n, blockSize, gridSize);
    // For each test iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      initRandomV1<<<gridSize,blockSize>>>(G0,n);
      cudaDeviceSynchronize();
      // Error check for initRandom..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initRandom: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to sequential initial state:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          G0seq[i][j]=G0[i][j];
        }
      }
      // Run both algorithms:
      isingV0(Gseq,G0seq,n,k);
      isingV1(G,G0,n,k,blockSize,gridSize);
      // Compare results and exit if there is an error:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(G[i][j]!=Gseq[i][j]){
            printf("Results don't match for n:%d and k:%d\n",n,k);
            exit(1);
          }
        }
      }
    } 
    printf("Size %d done\n",n);
    freeGridV0(Gseq);
    freeGridV0(G0seq);
    freeGridV1(G);
    freeGridV1(G0);
  }
  printf("Job done, testing was a success.\n");
  return 0;
}
