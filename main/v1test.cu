/* This script tests the correctness of the V1 implementation of the assignment.
 * It is assumed that V0 works correctly. 
 * For given parameters, a random state is initialized and hard copied to another grid.
 * The algorithms run for the same number of steps on the initial state and the results 
 * are compared. If there is an inequality of the results the program exits.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "isingV1.h"
#include "isingV0.h"

int main(){
  // Grids used for V1.
  char **G0=NULL;
  char **G=NULL;
  // Grids used for V0.
  char **G0seq=NULL;
  char **Gseq=NULL;
  // Define iteration parameters:
  int n_min=10;
  int n_max=2000;
  int n_step=20;
  int k_min=10;
  int k_max=40;
  int k_step=10;
  // CUDA variables:
  dim3 blockSize,gridSize;
  cudaError_t cudaError;

  // For each test size..
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids for both algorithms.
    // V0 
    allocateGridV0(&Gseq,n);
    allocateGridV0(&G0seq,n);
    // V1
    allocateGridV1(&G0,n);
    allocateGridV1(&G,n);
    getDimensionsV1(n, blockSize, gridSize);
    // For each test iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      initializeRandomGridV1<<<gridSize,blockSize>>>(G0,n);
      cudaDeviceSynchronize();
      // Error check for random state..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initalizeRandomGridV1: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to sequential initial state:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          G0seq[i][j]=G0[i][j];
        }
      }
      // Run both algorithms:
      evolveIsingGridV0(Gseq,G0seq,n,k,false);
      evolveIsingGridV1(G,G0,n,k,blockSize,gridSize);
      // Compare results and exit if there is an error:
      int error_count=0;
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(G[i][j]!=Gseq[i][j]){
            error_count++;
          }
        }
      }
      // If there is and error, report miss count and exit.
      if(error_count>0){
        printf("Results don't match for n:%d and k:%d. No. of errors: %d\n",n,k,error_count);
        exit(1);
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
