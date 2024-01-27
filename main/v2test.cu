/* Test that proves the correctness of version V2 of the implementation. It is assumed that V1 
 * works correctly. For each pair (n,k) an initial state is created using V2's implementation and 
 * is hard copied to V1's initial state. Both algorithms are run and if there are differences the 
 * program exits with an error message.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "isingV1.h"
#include "isingV2.h"

int main(){
  // Grids for V2.
  char **G0=NULL;
  char **G=NULL;
  // Grids for V1.
  char **G01=NULL;
  char **G1=NULL;
  // Iteration parameters.
  int n_min=10;
  int n_max=2000;
  int n_step=20;
  int k_min=20;
  int k_max=40;
  int k_step=20;
  // For the kernel calls.
  dim3 blockSizeV1,gridSizeV1;
  dim3 blockSizeV2,gridSizeV2;
  // Error checking.
  cudaError_t cudaError;
  // Prime number was chosen for better possibility of size mismatches.
  int threadBlockSize=7;
  
  // For each size.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids and get dimensions for algorithms.
    // V1. 
    allocateGridV1(&G1,n);
    allocateGridV1(&G01,n);
    getDimensionsV1(n,blockSizeV1,gridSizeV1);
    // V2.
    allocateGridV2(&G,n);
    allocateGridV2(&G0,n);
    getDimensionsV2(n,blockSizeV2,gridSizeV2,threadBlockSize);
    // For each iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      initializeRandomGridV2<<<gridSizeV2,blockSizeV2>>>(G0,n,threadBlockSize);
      cudaDeviceSynchronize();
      // Error check for random state..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initializeRandomGridV2: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to V1's initial state:
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          G01[i][j]=G0[i][j];
        }
      }
      // Run both algorithms.
      evolveIsingGridV1(G1,G01,n,k,blockSizeV1,gridSizeV1);
      evolveIsingGridV2(G,G0,n,k,blockSizeV1,gridSizeV1,threadBlockSize);
      // Compare results and exit if there is an error..
      int error_count=0;
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(G[i][j]!=G1[i][j]){
            error_count++;
          }
        }
      }
      if(error_count>0){
        printf("Results don't match for n:%d and k:%d. No. of errors: %d\n",n,k,error_count);
        exit(1);
      }
    }
    printf("Size %d done\n",n);
    // Cleanup.
    freeGridV1(G1);
    freeGridV1(G01);
    freeGridV2(G);
    freeGridV2(G0);
  }
  printf("Job done. Testing was successful.\n");
  return 0;
}
