/* Test that proves the correctness of version V3 of the implementation. It is assumed that V2 
 * works correctly. For each pair (n,k) an initial state is created using V3's implementation and 
 * is hard copied to V2's initial state. Both algorithms are run and if there are differences the 
 * program exits with an error message.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "isingV3.h"
#include "isingV2.h"

int main(){
  // Grids for V3.
  char *G0=NULL;
  char *G=NULL;
  char *HostG=NULL;
  // Grids for V2.
  char **G02=NULL;
  char **G2=NULL;
  // Iteration parameters.
  int n_min=10;
  int n_max=2000;
  int n_step=20;
  int k_min=20;
  int k_max=40;
  int k_step=20;
  // For the kernel calls.
  dim3 blockSizeV3,gridSizeV3;
  int initBlockLengthV3=16; 
  int iterBlockLengthV3;
  dim3 blockSizeV2,gridSizeV2;
  int threadBlockSizeV2=2;
  // Error checking.
  cudaError_t cudaError;
  // initBlockLength is for the initial state only.
  // For each size.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate grids for algorithms.
    // V2. 
    allocateGridV2(&G2,n);
    allocateGridV2(&G02,n);
    getDimensionsV2(n,blockSizeV2,gridSizeV2,threadBlockSizeV2);
    // V3.
    allocateGridV3(&G,n);
    allocateGridV3(&G0,n);
    HostG=(char*)malloc(n*n*sizeof(char));
    // For each iteration count..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Create random state:
      getInitializationDimensionsV3(n,blockSizeV3,gridSizeV3,initBlockLengthV3);
      initializeRandomGridV3<<<gridSizeV3,blockSizeV3>>>(G0,n,initBlockLengthV3);
      cudaDeviceSynchronize();
      // Error check for random state..
      cudaError=cudaGetLastError();
      if(cudaError!=cudaSuccess){
        printf("Kernel failed at initializeRandomGridV3: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Hard copy to V2 initial state:
      cudaError=cudaMemcpy(G02[0], G0, n*n*sizeof(char), cudaMemcpyDeviceToHost);
      if(cudaError!=cudaSuccess){
        printf("Error at cudaMemcpy: %s\n",cudaGetErrorString(cudaError));
        exit(1);
      }
      // Run both algorithms.
      getEvolutionDimensionsV3(n,blockSizeV3,gridSizeV3,iterBlockLengthV3);
      evolveIsingGridV3(HostG,G,G0,n,k,blockSizeV3,gridSizeV3,iterBlockLengthV3);
      evolveIsingGridV2(G2,G02,n,k,blockSizeV2,gridSizeV2,threadBlockSizeV2);
      // Compare results and exit if there is an error..
      int errCount=0;
      for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
          if(HostG[i*n+j]!=G2[i][j]){
            errCount++;
          }
        }
      }
      if(errCount>0){
        printf("Results don't match for n:%d and k:%d, No. of errors:%d\n",n,k,errCount);
        exit(1);
      }
    }
    printf("Size %d done\n",n);
    freeGridV2(G2);
    freeGridV2(G02);
    freeGridV3(G);
    freeGridV3(G0);
    free(HostG);
  }
  printf("Job done. Testing was successful.\n");
  return 0;
}
