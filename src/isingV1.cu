/* V1: One thread per moment. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "isingV1.h"

void getDimensionsV1(int n, dim3 &blockSize, dim3 &gridSize){
  // BLOCK MAX POWER OF 2.
  int blockLength=(n<BLOCK_MAX1)?(n):(BLOCK_MAX1);
  int gridLength=((n-1)/BLOCK_MAX1+1);
  blockSize= dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

void allocateGridV1(char ***G,int n){
  char *data;
  cudaMallocManaged(&data,n*n*sizeof(char));
  cudaMallocManaged(G,n*sizeof(char*));
  for(int i=0;i<n;i++){
    (*G)[i]=data+n*i;
  }
  return;
}

__global__ void initializeRandomGridV1(char **G,int n){
  // Each thread initializes one element.
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int j=blockIdx.y*blockDim.y+threadIdx.y;
  // Overallocated threads return.
  if(i>=n||j>=n){
    return;
  }
  // Initialization of curand. 
  curandState state;
  curand_init(clock64(),i+n*j,0,&state);
  G[i][j]=curand(&state)%2;
  return;
}

// Main algorithm.
void evolveIsingGridV1(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize){
  if(k<0){
    fprintf(stderr,"Error in evolveIsingGridV1: k must be a positive integer.\n");
    exit(1);
  }
  char **Gtemp;
  cudaError_t cudaError;
  // Run the evolution k times.
  for(int run_count=0;run_count<k;run_count++){
    // Send all kernels to 0 stream to ensure concurrency.
    generateNextGridStateV1<<<gridSize,blockSize,0>>>(G,G0,n); 
    cudaError=cudaGetLastError();
    if(cudaError!=cudaSuccess){
      fprintf(stderr,"Error in generateNextGridStateV1: %s\n",cudaGetErrorString(cudaError));
      exit(1);
    }
    // Swap pointers and pass by value for the next iteration.
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  // At each loop's end the original G contains the new state 
  // since it's passed by value so no need for additional swaps.

  // Ensure all is done.
  cudaDeviceSynchronize();
  return;
}

__global__ void generateNextGridStateV1(char **G,char **G0,int n){
  // Each thread in the grid is assigned to its own moment.
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  const int j=blockIdx.y*blockDim.y+threadIdx.y;
  // Overallocated threads are gone.
  if(i>=n||j>=n){
    return;
  }
  // Decide grid value by majority rule.
  char temp; 
  temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
  G[i][j]=(temp>2)?(1):(0);
  return;
}

// Free grid 
void freeGridV1(char **G){
  // If G is not initialized do nothing.
  if(G==NULL){
    return;
  }
  // Free data chunk.
  cudaFree(G[0]);
  // Free pointers.
  cudaFree(G);
  return;
}
