/* V1: One thread per moment
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV1.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

// Gets the dimensions required for the algorithms.
void getDimensionsV1(int n, dim3 &blockSize, dim3 &gridSize){
  // BLOCK MAX POWER OF 2.
  int blockLength=(n<BLOCK_MAX)?(n):(BLOCK_MAX);
  int gridLength=((n-1)/BLOCK_MAX+1);
  blockSize= dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

// Allocate memory for operations.
void gridAllocateV1(char ***G,int n){
  char *data;
  cudaMallocManaged(&data,n*n*sizeof(char));
  cudaMallocManaged(G,n*sizeof(char*));
  for(int i=0;i<n;i++){
    (*G)[i]=data+n*i;
  }
  return;
}

// Generate random grid.
__global__ void initRandomV1(char **G,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  if(i>=n||j>=n){
    return;
  }
  curandState state;
  curand_init(clock64(),i+n*j,0,&state);
  G[i][j]=curand(&state)%2;
  return;
}

// Main algorithm.
void isingV1(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize){
  char **Gtemp;
  cudaError_t cudaError;
  for(int run_count=0;run_count<k;run_count++){
    nextStateV1<<<gridSize,blockSize>>>(G,G0,n); 
    cudaDeviceSynchronize();
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  Gtemp=G;
  G=G0;
  G0=Gtemp;
  return;
}

// Create next state in parallel. 
__global__ void nextStateV1(char **G,char **G0,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  if(i>=n||j>=n){
    return;
  }
  char temp; 
  temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
  G[i][j]=(temp>2)?(1):(0);
  return;
}

// Free grid 
void freeGridV1(char **G){
  if(G==NULL){
    return;
  }
  cudaFree(G[0]);
  cudaFree(G);
  G=NULL;
  return;
}
