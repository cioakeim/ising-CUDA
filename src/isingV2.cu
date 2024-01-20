/* V2: One thread per block of moments
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV2.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

// Define dimensions of the grid structure based on n. 
void getDimensionsV2(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength){
  // Threads per dimension so that all n are covered (overallocating for imperfect fits).
  int threadsPerDimension=(n/threadBlockLength)+(n%threadBlockLength>0);
  // Block and grid size according to BLOCK_MAX (like v1).
  int blockLength=(threadsPerDimension<BLOCK_MAX)?(threadsPerDimension):(BLOCK_MAX);
  int gridLength=(threadsPerDimension-1)/BLOCK_MAX+1;
  blockSize=dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

// Allocation is not global so its separate. 
void gridAllocateV2(char ***G,int n){
  char *data;
  cudaMallocManaged(&data,n*n*sizeof(char));
  cudaMallocManaged(G,n*sizeof(char*));
  for(int i=0;i<n;i++){
    (*G)[i]=data+n*i;
  }
  return;
}

// Generate random grid based on the dimensions given.
__global__ void initRandomV2(char **G,int n, int threadBlockLength){
  // Each thread responsible for initializing its own block.
  // BlockI,J point to the starting position of the thread's block..
  int blockI=threadBlockLength*(blockIdx.x*blockDim.x+threadIdx.x);
  int blockJ=threadBlockLength*(blockIdx.y*blockDim.y+threadIdx.y);
  // Initialize RNG..
  curandState state;
  curand_init(clock64(),blockI*n+blockJ,0,&state);
  // Initialize whole block..
  for(int i=blockI;i<blockI+threadBlockLength;i++){
    for(int j=blockJ;j<blockJ+threadBlockLength;j++){
      // For the overallocated threads.
      if(i>=n || j>=n){
        return;
      }
      G[i][j]=curand(&state)%2;
    }
  }
  return;
}

// Main algorithm.
void isingV2(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize, int threadBlockLength){
  char **Gtemp;
  cudaError_t cudaError;
  for(int run_count=0;run_count<k;run_count++){
    nextStateV2<<<gridSize,blockSize>>>(G,G0,n,threadBlockLength);
    cudaError=cudaGetLastError();
    if(cudaError!=cudaSuccess){
      fprintf(stderr,"Error in nextStateV2: %s\n",cudaGetErrorString(cudaError));
      exit(1);
    }
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

// Generate next state.
__global__ void nextStateV2(char **G,char **G0,int n, int threadBlockLength){
  int blockI=threadBlockLength*(blockIdx.x*blockDim.x+threadIdx.x);
  int blockJ=threadBlockLength*(blockIdx.y*blockDim.y+threadIdx.y);
  char temp;
  for(int i=blockI;i<blockI+threadBlockLength;i++){
    for(int j=blockJ;j<blockJ+threadBlockLength;j++){
      // For the overallocated threads.
      if(i>=n || j>=n){
        return;
      }
      temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
      G[i][j]=(temp>2)?(1):(0);
    }
  }
  return;
}

// Free grid 
void freeGridV2(char **G){
  if(G==NULL){
    return;
  }
  cudaFree(G[0]);
  cudaFree(G);
  G=NULL;
  return;
}
