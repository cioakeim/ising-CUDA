/* V2: One thread per block of moments. */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "isingV2.h"

void getDimensionsV2(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength){
  // Threads per dimension so that all n are covered (overallocating for imperfect fits).
  const int threadsPerDimension=(n/threadBlockLength)+(n%threadBlockLength>0);
  // Block and grid size according to BLOCK_MAX (like v1).
  const int blockLength=(threadsPerDimension<BLOCK_MAX2)?(threadsPerDimension):(BLOCK_MAX2);
  const int gridLength=(threadsPerDimension-1)/BLOCK_MAX2+1;
  // Set outputs.
  blockSize=dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

void allocateGridV2(char ***G,int n){
  // Allocate 1 chunk of data for better locality.
  char *data;
  cudaMallocManaged(&data,n*n*sizeof(char));
  // Set up the row pointers.
  cudaMallocManaged(G,n*sizeof(char*));
  for(int i=0;i<n;i++){
    (*G)[i]=data+n*i;
  }
  return;
}

__global__ void initializeRandomGridV2(char **G,int n, int threadBlockLength){
  // Each thread responsible for initializing its own block.
  // BlockX,Y point to the starting position of the thread's block..
  const int blockX=threadBlockLength*(blockIdx.x*blockDim.x+threadIdx.x);
  const int blockY=threadBlockLength*(blockIdx.y*blockDim.y+threadIdx.y);
  // Initialize RNG..
  curandState state;
  curand_init(clock64(),blockX*n+blockY,0,&state);
  // Initialize whole block..
  for(int i=blockX;i<blockX+threadBlockLength;i++){
    for(int j=blockY;j<blockY+threadBlockLength;j++){
      // If the targeted element is out of bounds look for something else.
      if(i>=n || j>=n){
        continue;
      }
      G[i][j]=curand(&state)%2;
    }
  }
  return;
}

void evolveIsingGridV2(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize, 
                       int threadBlockLength){
  if(k<0){
    fprintf(stderr,"Error in evolveIsingGridV1: k must be a positive integer.\n");
    exit(1);
  }
  char **Gtemp;
  cudaError_t cudaError;
  // Call a kernel for each iteration.
  for(int run_count=0;run_count<k;run_count++){
    // Send everything to the 0 stream for concurrency.
    generateNextGridStateV2<<<gridSize,blockSize,0>>>(G,G0,n,threadBlockLength);
    cudaError=cudaGetLastError();
    if(cudaError!=cudaSuccess){
      fprintf(stderr,"Error in generateNextGridStateV2: %s\n",cudaGetErrorString(cudaError));
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

__global__ void generateNextGridStateV2(char **G,char **G0,int n, int threadBlockLength){
  // Each thread responsible for computing its own block.
  // BlockX,Y point to the starting position of the thread's block..
  const int blockX=threadBlockLength*(blockIdx.x*blockDim.x+threadIdx.x);
  const int blockY=threadBlockLength*(blockIdx.y*blockDim.y+threadIdx.y);
  char temp;
  for(int i=blockX;i<blockX+threadBlockLength;i++){
    for(int j=blockY;j<blockY+threadBlockLength;j++){
      // If the current element is out of bounds move to next possible element.
      if(i>=n || j>=n){
        continue;
      }
      // Compute using majority rule.
      temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
      G[i][j]=(temp>2)?(1):(0);
    }
  }
  return;
}

void freeGridV2(char **G){
  // If grid is not initialized do nothing.
  if(G==NULL){
    return;
  }
  // Free data chunk.
  cudaFree(G[0]);
  // Free row pointers.
  cudaFree(G);
  return;
}
