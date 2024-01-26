/* V3: Multiple threads sharing common input moments. */
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

#include "isingV3.h"

void getInitializationDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength){
  // Threads per dimension so that all n are covered (overallocating for imperfect fits).
  const int threadsPerDimension=(n/threadBlockLength)+(n%threadBlockLength>0);
  // Block and grid size according to BLOCK_MAX (like v1).
  const int blockLength=(threadsPerDimension<BLOCK_MAX3)?(threadsPerDimension):(BLOCK_MAX3);
  const int gridLength=(threadsPerDimension-1)/BLOCK_MAX3+1;
  // Set outputs.
  blockSize=dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

void getEvolutionDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int &blockLength){
  // Each gridBlock calculates a blockLength x blockLength portion of the total grid.
  // At each block, 4*blockLength threads are added to load the perimeter.
  blockLength=(n<BLOCK_MAX3)?n:BLOCK_MAX3;
  // Grid according to number of blockLength^2 squares needed.
  int gridLength=(n-1)/BLOCK_MAX3+1;
  // Set outputs.
  blockSize=dim3(blockLength*blockLength+4*blockLength,1);
  gridSize=dim3(gridLength,gridLength);
  return;
}

void allocateGridV3(char **G,int n){
  // Device memory only for performance.
  cudaError_t err;
  // 1 chunk of memory for better locality.
  err=cudaMalloc((void**)G,n*n*sizeof(char));
  if(err!=cudaSuccess){
    fprintf(stderr,"Error in cudaMalloc: %s\n",cudaGetErrorString(err));
  }
  // No row pointers for better memory access patterns.
  return;
}

__global__ void initializeRandomGridV3(char *G,int n, int threadBlockLength){
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
      G[i*n+j]=curand(&state)%2;
    }
  }
  return;
}

void evolveIsingGridV3(char *HostG,char *G, char *G0, int n, int k,
                       dim3 blockSize, dim3 gridSize,int blockLength){
  char *Gtemp;
  cudaError_t cudaError;
  // Calculate the dimensions that are used on the boundary blocks, and pass them.
  const int boundDimX=(gridSize.x*blockLength>n)?n-(gridSize.x-1)*blockLength:blockLength;
  const int boundDimY=(gridSize.y*blockLength>n)?n-(gridSize.y-1)*blockLength:blockLength;
  // Call a kernel for each iteration.
  for(int run_count=0;run_count<k;run_count++){
    // Send everything to the 0 stream for concurrency.
    generateNextGridStateV3<<<gridSize,blockSize,0>>>(G,G0,n,blockLength,boundDimX,boundDimY);
    cudaError=cudaGetLastError();
    if(cudaError!=cudaSuccess){
      fprintf(stderr,"Error in generateNextGridStateV3: %s\n",cudaGetErrorString(cudaError));
      exit(1);
    }
    // Swap pointers and pass by value for the next iteration.
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  // At each loop's end the original G contains the new state 
  // since it's passed by value so no need for additional swaps.
  
  // cudaMemcpy synchrinizes host with device.
  cudaMemcpy(HostG, G, n*n, cudaMemcpyDeviceToHost);
  return;
}

__global__ void generateNextGridStateV3(char *G,char *G0,int n,int blockLength,
                                        int boundDimX,int boundDimY){
  // These point to the (0,0) position of the block to be calculated. 
  const int blockX=blockIdx.x*blockLength;
  const int blockY=blockIdx.y*blockLength;
  // Due to thread overallocation find the actual dimensions of the 
  // block that will be calculated.
  // X is rows Y is columns.
  const int dimX=(blockIdx.x==gridDim.x-1)?boundDimX:blockLength;
  const int dimY=(blockIdx.y==gridDim.y-1)?boundDimY:blockLength;
  // Free extra threads if you are on the boundary. (Always on bunches).
  if(threadIdx.x>=dimX*dimY+2*(dimX+dimY)){
    return;
  }
  // Shared memory (the 4 corners of this block won't by used).
  __shared__ char Gs[(BLOCK_MAX3+2)*(BLOCK_MAX3+2)];
  // First 2 thread groups get the upper/lower rows, the rest get the mid section.
  switch(threadIdx.x/dimY){
    case 0:
      // Load first row (top left corner and top right corner are garbage)
      Gs[1+threadIdx.x]=G0[((n+blockX-1)%n)*n+blockY+threadIdx.x];
    break;
    case 1:
      // Load last row (bottom corners are garbage)     
      Gs[(dimY+2)*(dimX+1)+1+(threadIdx.x-dimY)]=G0[((blockX+dimX)%n)*n+
                                                    blockY+(threadIdx.x-dimY)];
    break;
    default:
      // Load the mid rows.
      Gs[(dimY+2)+(threadIdx.x-2*dimY)]=G0[(blockX+(threadIdx.x-2*dimY)/(dimY+2))*n
                                        +(n+blockY-1+(threadIdx.x-2*dimY)%(dimY+2))%n];
    break;
  }
  __syncthreads();
  // Free the threads that were here only for the boundary loads.
  if(threadIdx.x>=dimX*dimY){
    return;
  }
  // The rest just calculate based on the shared grid and store to G.
  G[(blockX+threadIdx.x/dimY)*n+blockY+threadIdx.x%dimY]=
    (Gs[(1+threadIdx.x/dimY)*(dimY+2)+(threadIdx.x%dimY)]+
    Gs[(1+threadIdx.x/dimY)*(dimY+2)+(1+threadIdx.x%dimY)]+
    Gs[(1+threadIdx.x/dimY)*(dimY+2)+(2+threadIdx.x%dimY)]+
    Gs[(threadIdx.x/dimY)*(dimY+2)+(1+threadIdx.x%dimY)]+
    Gs[(2+threadIdx.x/dimY)*(dimY+2)+(1+threadIdx.x%dimY)]>2)?1:0;
  return;
}

void freeGridV3(char *G){
  // If grid is not initialized do nothing.
  if(G==NULL){
    return;
  }
  // Free data chunk.
  cudaFree(G);
  return;
}
