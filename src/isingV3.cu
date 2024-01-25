/* V3: Memory sharing.
*/
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "isingV3.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

// Define dimensions of the grid structure based on n. 
// Same as V2.
void getInitDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength){
  // Threads per dimension so that all n are covered (overallocating for imperfect fits).
  const int threadsPerDimension=(n/threadBlockLength)+(n%threadBlockLength>0);
  // Block and grid size according to BLOCK_MAX (like v1).
  const int blockLength=(threadsPerDimension<BLOCK_MAX)?(threadsPerDimension):(BLOCK_MAX);
  const int gridLength=(threadsPerDimension-1)/BLOCK_MAX+1;
  blockSize=dim3(blockLength,blockLength);
  gridSize=dim3(gridLength,gridLength);
  return;
}

void getIterDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int &blockLength){
  // Each gridBlock calculates a BLOCK_MAX x BLOCK_MAX portion of the total grid.
  // At each block, 4*BLOCK_MAX threads are added to load the perimeter.
  blockLength=(n<BLOCK_MAX)?n:BLOCK_MAX;
  int gridLength=(n-1)/BLOCK_MAX+1;

  blockSize=dim3(blockLength*blockLength+4*blockLength,1);
  gridSize=dim3(gridLength,gridLength);
  return;
}

// Allocation is not global so its separate. 
// Same as V2.
void gridAllocateV3(char **G,int n){
  cudaError_t err;
  err=cudaMalloc((void**)G,n*n*sizeof(char));
  if(err!=cudaSuccess){
    fprintf(stderr,"Error in cudaMalloc: %s\n",cudaGetErrorString(err));
  }
  return;
}

// Generate random grid based on the dimensions given.
// Same as V2.
__global__ void initRandomV3(char *G,int n, int threadBlockLength){
  // Each thread responsible for initializing its own block.
  // BlockI,J point to the starting position of the thread's block..
  const int blockI=threadBlockLength*(blockIdx.x*blockDim.x+threadIdx.x);
  const int blockJ=threadBlockLength*(blockIdx.y*blockDim.y+threadIdx.y);
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
      G[i*n+j]=curand(&state)%2;
      //G[i*n+j]=100*i+j;
    }
  }
  return;
}

void isingV3(char *HostG,char *G, char *G0, int n, int k, dim3 blockSize, dim3 gridSize,int blockLength){
  char *Gtemp;
  cudaError_t cudaError;
  // Calculate the dimensions that are used on the last blocks, and pass them.
  const int boundDimX=(gridSize.x*blockLength>n)?n-(gridSize.x-1)*blockLength:blockLength;
  const int boundDimY=(gridSize.y*blockLength>n)?n-(gridSize.y-1)*blockLength:blockLength;

  printf("BoundX: %d, BoundY: %d\n",boundDimX,boundDimY);
  printf("blockLength: %d\n",blockLength);
  printf("Grid: %d x %d\n",gridSize.x,gridSize.y);

  

  for(int run_count=0;run_count<k;run_count++){
    nextStateV3<<<gridSize,blockSize,0>>>(G,G0,n,blockLength,boundDimX,boundDimY);
    cudaError=cudaGetLastError();
    if(cudaError!=cudaSuccess){
      fprintf(stderr,"Error in nextStateV3: %s\n",cudaGetErrorString(cudaError));
      exit(1);
    }
    //cudaDeviceSynchronize();
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  Gtemp=G;
  G=G0;
  G0=Gtemp;
  cudaMemcpy(HostG, G, n*n, cudaMemcpyDeviceToHost);
  return;
}

// blockLength is length of the blockLength x blockLength block that will be calculated.
__global__ void nextStateV3(char *G,char *G0,int n,int blockLength,int boundDimX,int boundDimY){
  // These point to the (0,0) position of the block to be calculated. 
  const int blockX=blockIdx.x*blockLength;
  const int blockY=blockIdx.y*blockLength;
  // Due to thread overallocation find the actual dimensions of the 
  // portion this block is calculating.
  // X is rows Y is columns.
  const int dimX=(blockIdx.x==gridDim.x-1)?boundDimX:blockLength;
  const int dimY=(blockIdx.y==gridDim.y-1)?boundDimY:blockLength;
  // Free extra threads.
  if(threadIdx.x>=dimX*dimY+2*(dimX+dimY)){
    return;
  }
  // Shared memory 
  __shared__ char Gs[(BLOCK_MAX+2)*(BLOCK_MAX+2)];
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
  //G[(blockX+threadIdx.x/dimY)*n+blockY+threadIdx.x%dimY]=
  //  Gs[(1+threadIdx.x/dimY)*(dimY+2)+(1+threadIdx.x%dimY)]+1;
  return;
}













// Free grid 
void freeGridV3(char *G){
  if(G==NULL){
    return;
  }
  cudaFree(G);
  G=NULL;
  return;
}
