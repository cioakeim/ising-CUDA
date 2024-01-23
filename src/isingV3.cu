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
void getDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength){
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
// Same as V2.
void gridAllocateV3(char ***G,int n){
  char *data;
  cudaMallocManaged(&data,n*n*sizeof(char));
  cudaMallocManaged(G,n*sizeof(char*));
  for(int i=0;i<n;i++){
    (*G)[i]=data+n*i;
  }
  return;
}

// Generate random grid based on the dimensions given.
// Same as V2.
__global__ void initRandomV3(char **G,int n, int threadBlockLength){
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

// Same as V2.
void isingV3(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize){
  char **Gtemp;
  cudaError_t cudaError;
  for(int run_count=0;run_count<k;run_count++){
    nextStateV3<<<gridSize,blockSize>>>(G,G0,n);
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
  return;
}

__global__ void nextStateV3(char **G,char **G0,int n){
  // These are for the whole grid.
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  // These are local.
  int ii=threadIdx.x+1;
  int jj=threadIdx.y+1;
  // This memory will be used from the whole block.
  // Extra 2 rows on each dimension are for the boundaries from other blocks.
  __shared__ char locG[BLOCK_MAX+2][BLOCK_MAX+2];
  // Overallocated threads are useless besides setting the bounds.
  if(i>=n || j>=n){
    if(i==n && j<n){
      locG[ii][jj]=G0[0][j];
    }
    if(i<n && j==n){
      locG[ii][jj]=G0[i][0];
    }
    return;
  }
  // Each thread retrieves its moment to the sharedBlock
  // ThreadIdx is the reletive id for the block. (locG's bounds are for the neighboring elements)
  locG[ii][jj]=G0[i][j];
  // The ones that need the boundaries bring them too.
  if(ii==1 || ii==blockDim.x){
    int boundDir=-(ii==1)+(ii==blockDim.x);
    locG[ii+boundDir][jj]=G0[(n+i+boundDir)%n][j];
  }
  if(jj==1 || jj==blockDim.y){
    int boundDir=-(jj==1)+(jj==blockDim.y);
    locG[ii][jj+boundDir]=G0[i][(n+j+boundDir)%n];
  }
  __syncthreads();
  // Now the block is in the shared memory.
  G[i][j]=(locG[ii][jj]+locG[ii][jj+1]+locG[ii][jj-1]+locG[ii+1][jj]+locG[ii-1][jj]>2)?(1):(0);
  return;
}













// Free grid 
void freeGridV3(char **G){
  if(G==NULL){
    return;
  }
  cudaFree(G[0]);
  cudaFree(G);
  G=NULL;
  return;
}
