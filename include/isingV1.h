/* Functions for v1 of the ising model (1 thread per moment) */ 
#ifndef ISING_V1
#define ISING_V1

#define BLOCK_MAX 16

// Define dimensions of the grid structure based on n. 
void getDimensionsV1(int n, dim3 &blockSize, dim3 &gridSize);

// Allocation is not global so its separate. 
void gridAllocateV1(char ***G,int n);

// Generate random grid based on the dimensions given.
__global__ void initRandomV1(char **G,int n);

// Main algorithm.
void isingV1(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize);

// Main algorithm.
__global__ void isingV11(char **G, char **G0, int n, int k);

// Generate next state.
__global__ void nextStateV1(char **G,char **G0,int n);

// Free grid 
void freeGridV1(char **G);

#endif
