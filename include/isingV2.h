/* Functions for v2 of the ising model (1 block of threads per moment)
* Each thread is responsible for a c by c block.
*/
#ifndef ISING_V2
#define ISING_V2

#define BLOCK_MAX 16

// Define dimensions of the grid structure based on n. 
void getDimensionsV2(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength);

// Allocation is not global so its separate. 
void gridAllocateV2(char ***G,int n);

// Generate random grid based on the dimensions given.
__global__ void initRandomV2(char **G,int n, int threadBlockLength);

// Main algorithm.
void isingV2(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize, int threadBlockLength);

// Generate next state.
__global__ void nextStateV2(char **G,char **G0,int n, int threadBlockLength);

// Free grid 
void freeGridV2(char **G);

#endif
