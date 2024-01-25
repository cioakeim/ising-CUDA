/* Functions for v3 of the ising model (block of threads sharing memory) */ 
#ifndef ISING_V3
#define ISING_V3

#define BLOCK_MAX 16

// Define dimensions of the grid structure based on n. 
// threadBLockLength is used only for initial state.
void getInitDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength);

void getIterDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize,int &blockLength);

// Allocation is not global so its separate. 
void gridAllocateV3(char **G,int n);

// Generate random grid based on the dimensions given.
__global__ void initRandomV3(char *G,int n, int threadBlockLength);

// Main algorithm.
void isingV3(char *HostG,char *G, char *G0, int n, int k, dim3 blockSize, dim3 gridSize,int blockLength);

// Generate next state.
__global__ void nextStateV3(char *G,char *G0,int n,int blockLength,int boundDimX,int boundDimY);

// Free grid are consequtive kernel calls sequentially executed? cuda
void freeGridV3(char *G);

#endif
