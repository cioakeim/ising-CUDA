/* V3: Multiple threads sharing common input moments.
 * This module contains all the methods that are usedin the computation of the evolution of an Ising 
 * model over k iterations on a 2-D n by n grid. This CUDA implementation improves upon the other 
 * two versions with the use of shared memory. Each thread's role is to computate 1 moment, just like
 * V1, but the use of shared memory allows less reads from the global memory.
 */
#ifndef ISING_V3
#define ISING_V3


// The maximum length of the square block that is computed by a thread block.
#define BLOCK_MAX3 16

/* The same as getDimensionsV2.
 *
 * This function calculates the grid and block dimensions for the following methods.
 * The procedure is similar to that of V1, but instead of n, the measurement for the need 
 * of threads is the amount of blocks that can cover n.
 * Like the previous version, if the sizes don't match there is thread overallocation.
 * Inputs:
 *  n (int): Length of the square grid. 
 *  threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *  to compute.
 * Outputs:
 *  blockSize (dim3): The dimensions of the thread blocks.
 *  gridSize (dim3): The dimensions of the grid.
 */
void getInitializationDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength);

/* This function calculates the grid and block dimensions for the evolution stage of the 
 * simulation. The difference between V1 is that now the block size is flattened to 1-D for better 
 * warp management. For each block to be computed, one thread is allocated for the computation of 
 * each element and on top of that there are threads allocated for the fetch of the boundaries of the 
 * block.
 * Inputs:
 *  n (int): Length of the square grid.
 * Outputs:
 *  blockSize (dim3): The dimensions of the thread blocks.
 *  gridSize (dim3): The dimensions of the grid.
 *  blockLength (int): The length of the square block that is computed at each thread block.
 */
void getEvolutionDimensionsV3(int n, dim3 &blockSize, dim3 &gridSize,int &blockLength);

/* This method allocates the data needed for the 2-D grid without setting up the 
 * row pointers. All handling is done in 1-D.
 * For performance purposes, the grid is allocated on device only.
 * Input:
 *  n (int): Length of the square grid. 
 * Output:
 *  G (char**): 1-D grid object.
 */
void allocateGridV3(char **G,int n);

/* Same as initializeRandomGridV2 with different indexing.
 *
 * Creates a random state of the grid.
 * Each thread creates a block of booleans according to it's assigned work load.
 * Inputs:
 *  n (int): Length of the square grid.
 *  threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *  to compute.
 *  G (char*): 1-D grid object.
 */
__global__ void initializeRandomGridV3(char *G,int n, int threadBlockLength);

/* Produces the evolution of an Ising Model of a 2-D grid 
 * for a number of steps starting from a given initial state.
 * This method's main job is to call kernels multiple times to generate new states.
 * After the computation, the result is copied to host memory.
 * Inputs:
 *   G0 (char*): Inital state of the model. (Device)
 *   G (char*): Resulting grid after the evolution. (Device)
 *   n (int): Length of the square grid.
 *   k (int): Number of evolution steps.
 *   threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *   to compute.
 *   blockSize (dim3): The thread block dimensions passed to the kernel calls.
 *   gridSize (dim3): The grid dimensions passed to the kernel calls.
 * Output:
 *   HostG (char*): The host copy of the resulting grid.
 */
void evolveIsingGridV3(char *HostG,char *G, char *G0, int n, int k,
                       dim3 blockSize, dim3 gridSize,int blockLength);

/* This function creates the next state from a given previous state.
 * The computation is done in 2 stages. First all the needed elements are 
 * loaded to shared memory. Then (after the spare threads are freed), each 
 * remaining thread performs the computation reading only from shared memory.
 * Inputs:
 *  G0 (char*): Inital 2-D grid state. (Device)
 *  n (int): Length of square grid.
 *  blockLength (int): The max length at each dimension of the resulting block.
 *  boundDimX (int): The dimensions on the X-axis if the currect block is on the 
 *  edge of the grid (due to overallocation)
 *  boundDimY (int): The dimensions on the Y-axis if the currect block is on the 
 *  edge of the grid (due to overallocation)
 * Output:
 *  G (char*): New grid state. (Device)
 */
__global__ void generateNextGridStateV3(char *G,char *G0,int n,int blockLength,
                                        int boundDimX,int boundDimY);

/* Frees the 1-D grid passed. The grid is device memory.
 * Use in pair with allocateGridV3.
 * Input:
 *  G (char*): 1-D grid to be freed.
 */
void freeGridV3(char *G);

#endif
