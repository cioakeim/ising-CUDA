/* V2: One thread per block of momenets version of the implementation.
 * This module contains all the methods that are used in the computation of the evolution of an Ising 
 * model over k iterations on a 2-D n by n grid. In this CUDA implementation, more work is assigned 
 * per thread so that each thread computes a block of moments, instead of just one. The block per 
 * thread was chosen to be square since it's a more compact shape.
 */
#ifndef ISING_V2
#define ISING_V2

// The maximum length of the square block of threads in the implementation.
#define BLOCK_MAX2 16

/* This function calculates the grid and block dimensions for the following methods.
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
void getDimensionsV2(int n, dim3 &blockSize, dim3 &gridSize, int threadBlockLength);

/* This method allocates the data needed for the 2-D grid and sets up the 
 * row pointers. All memory allocated is unified memory.
 * Input:
 *  n (int): Length of the square grid. 
 * Output:
 *  G (char**): 2-D grid object.
 */
void allocateGridV2(char ***G,int n);

/* Creates a random state of the grid.
 * Each thread creates a block of booleans according to it's assigned work load.
 * Inputs:
 *  n (int): Length of the square grid.
 *  threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *  to compute.
 *  G (char**): 2-D grid object.
 */
__global__ void initializeRandomGridV2(char **G,int n, int threadBlockLength);

/* Produces the evolution of an Ising Model of a 2-D grid 
 * for a number of steps starting from a given initial state.
 * This method's main job is to call kernels multiple times to generate new states.
 * Inputs:
 *   G0 (char**): Inital state of the model.
 *   n (int): Length of the square grid.
 *   k (int): Number of evolution steps.
 *   threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *   to compute.
 *   blockSize (dim3): The thread block dimensions passed to the kernel calls.
 *   gridSize (dim3): The grid dimensions passed to the kernel calls.
 * Output:
 *   G (char**): Resulting grid after the evolution.
 *   (G is allocated by the callee before the call)
 */
void evolveIsingGridV2(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize, 
                       int threadBlockLength);

/* This function creates the next state from a given previous state.
 * Each thread computes one block of elements independently.
 * Inputs:
 *  G0 (char**): Inital 2-D grid state.
 *  n (int): Length of square grid.
 *  threadBlockLength (int): The length (c) of the cxc square that each thread is assigned 
 *  to compute.
 * Output:
 *  G (char**): New grid state.
 */
__global__ void generateNextGridStateV2(char **G,char **G0,int n, int threadBlockLength);

/* Frees the 2-D grid passed. The grid is unified memory.
 * Use in pair with allocateGridV2.
 * Input:
 *  G (char**): 2-D grid to be freed.
 */
void freeGridV2(char **G);

#endif
