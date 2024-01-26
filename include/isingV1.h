/* V1: One thread per moment version the implementation.
 * This module contains all the methods that are used in the computation of the evolution of an 
 * Ising model over k iterations on a 2-D n by n grid. The implementation uses CUDA and it's a 
 * modification of isingV0, so the same principles applied. The visuals are excluded.
 *
 * In this implementation, at each iteration, each thread is assigned to compute the result 
 * of a single moment of the grid.
 */
#ifndef ISING_V1
#define ISING_V1

// The maximum length of the square thread block in the implementation.
#define BLOCK_MAX1 16

/* This function calculates the grid and block dimensions for the following methods. 
 * If the grid length is not a multiple of BLOCK_MAX1 and larger that BLOCK_MAX1, 
 * then there is overallocation of threads due to the desire to keep the block size fixed 
 * for all the grid blocks for the procedures.
 * Inputs: 
 *  n (int): Length of the square grid. 
 * Outputs:
 *  blockSize (dim3): The dimensions of the thread blocks.
 *  gridSize (dim3): The dimensions of the grid.
 */
void getDimensionsV1(int n, dim3 &blockSize, dim3 &gridSize);

/* This method allocates the data needed for the 2-D grid and sets up the 
 * row pointers. All memory allocated is unified memory.
 * Input:
 *  n (int): Length of the square grid. 
 * Output:
 *  G (char**): 2-D grid object.
 */
void allocateGridV1(char ***G,int n);

/* Creates a random state of the grid.
 * Each thread creates a random boolean for its assigned position.
 * Inputs:
 *  n (int): Length of the square grid.
 *  G (char**): 2-D grid object.
 */
__global__ void initializeRandomGridV1(char **G,int n);

/* Produces the evolution of an Ising Model of a 2-D grid 
 * for a number of steps starting from a given initial state.
 * This method's main job is to call kernels multiple times to generate new states.
 * Inputs:
 *   G0 (char**): Inital state of the model.
 *   n (int): Length of the square grid.
 *   k (int): Number of evolution steps.
 *   blockSize (dim3): The thread block dimensions passed to the kernel calls.
 *   gridSize (dim3): The grid dimensions passed to the kernel calls.
 * Output:
 *   G (char**): Resulting grid after the evolution.
 *   (G is allocated by the callee before the call)
 */
void evolveIsingGridV1(char **G, char **G0, int n, int k, dim3 blockSize, dim3 gridSize);

/* This function creates the next state from a given previous state.
 * Each thread computes one grid element separately.
 * Inputs:
 *  G0 (char**): Inital 2-D grid state.
 *  n (int): Length of square grid.
 * Output:
 *  G (char**): New grid state.
 */
__global__ void generateNextGridStateV1(char **G,char **G0,int n);

/* Frees the 2-D grid passed. The grid is unified memory.
 * Use in pair with allocateGridV1.
 * Input:
 *  G (char**): 2-D grid to be freed.
 */
void freeGridV1(char **G);

#endif
