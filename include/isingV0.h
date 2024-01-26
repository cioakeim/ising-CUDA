/* V0: Sequential version the evolution of an Ising Model.
 * This module contains all the methods that are used in the computation of the evolution of an 
 * Ising model over k iterations, on a 2-D n by n grid. Each grid represents the spin of a magnetic
 * dipole and its values are either 0 (negative spin) of 1 (positive spin).Methods are split so that 
 * there is an element of freedom for the simulation. 
 *
 * The two main methods are initiateRandomGrid and evolveIsingGridV0.
 *
 * Aside from the simulation there are methods that allow a visual demostration of the Ising model 
 * evolution with colored prints which are used on the visual script.
 */
#ifndef ISING_V0 
#define ISING_V0

#include <stdbool.h>

#define ANSI_COLOR_RED      "\x1b[31m"
#define ANSI_COLOR_GREEN    "\x1b[32m"
#define ANSI_COLOR_RESET    "\033[0m"

/* Method that allocated an n by n grid and links it to the 
 * pointer G. The grid is stored in row major order.
 * Inputs:
 *   n (int): Length of the square grid.
 * Outputs:
 *   G (char**): Pointer to the 2-D grid allocated.
 */
void allocateGridV0(char ***G,int n);

/* Creates a random state of the grid.
 * Inputs:
 *  n (int): Length of the square grid.
 *  G (char**): 2-D grid object.
 */
void initializeRandomGridV0(char **G,int n);

/* Produces the evolution of an Ising Model of a 2-D grid 
 * for a number of steps starting from a given initial state.
 * Inputs:
 *   G0 (char**): Inital state of the model.
 *   n (int): Length of the square grid.
 *   k (int): Number of evolution steps.
 *   visual (bool): If true the evolution is shown step by step.
 * Output:
 *   G (char**): Resulting grid after the evolution.
 *   (G is allocated by the callee before the call)
 */
void evolveIsingGridV0(char **G, char** G0,int n, int k,bool visual);

/* Frees the 2-D grid passed.
 * Input:
 *  G (char**): 2-D grid to be freed.
 */
void freeGridV0(char **G);

/* Prints the given 2-D grid with ANSI colours.
 * Ones are coloured green and zeros are red.
 *  Inputs:
 *    G (char**): Pointer to 2-D grid.
 *    n (int): Length of the square grid.
 */
void printGrid(char **G, int n);

/* Simple function that clears the screen for a 
 * visual display of the model evolution.
 */
void clearScreen();
#endif
