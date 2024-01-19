/* Sequential version of ising Model */
#ifndef ISING_V0 
#define ISING_V0

#define ANSI_COLOR_RED      "\x1b[31m"
#define ANSI_COLOR_GREEN    "\x1b[32m"
#define ANSI_COLOR_RESET    "\033[0m"

// Allocate grid. 
void gridAllocateV0(char ***G,int n);

// Generates random initial 2-D grid of size n x n.
void initRandomV0(char ***G,int n);

// Main algorithm, that performs k iterations on the n x n grid G0 and creates G.
void isingV0(char **G, char** G0,int n, int k);

// Free grid. 
void freeGridV0(char **G);

// Prints given grid with colors.
void printGrid(char **G, int n);

// Clears screen.
void clearScreen();
#endif
