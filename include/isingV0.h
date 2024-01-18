/* Sequential version of ising Model */
#ifndef ISING_V0 
#define ISING_V0

#define ANSI_COLOR_RED      "\x1b[31m"
#define ANSI_COLOR_GREEN    "\x1b[32m"
#define ANSI_COLOR_RESET    "\033[0m"

// Main algorithm, that performs k iterations on the n x n grid G0 and creates G.
int isingV0(char ***G, char** G0,int n, int k);

// Generates random initial 2-D grid of size n x n.
void initRandomV0(char ***G,int n);

// Prints given grid with colors.
void printGrid(char **G, int n);

// Free grid. 
void freeGrid(char **G);

// Clears screen.
void clearScreen();
#endif
