/* Simple script that produces the result of an Ising model 
 * of an n by n grid for k steps. Both n and k are passed from the terminal.
 */
#include <stdio.h>
#include <stdlib.h>
#include "isingV0.h"

int main(int argc, char **argv){
  if(argc!=3){
    printf("Usage: ./v0test [n] [k]\n");
    exit(1);
  }
  int n=atoi(argv[1]);
  int k=atoi(argv[2]);
  char **G;
  char **G0;
  allocateGridV0(&G,n);
  allocateGridV0(&G0,n);
  initializeRandomGridV0(G0,n);
  evolveIsingGridV0(G, G0, n, k,false); 
  printf("Resulting grid:\n");
  printGrid(G, n);
  freeGridV0(G);
  freeGridV0(G0);
  return 0;
}
