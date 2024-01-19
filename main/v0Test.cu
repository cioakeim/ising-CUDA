#include <stdio.h>
#include <stdlib.h>
#include "isingV0.h"

int main(){
  int n=10;
  int k=10;
  char **G;
  char **G0;
  gridAllocateV0(&G,n);
  gridAllocateV0(&G0,n);

  initRandomV0(&G0,n);

  isingV0(G, G0, n, k); 

  freeGridV0(G);
  freeGridV0(G0);
  return 0;
}
