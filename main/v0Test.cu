#include <stdio.h>
#include <stdlib.h>
#include "isingV0.h"

int main(){
  int n=10;
  int k=10;
  char **G;
  char **G0;
  initRandomV0(&G0,n);
  int ok=isingV0(&G, G0, n, k); 
  freeGrid(G);
  freeGrid(G0);
  return 0;
}
