/* V0: Sequential version of the ising Model */ 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <isingV0.h>

int isingV0(char ***Gfinal, char** G0,int n, int k){
  if(k<0){
    printf("k must be a positive integer.\n");
    exit(1);
  }
  // Init 2nd grid. 
  *Gfinal=(char**)malloc(n*sizeof(char*));
  char *data=(char*)malloc(n*n*sizeof(char));
  for(int i=0;i<n;i++){
    (*Gfinal)[i]=data+i*n;
  }
  // Rename for easier use.
  char **G=*Gfinal;
  char temp;
  char **Gtemp;
  for(int run_count=0;run_count<k;run_count++){
    // Create.
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
        temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
        G[i][j]=(temp>2)?(1):(0);
      }
    }
    // Swap.
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  // G0 contains final state so final swap: 
  Gtemp=G;
  G=G0;
  G0=Gtemp;
  return 0;
}

void initRandomV0(char ***G,int n){
  // Allocate memory:
  *G=(char**)malloc(n*sizeof(char*));
  char *data=(char*)malloc(n*n*sizeof(char));
  for(int i=0;i<n;i++){
    (*G)[i]=data+i*n;
  }
  // Random state: 
  srand(time(NULL));
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      (*G)[i][j]=rand()%2;
    }
  }
}

void printGrid(char **G, int n){
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      if(G[i][j]==1){
        printf(ANSI_COLOR_GREEN"%d " ANSI_COLOR_GREEN,G[i][j]);
      }
      else{
        printf(ANSI_COLOR_RED"%d " ANSI_COLOR_RED,G[i][j]);
      }
    }
    printf(ANSI_COLOR_RESET"\n" ANSI_COLOR_RESET);
  }
  printf("\n\n");
}

void freeGrid(char **G){
  if(G==NULL){
    return;
  }
  free(G[0]);
  free(G);
  G=NULL;
}

void clearScreen() {
    printf("\033[2J");  // ANSI escape code to clear the screen
    printf("\033[H");   // Move the cursor to the home position (top-left corner)
}
