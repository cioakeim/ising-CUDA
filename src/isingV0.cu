/* V0: Sequential version of the Ising Model. */ 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "isingV0.h"

void allocateGridV0(char ***G,int n){
  // Row pointers allocation.
  (*G)=(char**)malloc(n*sizeof(char*));
  // Whole data block allocation (together for locality).
  char *data=(char*)malloc(n*n*sizeof(char));
  for(int i=0;i<n;i++){
    (*G)[i]=data+i*n;
  }
  return;
}

void initializeRandomGridV0(char **G,int n){
  // Each element is either 0 or 1.
  srand(time(NULL));
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      G[i][j]=rand()%2;
    }
  }
}

void evolveIsingGridV0(char **G, char** G0,int n, int k,bool visual){
  if(k<0){
    printf("Error in evolveIsingGridV0: k must be a positive integer.\n");
    exit(1);
  }
  char temp;
  char **Gtemp;
  // Run the evolution k times.
  for(int run_count=0;run_count<k;run_count++){
    // For the visual print and wait for input.
    if(visual){
      clearScreen();
      printf("State %d:\n",run_count);
      printGrid(G0,n);
      printf("Press enter for the next iteration...\n");
      getchar();
    }
    // Generate the grid with the majority rule.
    for(int i=0;i<n;i++){
      for(int j=0;j<n;j++){
        temp=G0[i][j]+G0[i][(n+j+1)%n]+G0[i][(n+j-1)%n]+G0[(n+i+1)%n][j]+G0[(n+i-1)%n][j];
        G[i][j]=(temp>2)?(1):(0);
      }
    }
    // The new state is now the old state.
    Gtemp=G;
    G=G0;
    G0=Gtemp;
  }
  // At each loop's end the original G contains the new state 
  // since it's passed by value so no need for additional swaps.
  return;
}

void freeGridV0(char **G){
  // If G is already free don't do anything.
  if(G==NULL){
    return;
  }
  // Free the data chunk.
  free(G[0]);
  // Free the row pointers.
  free(G);
  return;
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

void clearScreen(){
  printf("\033[2J");  // ANSI escape code to clear the screen
  printf("\033[H");   // Move the cursor to the home position (top-left corner)
}
