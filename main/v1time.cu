/* Script that provides the execution times for v0 of the ising model. */
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <cuda_runtime.h>
#include "isingV1.h"

#define RUNS_PER_SIZE 3

int main(int argc, char** argv){
  if(argc!=2){
    printf("Usage: ./v0time [resultFolder]\n");
    exit(1);
  }
  // Get file 
  FILE* result_file;
  char *file_name=(char*)malloc(100*sizeof(char));
  snprintf(file_name,99,"%s/v1time.txt",argv[1]);
  result_file=fopen(file_name,"w");
  if(!result_file){
    printf("Error in opening result file..\n");
    exit(1);
  }

  // Init variables.
  char **G;
  char **G0;
  int n_min=200;
  int n_step=200;
  int n_max=2000;
  int k_min=20;
  int k_step=20;
  int k_max=100;
  float times[RUNS_PER_SIZE];
  float run_time;
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  dim3 blockSize,gridSize;
  cudaError_t cudaError;

  // For all sizes.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    gridAllocateV1(&G0,n);
    gridAllocateV1(&G,n);
    getDimensionsV1(n,blockSize,gridSize);
    // For all k until k_max
    for(int k=k_min;k<=k_max;k+=k_step){
      // Test many times..
      for(int run_count=0;run_count<RUNS_PER_SIZE;run_count++){
        // Run this..
        cudaEventRecord(start,0);
        initRandomV1<<<gridSize,blockSize>>>(G0,n);
        // Error check for initRandom..
        cudaError=cudaGetLastError();
        if(cudaError!=cudaSuccess){
          printf("Kernel failed at initRandom: %s\n",cudaGetErrorString(cudaError));
          exit(1);
        }
        cudaDeviceSynchronize();
        isingV1(G,G0,n,k,blockSize,gridSize);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&run_time,start,stop);
        // Reset events.
        cudaEventRecord(start,0);
        cudaEventRecord(stop,0);
        // Store time in buffer.
        times[run_count]=run_time;
      }
      // Get median for this pair (n,k):
      float median=0;
      int count;
      for(int i=0;i<RUNS_PER_SIZE;i++){
        count=0;
        for(int j=0;j<RUNS_PER_SIZE;j++){
          if(times[i]>=times[j]){
            count++;
          }
        }
        // Check if median:
        if(count==(RUNS_PER_SIZE/2)+1){
          median=times[i];
          break;
        }
      }
      // Median retrieved write to file:
      // [n] [k] [median]
      fprintf(result_file,"%d %d %f\n",n,k,median);
    }
    freeGridV1(G);
    freeGridV1(G0);
    printf("Size %d done.\n",n);
  }
  printf("V1 Timing gathered.\n");
  // Cleanup.
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  free(file_name);
  fclose(result_file);
  return 0; 
}
