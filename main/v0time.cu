/* Script that calculates the median execution times for the random state initialization and 
 * the evolution the Ising model with the V1 implementation. Each (n,k) pair is run multiple 
 * times and the median value of both processes execution is stored in the location specified 
 * from the terminal call.
 */
#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>

#include "isingV0.h"

// Each (n,k) pair is run this many times.
#define RUNS_PER_SIZE 1

int main(int argc, char** argv){
  if(argc!=2){
    printf("Usage: ./v0time [resultFolder]\n");
    exit(1);
  }
  // Get file 
  FILE* result_file;
  char *file_name=(char*)malloc(100*sizeof(char));
  snprintf(file_name,99,"%s/v0time.txt",argv[1]);
  result_file=fopen(file_name,"w");

  // Init variables.
  char **G;
  char **G0;
  // Range of length.
  int n_min=5000;
  int n_step=10000;
  int n_max=25000;
  // Range of iteration steps.
  int k_min=20;
  int k_step=40;
  int k_max=60;
  // For the median calculation. (Time is in ms)
  float init_times_ms[RUNS_PER_SIZE];
  float iter_times_ms[RUNS_PER_SIZE];
  float init_time_ms,iter_time_ms;
  cudaEvent_t start,mid,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&mid);
  cudaEventCreate(&stop);

  // For all sizes..
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate needed space..
    allocateGridV0(&G0,n);
    allocateGridV0(&G,n);
    // For all k..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Test many times..
      for(int run_count=0;run_count<RUNS_PER_SIZE;run_count++){
        // Start counting..
        cudaEventRecord(start,0);
        initializeRandomGridV0(G0,n);
        // Record between time..
        cudaEventRecord(mid,0);
        // Evolution..
        evolveIsingGridV0(G,G0,n,k,false);
        cudaEventRecord(stop,0);
        // Timings gathered..
        cudaEventSynchronize(mid);
        cudaEventSynchronize(stop);
        // Get time intervals..
        cudaEventElapsedTime(&init_time_ms,start,mid);
        cudaEventElapsedTime(&iter_time_ms,mid,stop);
        // Store time in buffer.
        init_times_ms[run_count]=init_time_ms;
        iter_times_ms[run_count]=iter_time_ms;
      }
      // Get median for this pair (n,k):
      // (Algorithm is inefficient af but the size is small so I don't care)
      float iter_median_ms=0;
      float init_median_ms=0;
      int init_count,iter_count;
      for(int i=0;i<RUNS_PER_SIZE;i++){
        init_count=0;
        iter_count=0;
        for(int j=0;j<RUNS_PER_SIZE;j++){
          if(init_times_ms[i]>=init_times_ms[j]){
            init_count++;
          }
          if(iter_times_ms[i]>=iter_times_ms[j]){
            iter_count++;
          }
        }
        // Check if median:
        if(init_count==(RUNS_PER_SIZE/2)+1){
          init_median_ms=init_times_ms[i];
        }
        if(iter_count==(RUNS_PER_SIZE/2)+1){
          iter_median_ms=iter_times_ms[i];
        }
      }
      // Median retrieved write to file:
      // [n] [k] [init_median] [iter_median]
      fprintf(result_file,"%d %d %f %f\n",n,k,init_median_ms,iter_median_ms);
    }
    // Free grids for next use.
    freeGridV0(G);
    freeGridV0(G0);
    printf("Size %d done.\n",n);
  }
  printf("Job done. V0 times gathered.\n");
  // Cleanup.
  cudaEventDestroy(start);
  cudaEventDestroy(mid);
  cudaEventDestroy(stop);
  free(file_name);
  fclose(result_file);
  return 0; 
}
