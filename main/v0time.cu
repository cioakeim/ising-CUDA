/* Script that provides the execution times for v0 of the ising model. */
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>
#include "isingV0.h"

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
  int n_min=400;
  int n_step=400;
  int n_max=4000;
  int k_min=20;
  int k_step=20;
  int k_max=100;
  float init_times[RUNS_PER_SIZE];
  float iter_times[RUNS_PER_SIZE];
  float init_time,iter_time;
  cudaEvent_t start,mid,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&mid);
  cudaEventCreate(&stop);

  // For all sizes until n_max 
  for(int n=n_min;n<=n_max;n+=n_step){
    gridAllocateV0(&G0,n);
    gridAllocateV0(&G,n);
    // For all k until k_max
    for(int k=k_min;k<=k_max;k+=k_step){
      // Test this many times 
      for(int run_count=0;run_count<RUNS_PER_SIZE;run_count++){
        // Run the program:
        cudaEventRecord(start,0);
        initRandomV0(&G0,n);
        cudaEventRecord(mid,0);
        isingV0(G, G0, n, k);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(mid);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&init_time,start,mid);
        cudaEventElapsedTime(&iter_time,mid,stop);
        // Reset events.
        cudaEventRecord(start,0);
        cudaEventRecord(mid,0);
        cudaEventRecord(stop,0);
        // Store time in buffer.
        init_times[run_count]=init_time;
        iter_times[run_count]=iter_time;
      }
      // Experiment for (n,k) pair is done: Get median 
      // Ineffiecient code but small size so I don't care.
      float iter_median=0;
      float init_median=0;
      int init_count,iter_count;
      for(int i=0;i<RUNS_PER_SIZE;i++){
        init_count=0;
        iter_count=0;
        for(int j=0;j<RUNS_PER_SIZE;j++){
          if(init_times[i]>=init_times[j]){
            init_count++;
          }
          if(iter_times[i]>=iter_times[j]){
            iter_count++;
          }
        }
        // Check if median:
        if(init_count==(RUNS_PER_SIZE/2)+1){
          init_median=init_times[i];
        }
        if(iter_count==(RUNS_PER_SIZE/2)+1){
          iter_median=iter_times[i];
        }
      }
      // Median retrieved write to file:
      // [n] [k] [median]
      fprintf(result_file,"%d %d %f %f\n",n,k,init_median,iter_median);
    }
    // Free grids for next size try.
    freeGridV0(G);
    freeGridV0(G0);
  }
  printf("V0 Timing done.\n");
  // Cleanup.
  cudaEventDestroy(start);
  cudaEventDestroy(mid);
  cudaEventDestroy(stop);
  free(file_name);
  fclose(result_file);
  return 0; 
}
