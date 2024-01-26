/* Script that calculates the median execution times for the random state initialization and 
 * the evolution the Ising model with the V2 implementation. Each (n,k) pair is run multiple 
 * times and the median value of both processes execution is stored in the location specified 
 * from the terminal call. Also from the call, the work block length of each thread is specified.
 */
#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>

#include "isingV2.h"

// Each (n,k) pairs is run this many times.
#define RUNS_PER_SIZE 5

int main(int argc, char** argv){
  if(argc!=3){
    printf("Usage: ./v2time [resultFolder] [threadBlockLength]\n");
    exit(1);
  }
  // Get file 
  FILE* result_file;
  char *file_name=(char*)malloc(100*sizeof(char));
  snprintf(file_name,99,"%s/v2time_%s.txt",argv[1],argv[2]);
  result_file=fopen(file_name,"w");
  if(!result_file){
    printf("Error in opening result file..\n");
    exit(1);
  }
  // Init variables.
  char **G;
  char **G0;
  // Range of length.
  int n_min=5000;
  int n_step=10000;
  int n_max=35000;
  // Range of iteration steps.
  int k_min=30;
  int k_step=50;
  int k_max=80;
  // For the median calculation. (Time is in ms)
  float init_times_ms[RUNS_PER_SIZE];
  float iter_times_ms[RUNS_PER_SIZE];
  float init_time_ms,iter_time_ms;
  cudaEvent_t start,stop,mid;
  cudaEventCreate(&start);
  cudaEventCreate(&mid);
  cudaEventCreate(&stop);
  // For kernel calls.
  dim3 blockSize,gridSize;
  // For error checking.
  cudaError_t cudaError;
  int threadBlockLength=atoi(argv[2]);
  printf("threadBlockLength: %d\n",threadBlockLength);
  // For all sizes.. 
  for(int n=n_min;n<=n_max;n+=n_step){
    // Allocate needed space.
    allocateGridV2(&G0,n);
    allocateGridV2(&G,n);
    getDimensionsV2(n,blockSize,gridSize,threadBlockLength);
    // For all k..
    for(int k=k_min;k<=k_max;k+=k_step){
      // Test many times..
      for(int run_count=0;run_count<RUNS_PER_SIZE;run_count++){
        // Start counting..
        cudaEventRecord(start,0);
        initializeRandomGridV2<<<gridSize,blockSize>>>(G0,n,threadBlockLength);
        cudaDeviceSynchronize();
        // Error check for random state..
        cudaError=cudaGetLastError();
        if(cudaError!=cudaSuccess){
          printf("Kernel failed at initializeRandomGridV2: %s\n",cudaGetErrorString(cudaError));
          exit(1);
        }
        // Record between time..
        cudaEventRecord(mid,0);
        // Evolution..
        evolveIsingGridV2(G,G0,n,k,blockSize,gridSize,threadBlockLength);
        cudaEventRecord(stop,0);
        // Timings gathered..
        cudaEventSynchronize(mid);
        cudaEventSynchronize(stop);
        // Get time intervals.
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
    // Free for next use.
    freeGridV2(G);
    freeGridV2(G0);
    printf("Size %d done.\n",n);
  }
  printf("Job done. V2 times gathered.\n");
  // Cleanup.
  cudaEventDestroy(start);
  cudaEventDestroy(mid);
  cudaEventDestroy(stop);
  free(file_name);
  fclose(result_file);
  return 0; 
}
