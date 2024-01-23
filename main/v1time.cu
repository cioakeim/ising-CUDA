/* Script that provides the execution times for v1 of the ising model. */
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <cuda_runtime.h>
#include "isingV1.h"

#define RUNS_PER_SIZE 5

int main(int argc, char** argv){
  if(argc!=2){
    printf("Usage: ./v1time [resultFolder]\n");
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
  int n_min=5000;
  int n_step=10000;
  int n_max=35000;
  int k_min=30;
  int k_step=50;
  int k_max=80;
  float init_times[RUNS_PER_SIZE];
  float iter_times[RUNS_PER_SIZE];
  float init_time,iter_time;
  cudaEvent_t start,stop,mid;
  cudaEventCreate(&start);
  cudaEventCreate(&mid);
  cudaEventCreate(&stop);
  dim3 blockSize,gridSize;
  cudaError_t cudaError;

  printf("Block size: %d\n",BLOCK_MAX);
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
        cudaEventRecord(mid,0);
        isingV1(G,G0,n,k,blockSize,gridSize);
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
      // Get median for this pair (n,k):
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
    freeGridV1(G);
    freeGridV1(G0);
    printf("Size %d done.\n",n);
  }
  printf("V1 Timing gathered.\n");
  // Cleanup.
  cudaEventDestroy(start);
  cudaEventDestroy(mid);
  cudaEventDestroy(stop);
  free(file_name);
  fclose(result_file);
  return 0; 
}
