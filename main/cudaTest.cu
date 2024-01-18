#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

__global__ void hello(){
  printf("Hello from cuda block %d thread %d\n",blockIdx.x,threadIdx.x);
}

__global__ void set_your_element(float *a){
  curandState state;
  curand_init(clock64(),threadIdx.x,0,&state);
  a[threadIdx.x]=curand_uniform(&state);
  printf("This is thread %d generating %f\n",threadIdx.x,a[threadIdx.x]);
}


int main(){
  int n=16;
  float *a;
  printf("Normal print statement\n");
  hello<<<1,16>>>();

  cudaMallocManaged(&a,n*sizeof(float));
  set_your_element<<<1,n>>>(a);
  cudaDeviceSynchronize();
  for(int i=0;i<n;i++){
    printf("%f ",a[i]);
  }
  printf("\n");
  return 0;
}
