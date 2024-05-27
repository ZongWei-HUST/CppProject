#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "stdio.h"

#define N 1024
#define threadsPerBlock 512
#define nsf N * sizeof(float)
#define cpu_sum(x) (x * (x + 1))

__global__ void gpu_dot(float* d_a, float* d_b, float* d_c) {
  // Declare shared memory
  __shared__ float partial_sum[threadsPerBlock];
  //   printf("blockDim:%d, gridDim:%d \n", blockDim.x, gridDim.x);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int index = threadIdx.x;
  float sum = 0;
  while (tid < N) {
    sum += d_a[tid] * d_b[tid];
    tid += blockDim.x * gridDim.x;
  }
  //   sum += d_a[tid] * d_b[tid];
  // store partial sum in shared memory
  partial_sum[index] = sum;
  // synchronize threads
  __syncthreads();
  // calculate partial sum for whole block in reduce operation
  int i = blockDim.x / 2;
  while (i != 0) {
    if (index < i) partial_sum[index] += partial_sum[index + i];
    __syncthreads();
    i /= 2;
  }
  // store block partial sum in global memrory
  if (index == 0) {
    d_c[blockIdx.x] = partial_sum[0];
    printf("block %d sum is: %f,\n", blockIdx.x, partial_sum[0]);
  }
}

int main() {
  float *h_a, *h_b, h_sum, *partial_sum;  // Host Array
  float *d_a, *d_b, *d_partial_sum;       // device Array
  int block_calc = (N + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGrid = (block_calc > 32 ? 32 : block_calc);
  h_a = (float*)malloc(nsf);
  h_b = (float*)malloc(nsf);
  partial_sum = (float*)malloc(blocksPerGrid * sizeof(float));
  // allocate the memory on the device
  cudaMalloc((void**)&d_a, nsf);
  cudaMalloc((void**)&d_b, nsf);
  cudaMalloc((void**)&d_partial_sum, blocksPerGrid * sizeof(float));
  // fill the host array with data
  for (int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = 2;
  }
  cudaMemcpy(d_a, h_a, nsf, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nsf, cudaMemcpyHostToDevice);
  gpu_dot<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_partial_sum);
  cudaMemcpy(partial_sum, d_partial_sum, blocksPerGrid * sizeof(float),
             cudaMemcpyDeviceToHost);
  // calculate final dot product on host
  h_sum = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    h_sum += partial_sum[i];
  }
  printf("The computed dot product is: %f\n", h_sum);
  if (h_sum == cpu_sum((float)(N - 1))) {
    printf("The dot product computed by GPU is correct\n");
  } else {
    printf("Error in dot product computation");
  }
  // free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_partial_sum);
  free(h_a);
  free(h_b);
  free(partial_sum);
}