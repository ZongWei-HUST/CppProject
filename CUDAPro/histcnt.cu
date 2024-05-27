#include <cuda_runtime.h>

#include "stdio.h"
#define SIZE 1000
#define NUM_BIN 256

__global__ void histogram_shared_memory(int* d_a, int* d_b) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  __shared__ int cache[NUM_BIN];
  // 每个块的共享内存
  cache[threadIdx.x] = 0;
  __syncthreads();
  while (tid < SIZE) {
    atomicAdd(&cache[d_a[tid]], 1);  // 线程索引==像素值,索引值==像素个数
    tid += offset;
  }
  __syncthreads();
  atomicAdd(&d_b[threadIdx.x], cache[threadIdx.x]);
}

int main() {
  int h_a[SIZE];
  for (int i = 0; i < SIZE; ++i) {
    h_a[i] = i % NUM_BIN;
  }
  int h_b[NUM_BIN];
  for (int i = 0; i < NUM_BIN; ++i) {
    h_b[i] = 0;
  }
  // declare GPU memory pointers
  int* d_a;
  int* d_b;
  cudaMalloc((void**)&d_a, SIZE * sizeof(int));
  cudaMalloc((void**)&d_b, NUM_BIN * sizeof(int));
  cudaMemcpy(d_a, h_a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, NUM_BIN * sizeof(int), cudaMemcpyHostToDevice);
  histogram_shared_memory<<<SIZE / NUM_BIN, NUM_BIN>>>(d_a, d_b);
  cudaMemcpy(h_b, d_b, NUM_BIN * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Histogram using 16 bin is: ");
  for (int i = 0; i < NUM_BIN; i++) {
    printf("bin %d: count %d\n", i, h_b[i]);
  }

  // free GPU memory allocation
  cudaFree(d_a);
  cudaFree(d_b);
}