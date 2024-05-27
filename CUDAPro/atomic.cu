#include <stdio.h>

#define NUM_THREADS 10000
#define SIZE 10
#define BLOCK_WIDTH 100

__global__ void gpu_increment_without_atomic(int *d_a) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // 理想情况下d_a每个位置会加10000 / 10次
  tid = tid % SIZE;
  // d_a[tid] += 1; // 不具有原子性,同时被线程访问
  atomicAdd(&d_a[tid], 1);  // 原子操作
}

int main(int argc, char **argv) {
  printf("%d total threads in %d blocks writing into %d array elements\n",
         NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, SIZE);

  int h_a[SIZE];
  const int ARRAY_BYTES = SIZE * sizeof(int);

  int *d_a;
  cudaMalloc((void **)&d_a, ARRAY_BYTES);
  cudaMemset((void *)d_a, 0, ARRAY_BYTES);  // 直接初始化d_a

  gpu_increment_without_atomic<<<NUM_THREADS / BLOCK_WIDTH, BLOCK_WIDTH>>>(d_a);

  // copy back the array to host memory
  cudaMemcpy(h_a, d_a, ARRAY_BYTES, cudaMemcpyDeviceToHost);

  printf(
      "Number of times a particular Array index has been incremented without "
      "atomic add is: \n");
  for (int i = 0; i < SIZE; i++) {
    printf("index: %d --> %d times\n ", i, h_a[i]);
  }
  cudaFree(d_a);
  return 0;
}