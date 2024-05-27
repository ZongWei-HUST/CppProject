#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define arraySize 512
#define threadPerBlock 256

__global__ void addKernel(int *d_a, int *d_b) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int val = d_a[tid];
  int count = 0;
  for (int i = 0; i < arraySize; ++i) {
    if (val > d_a[i]) count++;
  }
  d_b[count] = val;
}

int main() {
  //   int h_a[arraySize] = {5, 9, 3, 4, 8};
  int h_a[arraySize];
  int h_b[arraySize];
  int *d_a, *d_b;
  for (int i = 0; i < arraySize; ++i) h_a[i] = i + 1;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(h_a, h_a + arraySize, g);

  // 输出数组
  for (int i = 0; i < arraySize; i++) {
    std::cout << h_a[i] << " ";
  }
  std::cout << std::endl;

  cudaMalloc((void **)&d_b, arraySize * sizeof(int));
  cudaMalloc((void **)&d_a, arraySize * sizeof(int));

  // Copy input vector from host memory to GPU buffers.
  cudaMemcpy(d_a, h_a, arraySize * sizeof(int), cudaMemcpyHostToDevice);

  // Launch a kernel on the GPU with one thread for each element.
  addKernel<<<arraySize / threadPerBlock, threadPerBlock>>>(d_a, d_b);

  cudaDeviceSynchronize();
  // Copy output vector from GPU buffer to host memory.
  cudaMemcpy(h_b, d_b, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
  printf("The Enumeration sorted Array is: \n");
  for (int i = 0; i < arraySize; i++) {
    printf("%d ", h_b[i]);
  }
  printf("\n");
  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}