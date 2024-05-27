#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
using std::cout;
using std::endl;

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
  *d_c = *d_a + *d_b;
  printf("This is block: %d, thread: %d\n", blockIdx.x, threadIdx.x);
}

int main() {
  // 为CPU/GPU分配内存
  int h_c, h_a, h_b;
  int *d_c, *d_a, *d_b;
  h_a = 1;
  h_b = 4;
  cudaMalloc((void**)&d_c, sizeof(int));  // 为GPU分配内存
  cudaMalloc((void**)&d_a, sizeof(int));
  cudaMalloc((void**)&d_b, sizeof(int));
  // 将数据从CPU拷贝到GPU
  cudaMemcpy(d_c, &h_c, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);
  // 内核调用
  gpuAdd<<<8, 2>>>(d_a, d_b, d_c);
  // 将数据从GPU拷贝到CPY
  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("1 + 4 = %d\n", h_c);
  // 释放主机和设备内存
  cudaFree(d_c);
  cudaFree(d_a);
  cudaFree(d_b);
  return 0;
}