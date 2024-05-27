#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define N 50000
#define si sizeof(int)

// void cpuAdd(int* h_a, int* h_b, int* h_c) {
//   int tid = 0;
//   while (tid < N) {
//     h_c[tid] = h_a[tid] + h_b[tid];
//     ++tid;
//   }
// }

// int main() {
//   int h_a[N], h_b[N], h_c[N];
//   for (int i = 0; i < N; ++i) {
//     h_a[i] = i * 2 * 2;
//     h_b[i] = i;
//   }
//   clock_t start_time = clock();
//   cpuAdd(h_a, h_b, h_c);
//   clock_t end_time = clock();
//   double time_d = (double)(end_time - start_time) / CLOCKS_PER_SEC;
//   printf("spend time: %f", time_d);
//   printf("Vector addition on CPU\n");
//   //   for (int i = 0; i < N; i++) {
//   //     printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i],
//   //            h_c[i]);
//   //   }
//   return 0;
// }

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < N) {
    d_c[tid] = d_a[tid] + d_b[tid];
    tid += blockDim.x * gridDim.x;
  }
}

int main() {
  // Initialize
  int h_a[N], h_b[N], h_c[N];
  int *d_a, *d_b, *d_c;
  cudaEvent_t e_start, e_end;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_end);
  cudaEventRecord(e_start, 0);
  cudaMalloc((void**)&d_a, N * si);
  cudaMalloc((void**)&d_b, N * si);
  cudaMalloc((void**)&d_c, N * si);
  for (int i = 0; i < N; i++) {
    h_a[i] = 2 * i * i;
    h_b[i] = i;
  }
  // CPU data -> GPU
  cudaMemcpy(d_a, h_a, N * si, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * si, cudaMemcpyHostToDevice);
  // call kernel
  gpuAdd<<<512, 512>>>(d_a, d_b, d_c);
  // GPU res -> CPU
  cudaMemcpy(h_c, d_c, N * si, cudaMemcpyDeviceToHost);
  // get itme
  cudaDeviceSynchronize();
  cudaEventRecord(e_end, 0);
  cudaEventSynchronize(e_end);
  float time;
  cudaEventElapsedTime(&time, e_start, e_end);
  printf("Time to add %d numbers: %3.1f ms\n", N, time);

  // Free
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  printf("Vector addition on GPU \n");
  // Printing result on console
  //   for (int i = 0; i < N; i++) {
  //     printf("The sum of %d element is %d + %d = %d\n", i, h_a[i], h_b[i],
  //            h_c[i]);
  //   }
}