#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#define N 5
#define si sizeof(int)
#define sf sizeof(float)

__global__ void gpuSquare(float *d_in, float *d_out) {
  int tid = threadIdx.x;
  float tmp = d_in[tid];
  d_out[tid] = tmp * tmp;
}

int main() {
  float h_in[N], h_out[N];
  float *d_in, *d_out;
  for (int i = 0; i < N; i++) {
    h_in[i] = i;
  }
  cudaMalloc((void **)&d_in, sf * N);
  cudaMalloc((void **)&d_out, sf * N);
  cudaMemcpy(d_in, h_in, sf * N, cudaMemcpyHostToDevice);
  gpuSquare<<<1, N>>>(d_in, d_out);
  cudaMemcpy(h_out, d_out, sf * N, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    printf("The square of %f is %f\n", h_in[i], h_out[i]);
  }
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}