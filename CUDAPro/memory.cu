#include <stdio.h>

__global__ void gpu_shared_memory(float* d_in) {
  int i, index = threadIdx.x;
  float average, sum = 0.0f;
  // 共享内存
  __shared__ float sh_arr[10];
  sh_arr[index] = d_in[index];
  __syncthreads();  // 将所有数据写入共享内存后再操作线程
  for (i = 0; i <= index; ++i) {
    sum += sh_arr[i];
  }
  average = sum / (index + 1.0f);
  d_in[index] = average;
}

int main() {
  float h_a[10];
  float* d_a;
  for (int i = 0; i < 10; i++) {
    h_a[i] = i;
  }
  cudaMalloc((void**)&d_a, 10 * sizeof(float));
  cudaMemcpy(d_a, h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);
  gpu_shared_memory<<<1, 10>>>(d_a);
  cudaMemcpy(h_a, d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
  printf("Use of Shared Memory on GPU:  \n");
  // Printing result on console
  for (int i = 0; i < 10; i++) {
    printf("The running average after %d element is %f \n", i, h_a[i]);
  }
  return 0;
}
