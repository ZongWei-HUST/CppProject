#include <iostream>
using std::cout;
using std::endl;

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
  *d_c = *d_a + *d_b;
  printf("This is block: %d, thread: %d\n", blockIdx.x, threadIdx.x);
}
