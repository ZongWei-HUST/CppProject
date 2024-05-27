#include <iostream>

__global__ void myfirstkkernel(void) {}

int main() {
  myfirstkkernel<<<1, 1>>>();
  printf("Hello, CUDA!\n");
  return 0;
}
