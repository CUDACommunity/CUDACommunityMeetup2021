#include <iostream>

__global__ void kernel (int *data, int *result)
{
  if (threadIdx.x == 0 && blockIdx.x == 110)
    if (data[0])
      result[0] = 1;
}


int main ()
{
  int blocks = 1; // 256;
  int threads = 32; //256;

  int *data = nullptr;
  int *result = nullptr;

  cudaDeviceReset ();

  cudaMalloc (&data, 1024 * sizeof (int));

  kernel<<<blocks, threads>>> (data, result);

  cudaFree (data);
  cudaDeviceReset ();
}
