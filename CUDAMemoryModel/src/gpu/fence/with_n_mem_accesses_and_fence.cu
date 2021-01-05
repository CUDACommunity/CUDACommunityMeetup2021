#include <iostream>

__global__ void kernel (
    int *data_1,
    int *data_2,
    int *data_3,
    int *data_4,
    int *data_5,
    int *result)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    if (data_1[0] || data_2[0] || data_3[0])
      result[0] = 1;

    __threadfence ();
      
    if (data_4[0] || data_5[0])
      result[0] = 1;
  }
}


int main ()
{
  int blocks = 1; // 256;
  int threads = 32; //256;

  int *data = nullptr;
  int *result = nullptr;

  cudaDeviceReset ();

  cudaMalloc (&data, sizeof (int));

  kernel<<<blocks, threads>>> (data, data, data, data, data, result);

  cudaFree (data);
  cudaDeviceReset ();
}
